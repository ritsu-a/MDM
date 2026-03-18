"""
BEAT_v2 audio+text to motion inference from an arbitrary-length wav file.

Goal:
  During diffusion sampling, combine two diffusion models' noise (epsilon) predictions:
    - a2m model: conditioned on audio (Whisper features from wav)
    - t2m model: conditioned on text prompt (per 5-second chunk; can be empty)

At each diffusion step:
  eps = w_audio * eps_cfg_audio + w_text * eps_cfg_text
where eps_cfg_* is classifier-free guidance (CFG) noise:
  eps_cfg = eps_uncond + scale * (eps_cond - eps_uncond)

Chunking / stitching:
  - Slice wav into 5-second chunks with 1-second overlap.
  - Generate motion per chunk (default 5s -> 300 frames at 60fps).
  - Stitch per-chunk motions with linear cross-fade over the overlap frames.

Outputs:
  - <out_dir>/results.npy (same key style as other sample scripts)
  - optional: <out_dir>/npz/sample0000.npz with key 'qpos'
  - optional: <out_dir>/videos/sample0000.mp4 and sample0000_with_wav.mp4
"""

import argparse
import json
import os
import shutil
import subprocess
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from data_loaders.tensors import lengths_to_mask
from data_loaders.beat_v2_dataset import compute_mean_std


def _load_args_json(model_path: str) -> dict:
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.isfile(args_path):
        return {}
    with open(args_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_mean_std(data_root: str):
    mean_path = os.path.join(data_root, "Mean.npy")
    std_path = os.path.join(data_root, "Std.npy")
    if os.path.isfile(mean_path) and os.path.isfile(std_path):
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        return mean, std
    return compute_mean_std(data_root, os.path.join(data_root, "train.txt"))


def _linear_crossfade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape and a.ndim == 2
    k = a.shape[0]
    if k == 0:
        return b
    w = np.linspace(0.0, 1.0, num=k, dtype=np.float32)[:, None]
    return (1.0 - w) * a + w * b


def _stitch_with_overlap(segments: List[np.ndarray], overlap: int) -> np.ndarray:
    if not segments:
        raise ValueError("No segments to stitch")
    out = segments[0].copy()
    for seg in segments[1:]:
        if overlap <= 0:
            out = np.concatenate([out, seg], axis=0)
            continue
        if out.shape[0] < overlap or seg.shape[0] < overlap:
            raise ValueError(f"Overlap too large: overlap={overlap}, prev={out.shape}, next={seg.shape}")
        blended = _linear_crossfade(out[-overlap:], seg[:overlap])
        out = np.concatenate([out[:-overlap], blended, seg[overlap:]], axis=0)
    return out


def _slice_audio(wav: np.ndarray, sr: int, chunk_sec: float, overlap_sec: float) -> List[np.ndarray]:
    chunk = int(round(chunk_sec * sr))
    overlap = int(round(overlap_sec * sr))
    step = chunk - overlap
    if step <= 0:
        raise ValueError("chunk_sec must be larger than overlap_sec")
    if wav.ndim != 1:
        raise ValueError(f"Expected mono wav 1D, got {wav.shape}")
    n = wav.shape[0]
    chunks = []
    start = 0
    while start < n:
        end = start + chunk
        cur = wav[start:end]
        if cur.shape[0] < chunk:
            pad = np.zeros((chunk - cur.shape[0],), dtype=cur.dtype)
            cur = np.concatenate([cur, pad], axis=0)
        chunks.append(cur)
        if end >= n:
            break
        start += step
    return chunks


@torch.no_grad()
def _whisper_base_encode_features(audio_5s_16k: np.ndarray, device: torch.device) -> np.ndarray:
    try:
        import whisper  # openai-whisper
    except Exception as e:
        raise ImportError("Missing dependency 'whisper'. Install with: pip install -U openai-whisper") from e

    if not hasattr(_whisper_base_encode_features, "_model"):
        _whisper_base_encode_features._model = whisper.load_model("base", device=str(device))
        _whisper_base_encode_features._model.eval()

    model = _whisper_base_encode_features._model
    audio_30s = whisper.pad_or_trim(audio_5s_16k, length=whisper.audio.N_SAMPLES).astype(np.float32)
    mel = whisper.log_mel_spectrogram(audio_30s).to(device)
    enc = model.encoder(mel.unsqueeze(0))[0].detach().float().cpu().numpy()  # (1500, 512)
    feat_5s = enc[:250]
    if feat_5s.shape != (250, 512):
        raise ValueError(f"Unexpected whisper feature shape: {feat_5s.shape}")
    return feat_5s.astype(np.float32)


@torch.no_grad()
def _whisper_base_transcribe(audio_5s_16k: np.ndarray, device: torch.device) -> str:
    """
    ASR for a single 5s chunk using Whisper(base).
    Returns a plain text string (may be empty).
    """
    try:
        import whisper  # openai-whisper
    except Exception as e:
        raise ImportError("Missing dependency 'whisper'. Install with: pip install -U openai-whisper") from e

    # Reuse the same singleton model loaded in feature extractor, if available.
    if hasattr(_whisper_base_encode_features, "_model"):
        model = _whisper_base_encode_features._model
    else:
        model = whisper.load_model("base", device=str(device))
        model.eval()

    audio_30s = whisper.pad_or_trim(audio_5s_16k, length=whisper.audio.N_SAMPLES).astype(np.float32)
    mel = whisper.log_mel_spectrogram(audio_30s).to(device)
    options = whisper.DecodingOptions(without_timestamps=True)
    result = whisper.decode(model, mel, options)
    return (result.text or "").strip()


def _default_prompts_path(input_wav: str) -> str:
    d = os.path.dirname(os.path.abspath(input_wav))
    base = os.path.splitext(os.path.basename(input_wav))[0]
    return os.path.join(d, f"{base}.txt")


def _load_prompts(prompts_path: str) -> List[str]:
    if not prompts_path:
        return []
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(prompts_path)
    with open(prompts_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f]

    def _extract_bracket_prompt(s: str) -> Optional[str]:
        # Take the last [...] block to allow ASR text to contain brackets.
        lb = s.rfind("[")
        rb = s.rfind("]")
        if lb != -1 and rb != -1 and rb > lb:
            return s[lb + 1 : rb].strip()
        return None

    prompts: List[str] = []
    for ln in raw_lines:
        if not ln:
            # empty line => empty prompt
            prompts.append("")
            continue
        if ln.lstrip().startswith("#"):
            # comment line: skip
            continue
        # Preferred format: prompt inside the last [ ... ] block
        bp = _extract_bracket_prompt(ln)
        if bp is not None:
            prompts.append(bp)
            continue
        # Supported formats:
        #  1) Plain: <prompt>
        #  2) TSV: chunk_id \t time \t asr \t prompt
        parts = ln.split("\t")
        if len(parts) >= 4:
            prompts.append(parts[3].strip())
        else:
            prompts.append(ln)
    return prompts


class WeightedDualCFGModel(nn.Module):
    """
    A sampling-only wrapper: returns combined epsilon prediction for diffusion.
    """

    def __init__(
        self,
        t2m_model: nn.Module,
        a2m_model: nn.Module,
        guidance_text: float,
        guidance_audio: float,
        weight_text: float,
        weight_audio: float,
    ):
        super().__init__()
        self.t2m_model = t2m_model
        self.a2m_model = a2m_model
        self.guidance_text = float(guidance_text)
        self.guidance_audio = float(guidance_audio)
        self.weight_text = float(weight_text)
        self.weight_audio = float(weight_audio)

        # expose fields used by diffusion / render utilities
        for attr in ["rot2xyz", "translation", "njoints", "nfeats", "data_rep", "cond_mode", "text_encoder_type"]:
            if hasattr(self.a2m_model, attr):
                setattr(self, attr, getattr(self.a2m_model, attr))
            elif hasattr(self.t2m_model, attr):
                setattr(self, attr, getattr(self.t2m_model, attr))

    def encode_text(self, texts):
        # gaussian_diffusion caches text embeddings via model.encode_text()
        if not hasattr(self.t2m_model, "encode_text"):
            raise AttributeError("t2m_model has no encode_text()")
        return self.t2m_model.encode_text(texts)

    def _cfg_eps(self, model: nn.Module, x, t, y: dict, scale: float, cond_enabled: bool):
        if not cond_enabled:
            y_u = deepcopy(y)
            y_u["uncond"] = True
            return model(x, t, y_u)
        if scale == 1.0:
            return model(x, t, y)
        # if scale != 1, do CFG by explicit 2-pass (cond + uncond)
        y_u = deepcopy(y)
        y_u["uncond"] = True
        eps_cond = model(x, t, y)
        eps_uncond = model(x, t, y_u)
        return eps_uncond + (scale * (eps_cond - eps_uncond))

    def forward(self, x, timesteps, y=None):
        if y is None:
            raise ValueError("Expected model_kwargs['y'] for conditioning.")

        # shared fields
        mask = y["mask"]
        lengths = y["lengths"]

        # audio branch (always enabled if audio exists)
        audio = y.get("audio", None)
        if audio is None:
            raise KeyError("Missing y['audio'] for a2m branch.")
        y_audio = {"mask": mask, "lengths": lengths, "audio": audio}
        eps_audio = self._cfg_eps(
            self.a2m_model, x, timesteps, y_audio, scale=self.guidance_audio, cond_enabled=True
        )

        # text branch (can be disabled per-chunk when prompt is empty)
        texts = y.get("text", None)
        text_embed = y.get("text_embed", None)
        text_enabled = bool(y.get("text_enabled", True))
        if texts is None:
            # treat as disabled text
            text_enabled = False
        if not text_enabled:
            # IMPORTANT: if prompt is empty, do NOT inject t2m's unconditional eps.
            # Otherwise the result will differ from pure a2m even with empty prompts.
            eps_text = torch.zeros_like(eps_audio)
        else:
            y_text = {"mask": mask, "lengths": lengths}
            if texts is not None:
                y_text["text"] = texts
            if text_embed is not None:
                y_text["text_embed"] = text_embed
            eps_text = self._cfg_eps(
                self.t2m_model, x, timesteps, y_text, scale=self.guidance_text, cond_enabled=True
            )

        wt = self.weight_text
        wa = self.weight_audio
        return (wa * eps_audio) + (wt * eps_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_model_path", required=True, type=str)
    parser.add_argument("--t2m_model_path", required=True, type=str)
    # a2m and t2m can come from different dataset configs, but must share motion shape + diffusion config.
    parser.add_argument("--a2m_dataset", default="beat_v2", type=str)
    parser.add_argument("--t2m_dataset", default="motion_stat_300", type=str)
    parser.add_argument("--data_dir", default="", type=str, help="BEAT_v2 root dir containing Mean.npy/Std.npy (used for de-normalization)")
    parser.add_argument("--input_wav", required=True, type=str)
    parser.add_argument("--prompts_path", default="", type=str, help="One prompt per line, aligned to 5s chunks; can be empty lines.")
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_repetitions", default=1, type=int)

    # CFG / weights
    parser.add_argument("--guidance_audio", default=1.0, type=float)
    parser.add_argument("--guidance_text", default=1.0, type=float)
    parser.add_argument("--weight_audio", default=1.0, type=float)
    parser.add_argument("--weight_text", default=1.0, type=float)

    # chunking / fps
    parser.add_argument("--chunk_sec", default=5.0, type=float)
    parser.add_argument("--overlap_sec", default=1.0, type=float)
    parser.add_argument("--motion_fps", default=60, type=int)

    # output options
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_video", action="store_true", help="Render mp4 videos (requires external/GMR)")
    parser.add_argument("--attach_wav", action="store_true", help="Mux original wav into rendered mp4 (requires ffmpeg).")
    parser.add_argument("--vis_script", default="external/GMR/scripts/vis_npz_motion.py", type=str)

    # model/diffusion defaults (overwritten by args.json next to each checkpoint)
    parser.add_argument("--arch", default="trans_enc", type=str, choices=["trans_enc", "trans_dec", "gru"])
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--cond_mask_prob", default=0.0, type=float)
    parser.add_argument("--pos_embed_max_len", default=5000, type=int)
    parser.add_argument("--mask_frames", action="store_true")
    parser.add_argument("--emb_trans_dec", action="store_true")
    parser.add_argument("--text_encoder_type", default="clip", type=str, choices=["clip", "bert"])
    parser.add_argument("--noise_schedule", default="cosine", choices=["linear", "cosine"], type=str)
    parser.add_argument("--diffusion_steps", default=1000, type=int)
    parser.add_argument("--sigma_small", default=True, type=bool)
    parser.add_argument("--lambda_vel", default=0.0, type=float)
    parser.add_argument("--lambda_rcxyz", default=0.0, type=float)
    parser.add_argument("--lambda_fc", default=0.0, type=float)
    parser.add_argument("--lambda_target_loc", default=0.0, type=float)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--unconstrained", action="store_true")

    args = parser.parse_args()
    if args.a2m_dataset != "beat_v2":
        raise ValueError(
            f"generate_a2m_t2m_from_wav currently requires --a2m_dataset beat_v2 (audio pipeline is BEAT_v2-specific), got {args.a2m_dataset}"
        )

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    dev = dist_util.dev()

    # Load per-model args from args.json
    saved_a = _load_args_json(args.a2m_model_path)
    saved_t = _load_args_json(args.t2m_model_path)
    overwrite_keys = [
        "dataset",
        "arch",
        "latent_dim",
        "layers",
        "cond_mask_prob",
        "pos_embed_max_len",
        "mask_frames",
        "emb_trans_dec",
        "text_encoder_type",
        "noise_schedule",
        "diffusion_steps",
        "sigma_small",
        "lambda_vel",
        "lambda_rcxyz",
        "lambda_fc",
        "lambda_target_loc",
        "use_ema",
        "unconstrained",
    ]

    class _Dummy:
        pass

    dummy = _Dummy()
    dummy.dataset = _Dummy()

    # build a2m model/diffusion (we will use its diffusion config)
    args_a = deepcopy(args)
    args_a.dataset = args.a2m_dataset
    for k in overwrite_keys:
        if k in saved_a:
            setattr(args_a, k, saved_a[k])
    model_a, diffusion = create_model_and_diffusion(args_a, dummy)
    load_saved_model(model_a, args.a2m_model_path, use_avg=bool(args.use_ema))
    model_a.to(dev)
    model_a.eval()

    # build t2m model (must match diffusion config)
    args_t = deepcopy(args)
    args_t.dataset = args.t2m_dataset
    for k in overwrite_keys:
        if k in saved_t:
            setattr(args_t, k, saved_t[k])
    model_t, diffusion_t = create_model_and_diffusion(args_t, dummy)
    load_saved_model(model_t, args.t2m_model_path, use_avg=bool(args.use_ema))
    model_t.to(dev)
    model_t.eval()

    # sanity checks: diffusion schedules & output shapes must match
    if diffusion.num_timesteps != diffusion_t.num_timesteps:
        raise ValueError(
            f"Diffusion steps mismatch: a2m={diffusion.num_timesteps}, t2m={diffusion_t.num_timesteps}. "
            "You can only combine models trained with identical diffusion configs."
        )
    if (model_a.njoints != model_t.njoints) or (model_a.nfeats != model_t.nfeats):
        raise ValueError(
            f"Motion shape mismatch: a2m (njoints={model_a.njoints}, nfeats={model_a.nfeats}) vs "
            f"t2m (njoints={model_t.njoints}, nfeats={model_t.nfeats})."
        )

    # guidance scaling requires cond_mask_prob>0 in training for each branch if scale != 1
    if args.guidance_audio != 1.0 and not (getattr(model_a, "cond_mask_prob", 0.0) and model_a.cond_mask_prob > 0):
        print("[Warning] guidance_audio!=1 requires a2m model trained with cond_mask_prob>0; falling back to 1.0")
        args.guidance_audio = 1.0
    if args.guidance_text != 1.0 and not (getattr(model_t, "cond_mask_prob", 0.0) and model_t.cond_mask_prob > 0):
        print("[Warning] guidance_text!=1 requires t2m model trained with cond_mask_prob>0; falling back to 1.0")
        args.guidance_text = 1.0

    combo_model = WeightedDualCFGModel(
        t2m_model=model_t,
        a2m_model=model_a,
        guidance_text=args.guidance_text,
        guidance_audio=args.guidance_audio,
        weight_text=args.weight_text,
        weight_audio=args.weight_audio,
    ).to(dev)
    combo_model.eval()

    # output dir naming (match other sample scripts)
    if args.output_dir:
        out_dir = args.output_dir
    else:
        base = os.path.splitext(os.path.basename(args.input_wav))[0]
        a_name = os.path.basename(os.path.dirname(args.a2m_model_path))
        a_iter = os.path.basename(args.a2m_model_path).replace("model", "").replace(".pt", "")
        t_name = os.path.basename(os.path.dirname(args.t2m_model_path))
        t_iter = os.path.basename(args.t2m_model_path).replace("model", "").replace(".pt", "")
        out_dir = os.path.join(
            os.path.dirname(args.a2m_model_path),
            f"samples_dual_{a_name}_{a_iter}__{t_name}_{t_iter}_seed{args.seed}_{base}",
        )
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # load wav (resample to 16k) using soundfile + librosa (fallback)
    wav_path = args.input_wav
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)
    try:
        import soundfile as sf

        wav, sr = sf.read(wav_path, always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
    except Exception:
        try:
            import librosa
        except Exception as e:
            raise ImportError("Need either 'soundfile' or 'librosa' to read wav. Install: pip install soundfile librosa") from e
        wav, sr = librosa.load(wav_path, sr=None, mono=True)
        wav = wav.astype(np.float32)

    if sr != 16000:
        try:
            import librosa
        except Exception as e:
            raise ImportError("Resampling requires librosa. Install: pip install librosa") from e
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000

    audio_chunks = _slice_audio(wav, sr, chunk_sec=args.chunk_sec, overlap_sec=args.overlap_sec)
    if len(audio_chunks) == 0:
        raise ValueError("Empty wav")

    # If prompts_path is not given, default to "<wav_dir>/<wav_base>.txt"
    if not args.prompts_path:
        args.prompts_path = _default_prompts_path(args.input_wav)

    # First-run UX: if prompt file doesn't exist, create a single template file
    # that includes ASR reference + editable prompt column, then exit.
    if not os.path.isfile(args.prompts_path):
        os.makedirs(os.path.dirname(os.path.abspath(args.prompts_path)), exist_ok=True)

        with open(args.prompts_path, "w", encoding="utf-8") as f:
            f.write("# One line per 5s chunk. Put your prompt INSIDE the brackets: [ ... ]\n")
            f.write("# Empty brackets [] => disable text conditioning for that chunk.\n")
            f.write("# You may keep the left columns (id/time/asr) for reference; the loader extracts the last [ ... ] block.\n")
            step_sec = float(args.chunk_sec - args.overlap_sec)
            for i, chunk in enumerate(audio_chunks):
                start_s = i * step_sec
                end_s = start_s + float(args.chunk_sec)
                asr = _whisper_base_transcribe(chunk, device=dev)
                f.write(f"{i:04d}\t{start_s:.2f}-{end_s:.2f}s\t{asr}\t[]\n")

        print(
            "[Init] Prompt file not found. Created a prompt template with ASR reference:\n"
            f"  prompts: {os.path.abspath(args.prompts_path)}\n"
            "Edit prompts inside brackets [ ... ] and re-run."
        )
        return

    prompts = _load_prompts(args.prompts_path)
    # align prompt count to chunks: pad with empty prompts
    if len(prompts) < len(audio_chunks):
        prompts = prompts + [""] * (len(audio_chunks) - len(prompts))

    # mean/std for de-normalization (assume shared representation)
    if not args.data_dir:
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "BEAT_v2", "segment_1wayne"))
    else:
        data_root = args.data_dir
    mean, std = _load_mean_std(data_root)
    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)

    seg_frames = int(round(args.chunk_sec * args.motion_fps))
    overlap_frames = int(round(args.overlap_sec * args.motion_fps))
    if overlap_frames >= seg_frames:
        raise ValueError("overlap_sec must be smaller than chunk_sec")

    bs = 1
    motion_shape = (bs, model_a.njoints, model_a.nfeats, seg_frames)
    lengths = torch.tensor([seg_frames], device=dev)
    mask = lengths_to_mask(lengths, seg_frames).unsqueeze(1).unsqueeze(1)

    final_motions = []
    for rep_i in range(int(args.num_repetitions)):
        print(f"### Sampling repetition #{rep_i}")
        seg_qpos_list: List[np.ndarray] = []
        for ci, chunk in enumerate(audio_chunks):
            whisper_feat = _whisper_base_encode_features(chunk, device=dev)  # (250,512)
            prompt = prompts[ci] if ci < len(prompts) else ""
            prompt_stripped = (prompt or "").strip()
            text_enabled = len(prompt_stripped) > 0

            model_kwargs = {
                "y": {
                    "mask": mask,
                    "lengths": lengths,
                    "audio": torch.from_numpy(whisper_feat).to(dev).unsqueeze(0),  # [1,250,512]
                    "text_enabled": text_enabled,
                }
            }
            # Only pass 'text' when enabled, to avoid pointless text encoding.
            if text_enabled:
                model_kwargs["y"]["text"] = [prompt_stripped]

            sample = diffusion.p_sample_loop(
                combo_model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            pred_norm = sample.detach().cpu().permute(0, 3, 1, 2).numpy()  # [1,T,60,1]
            pred_norm = np.squeeze(pred_norm, axis=-1)  # [1,T,60]
            pred = pred_norm * std + mean  # [1,T,60]
            qpos = pred[0].astype(np.float32)  # (T,60)
            seg_qpos_list.append(qpos)
            print(f"  chunk {ci+1}/{len(audio_chunks)} prompt={'<empty>' if not text_enabled else 'yes'} -> motion {qpos.shape}")

        stitched = _stitch_with_overlap(seg_qpos_list, overlap=overlap_frames)
        final_motions.append(stitched)

    all_motions = np.stack(final_motions, axis=0)  # [R, T_total, 60]
    all_text = [os.path.abspath(args.input_wav)] * all_motions.shape[0]
    all_lengths = np.array([all_motions.shape[1]] * all_motions.shape[0], dtype=np.int64)

    npy_path = os.path.join(out_dir, "results.npy")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": all_lengths,
            "num_samples": 1,
            "num_repetitions": int(args.num_repetitions),
        },
    )
    with open(npy_path.replace(".npy", ".txt"), "w", encoding="utf-8") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w", encoding="utf-8") as fw:
        fw.write("\n".join([str(int(l)) for l in all_lengths]))

    if args.save_npz:
        npz_dir = os.path.join(out_dir, "npz")
        os.makedirs(npz_dir, exist_ok=True)
        for i in range(all_motions.shape[0]):
            out_npz = os.path.join(npz_dir, f"sample{i:04d}.npz")
            np.savez(out_npz, qpos=all_motions[i])
            print(f"Saved: {out_npz}")

    if args.save_video or args.attach_wav:
        if not args.save_npz:
            args.save_npz = True
            npz_dir = os.path.join(out_dir, "npz")
            os.makedirs(npz_dir, exist_ok=True)
            for i in range(all_motions.shape[0]):
                out_npz = os.path.join(npz_dir, f"sample{i:04d}.npz")
                np.savez(out_npz, qpos=all_motions[i])
                print(f"Saved: {out_npz}")

        vis_script = os.path.abspath(args.vis_script)
        if not os.path.isfile(vis_script):
            raise FileNotFoundError(f"Requested --save_video/--attach_wav but visualization script not found: {vis_script}")
        video_dir = os.path.join(out_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        for i in range(all_motions.shape[0]):
            in_npz = os.path.join(out_dir, "npz", f"sample{i:04d}.npz")
            out_mp4 = os.path.join(video_dir, f"sample{i:04d}.mp4")
            cmd = [
                "python",
                vis_script,
                "--npz_path",
                in_npz,
                "--video_path",
                out_mp4,
                "--motion_fps",
                str(int(args.motion_fps)),
            ]
            print("Render video:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            if args.attach_wav:
                if shutil.which("ffmpeg") is None:
                    print("[Warning] ffmpeg not found; skipping audio mux.")
                    continue
                out_mp4_audio = os.path.join(video_dir, f"sample{i:04d}_with_wav.mp4")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    out_mp4,
                    "-i",
                    os.path.abspath(args.input_wav),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    out_mp4_audio,
                ]
                print("Mux wav audio:", " ".join(ffmpeg_cmd))
                subprocess.run(ffmpeg_cmd, check=True)
                print(f"Saved: {out_mp4_audio}")

    print(f"[Done] Results are at [{os.path.abspath(out_dir)}]")


if __name__ == "__main__":
    main()

