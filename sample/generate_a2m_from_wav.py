"""
BEAT_v2 audio-to-motion inference from an arbitrary-length wav file.

Pipeline:
  1) Load wav, resample to 16k.
  2) Slice into 5-second chunks with 1-second overlap.
  3) For each chunk:
       - Extract Whisper(base) encoder features: (1500, 512) for 30s window,
         then take the first 250 frames -> (250, 512) for 5s.
       - Run diffusion to generate 5s motion (default 300 frames at 60fps).
  4) Stitch per-chunk motions with linear cross-fade over the 1s overlap.

Outputs (same style as sample.generate):
  - <out_dir>/results.npy
  - <out_dir>/results.txt
  - <out_dir>/results_len.txt
  - optional: <out_dir>/npz/sample0000.npz with key 'qpos'
"""

import argparse
import json
import os
import shutil
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
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
    """
    a, b: (K, D)
    returns: (K, D) blending from a -> b
    """
    assert a.shape == b.shape and a.ndim == 2
    k = a.shape[0]
    if k == 0:
        return b
    w = np.linspace(0.0, 1.0, num=k, dtype=np.float32)[:, None]
    return (1.0 - w) * a + w * b


def _stitch_with_overlap(segments: List[np.ndarray], overlap: int) -> np.ndarray:
    """
    segments: list of (T, D)
    overlap: number of frames to cross-fade
    """
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
    """
    audio_5s_16k: (5*16000,) float32 in [-1,1]
    returns: (250, 512) encoder features for Whisper(base)
    """
    try:
        import whisper  # openai-whisper
    except Exception as e:
        raise ImportError(
            "Missing dependency 'whisper'. Install with: pip install -U openai-whisper"
        ) from e

    # Cache model singleton on function attribute
    if not hasattr(_whisper_base_encode_features, "_model"):
        _whisper_base_encode_features._model = whisper.load_model("base", device=str(device))
        _whisper_base_encode_features._model.eval()

    model = _whisper_base_encode_features._model

    # Whisper expects 30s window for mel -> encoder. We pad/trim 5s audio into 30s.
    audio_30s = whisper.pad_or_trim(audio_5s_16k, length=whisper.audio.N_SAMPLES).astype(np.float32)
    mel = whisper.log_mel_spectrogram(audio_30s)  # (80, 3000)
    mel = mel.to(device)
    # encoder out: (1, 1500, 512) for base (conv stride 2)
    enc = model.encoder(mel.unsqueeze(0))
    enc = enc[0].detach().float().cpu().numpy()  # (1500, 512)

    # 30s -> 1500 frames, so 5s corresponds to 1500 * (5/30) = 250 frames
    feat_5s = enc[:250]
    if feat_5s.shape != (250, 512):
        raise ValueError(f"Unexpected whisper feature shape: {feat_5s.shape}")
    return feat_5s.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--dataset", default="beat_v2", type=str)
    parser.add_argument("--data_dir", default="", type=str, help="BEAT_v2 root dir containing Mean.npy/Std.npy")
    parser.add_argument("--input_wav", required=True, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_repetitions", default=1, type=int)
    parser.add_argument("--guidance_param", default=1.0, type=float)
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_video", action="store_true", help="Render mp4 videos (requires external/GMR)")
    parser.add_argument(
        "--attach_wav",
        action="store_true",
        help="Mux original wav as audio track into rendered mp4 (requires ffmpeg).",
    )
    parser.add_argument(
        "--vis_script",
        default="external/GMR/scripts/vis_npz_motion.py",
        type=str,
        help="Path to external/GMR/scripts/vis_npz_motion.py",
    )

    # chunking / fps
    parser.add_argument("--chunk_sec", default=5.0, type=float)
    parser.add_argument("--overlap_sec", default=1.0, type=float)
    parser.add_argument("--motion_fps", default=60, type=int)

    # model/diffusion defaults (will be overwritten by args.json when exists)
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
    if args.dataset != "beat_v2":
        raise ValueError(f"generate_a2m_from_wav only supports --dataset beat_v2, got {args.dataset}")

    saved = _load_args_json(args.model_path)
    for k in [
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
    ]:
        if k in saved:
            setattr(args, k, saved[k])

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    dev = dist_util.dev()

    # output dir naming (match sample.generate style)
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    if args.output_dir:
        out_dir = args.output_dir
    else:
        model_dir = os.path.dirname(args.model_path)
        base = os.path.splitext(os.path.basename(args.input_wav))[0]
        out_dir = os.path.join(model_dir, f"samples_{name}_{niter}_seed{args.seed}_{base}")
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
        # fallback to librosa
        try:
            import librosa
        except Exception as e:
            raise ImportError(
                "Need either 'soundfile' or 'librosa' to read wav. "
                "Install: pip install soundfile librosa"
            ) from e
        wav, sr = librosa.load(wav_path, sr=None, mono=True)
        wav = wav.astype(np.float32)

    if sr != 16000:
        try:
            import librosa
        except Exception as e:
            raise ImportError("Resampling requires librosa. Install: pip install librosa") from e
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000

    # chunk wav into 5s windows with 1s overlap
    audio_chunks = _slice_audio(wav, sr, chunk_sec=args.chunk_sec, overlap_sec=args.overlap_sec)
    if len(audio_chunks) == 0:
        raise ValueError("Empty wav")

    # mean/std for de-normalization
    if not args.data_dir:
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "BEAT_v2", "segment_1wayne"))
    else:
        data_root = args.data_dir
    mean, std = _load_mean_std(data_root)  # (60,)
    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)

    # create model/diffusion
    class _Dummy:
        pass

    dummy = _Dummy()
    dummy.dataset = _Dummy()
    model, diffusion = create_model_and_diffusion(args, dummy)
    load_saved_model(model, args.model_path, use_avg=bool(args.use_ema))
    if args.guidance_param != 1:
        if getattr(model, "cond_mask_prob", 0.0) and model.cond_mask_prob > 0:
            model = ClassifierFreeSampleModel(model)
        else:
            print("[Warning] guidance_param>1 requires a model trained with cond_mask_prob>0. "
                  "Falling back to guidance_param=1.0.")
            args.guidance_param = 1.0
    model.to(dev)
    model.eval()

    # per-chunk generation length: 5s * fps
    seg_frames = int(round(args.chunk_sec * args.motion_fps))
    overlap_frames = int(round(args.overlap_sec * args.motion_fps))
    if overlap_frames >= seg_frames:
        raise ValueError("overlap_sec must be smaller than chunk_sec")

    bs = 1
    motion_shape = (bs, model.njoints, model.nfeats, seg_frames)
    lengths = torch.tensor([seg_frames], device=dev)
    mask = lengths_to_mask(lengths, seg_frames).unsqueeze(1).unsqueeze(1)

    # generate per repetition (usually 1)
    final_motions = []
    for rep_i in range(int(args.num_repetitions)):
        print(f"### Sampling repetition #{rep_i}")
        seg_qpos_list: List[np.ndarray] = []
        for ci, chunk in enumerate(audio_chunks):
            # whisper features for this 5s chunk: (250,512)
            whisper_feat = _whisper_base_encode_features(chunk, device=dev)  # np (250,512)
            model_kwargs = {
                "y": {
                    "mask": mask,
                    "lengths": lengths,
                    "audio": torch.from_numpy(whisper_feat).to(dev).unsqueeze(0),  # [1,250,512]
                }
            }
            if args.guidance_param != 1:
                model_kwargs["y"]["scale"] = torch.ones(bs, device=dev) * float(args.guidance_param)

            sample = diffusion.p_sample_loop(
                model,
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
            print(f"  chunk {ci+1}/{len(audio_chunks)} -> motion {qpos.shape}")

        stitched = _stitch_with_overlap(seg_qpos_list, overlap=overlap_frames)  # (T_total,60)
        final_motions.append(stitched)

    # save (match sample.generate keys)
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
        # Video rendering relies on the NPZ visualization script.
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
            raise FileNotFoundError(
                f"Requested --save_video/--attach_wav but visualization script not found: {vis_script}"
            )

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
                # Mux original wav as audio. Re-encode audio to AAC for mp4 compatibility.
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

