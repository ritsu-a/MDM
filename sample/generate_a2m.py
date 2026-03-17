"""
BEAT_v2 audio-to-motion inference.

Input: a single BEAT_v2 .npz with keys:
  - whisper: (T_a, 512)
  - (optional) motion: (T_m, 60)  # not used for generation, only for reference

Output:
  - results.npy: a dict with keys {'motion','text','lengths',...} (same shape as sample.generate)
  - npz/sampleXXXX.npz: per-sample npz (same key name as t2m: 'qpos')
"""

import argparse
import json
import os
from os.path import join as pjoin

import numpy as np
import torch

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.tensors import lengths_to_mask
from data_loaders.beat_v2_dataset import AUDIO_MAX_LEN, compute_mean_std


def _load_args_json(model_path: str) -> dict:
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.isfile(args_path):
        return {}
    with open(args_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_mean_std(data_root: str):
    mean_path = pjoin(data_root, "Mean.npy")
    std_path = pjoin(data_root, "Std.npy")
    if os.path.isfile(mean_path) and os.path.isfile(std_path):
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        return mean, std
    # fallback: compute from train split
    return compute_mean_std(data_root, pjoin(data_root, "train.txt"))


def _prepare_whisper(whisper: np.ndarray) -> np.ndarray:
    if whisper.ndim != 2 or whisper.shape[1] != 512:
        raise ValueError(f"Expected whisper shape (T,512), got {whisper.shape}")
    T = whisper.shape[0]
    if T >= AUDIO_MAX_LEN:
        whisper_cut = whisper[:AUDIO_MAX_LEN]
    else:
        pad = np.zeros((AUDIO_MAX_LEN - T, 512), dtype=np.float32)
        whisper_cut = np.concatenate([whisper.astype(np.float32), pad], axis=0)
    return whisper_cut.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    # mirror sample.generate's "sampling" options naming for consistency
    parser.add_argument("--model_path", required=True, type=str, help="Path to model####.pt")
    parser.add_argument("--dataset", default="beat_v2", type=str, help="Must be beat_v2")
    parser.add_argument("--data_dir", default="", type=str, help="BEAT_v2 root dir containing train.txt/val.txt/Mean.npy/Std.npy")
    parser.add_argument("--input_npz", required=True, type=str, help="A single BEAT_v2 val .npz path")
    parser.add_argument("--output_dir", default="", type=str, help="Path to results dir (auto created if empty).")
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_repetitions", default=1, type=int)
    parser.add_argument("--guidance_param", default=1.0, type=float, help="CFG scale; use 1.0 to disable")
    parser.add_argument("--motion_frames", default=300, type=int, help="Number of motion frames to generate")
    parser.add_argument("--save_npz", action="store_true", help="If true, save per-sample .npz files under output_dir/npz")

    # keep consistent with training defaults (will be overwritten by args.json next to model_path when exists)
    parser.add_argument("--arch", default="trans_enc", type=str, choices=["trans_enc", "trans_dec", "gru"])
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--cond_mask_prob", default=0.0, type=float)
    parser.add_argument("--pos_embed_max_len", default=5000, type=int)
    parser.add_argument("--mask_frames", action="store_true")
    parser.add_argument("--emb_trans_dec", action="store_true")
    parser.add_argument("--text_encoder_type", default="clip", type=str, choices=["clip", "bert"])

    # diffusion defaults (will be overwritten by args.json when exists)
    parser.add_argument("--noise_schedule", default="cosine", choices=["linear", "cosine"], type=str)
    parser.add_argument("--diffusion_steps", default=1000, type=int)
    parser.add_argument("--sigma_small", default=True, type=bool)
    parser.add_argument("--lambda_vel", default=0.0, type=float)
    parser.add_argument("--lambda_rcxyz", default=0.0, type=float)
    parser.add_argument("--lambda_fc", default=0.0, type=float)
    parser.add_argument("--lambda_target_loc", default=0.0, type=float)

    # compatibility flags referenced in codepaths
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--unconstrained", action="store_true")

    args = parser.parse_args()
    if args.dataset != "beat_v2":
        raise ValueError(f"generate_a2m only supports --dataset beat_v2, got {args.dataset}")

    # overwrite model/diffusion args from the checkpoint's args.json (same behavior as sample.generate)
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

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    if args.output_dir:
        out_dir = args.output_dir
    else:
        model_dir = os.path.dirname(args.model_path)
        out_dir = os.path.join(model_dir, f"samples_{name}_{niter}_seed{args.seed}")
        out_dir += "_" + os.path.splitext(os.path.basename(args.input_npz))[0]

    # match sample.generate behavior: re-create output dir
    if os.path.exists(out_dir):
        # keep it simple: remove existing dir tree
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    npz = np.load(args.input_npz, allow_pickle=True)
    if "whisper" not in npz:
        raise KeyError(f"Expected key 'whisper' in {args.input_npz}, got {list(npz.keys())}")
    whisper = np.asarray(npz["whisper"], dtype=np.float32)
    whisper_used = _prepare_whisper(whisper)  # (AUDIO_MAX_LEN, 512)

    # load mean/std for de-normalization
    if not args.data_dir:
        # default layout consistent with BeatV2 wrapper
        data_root = os.path.join(os.path.dirname(__file__), "..", "data", "BEAT_v2", "segment_1wayne")
        data_root = os.path.abspath(data_root)
    else:
        data_root = args.data_dir
    mean, std = _load_mean_std(data_root)  # (60,), (60,)
    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)

    # create model/diffusion (create_model_and_diffusion 只依赖 data.dataset 是否含 num_actions)
    class _Dummy:
        pass
    dummy = _Dummy()
    dummy.dataset = _Dummy()

    model, diffusion = create_model_and_diffusion(args, dummy)
    load_saved_model(model, args.model_path, use_avg=False)
    if args.guidance_param != 1:
        # CFG requires cond_mask_prob > 0 during training
        if getattr(model, "cond_mask_prob", 0.0) and model.cond_mask_prob > 0:
            model = ClassifierFreeSampleModel(model)
        else:
            print("[Warning] guidance_param>1 requires a model trained with cond_mask_prob>0. "
                  "Falling back to guidance_param=1.0.")
            args.guidance_param = 1.0
    model.to(dist_util.dev())
    model.eval()

    bs = 1
    n_frames = int(args.motion_frames)
    motion_shape = (bs, model.njoints, model.nfeats, n_frames)

    lengths = torch.tensor([n_frames], device=dist_util.dev())
    mask = lengths_to_mask(lengths, n_frames).unsqueeze(1).unsqueeze(1)  # [bs,1,1,T]
    model_kwargs = {
        "y": {
            "mask": mask,
            "lengths": lengths,
            "audio": torch.from_numpy(whisper_used).to(dist_util.dev()).unsqueeze(0),  # [1, Ta, 512]
        }
    }
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(bs, device=dist_util.dev()) * float(args.guidance_param)

    all_motions = []
    all_lengths = []
    all_text = []
    for rep_i in range(int(args.num_repetitions)):
        print(f"### Sampling repetition #{rep_i}")
        sample = diffusion.p_sample_loop(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        # sample: [bs, 60, 1, T] because njoints=60,nfeats=1 (hml_vec)
        # sample: [bs, 60, 1, T]
        pred_norm = sample.detach().cpu().permute(0, 3, 1, 2).numpy()  # [B,T,60,1]
        pred_norm = np.squeeze(pred_norm, axis=-1)  # [B,T,60]
        pred = pred_norm * std + mean  # [B,T,60] de-normalized

        # sample.generate uses 'qpos' in original feature space
        qpos = pred.astype(np.float32)
        all_motions.append(qpos)
        all_lengths.append(np.array([n_frames], dtype=np.int64))
        all_text.append(os.path.abspath(args.input_npz))

    all_motions = np.concatenate(all_motions, axis=0)  # [R, T, 60]
    all_lengths = np.concatenate(all_lengths, axis=0)

    # results.npy (match sample.generate keys)
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

    # per-sample npz (match sample.generate naming and directory)
    if args.save_npz:
        npz_dir = os.path.join(out_dir, "npz")
        os.makedirs(npz_dir, exist_ok=True)
        for i in range(all_motions.shape[0]):
            out_npz = os.path.join(npz_dir, f"sample{i:04d}.npz")
            np.savez(
                out_npz,
                qpos=all_motions[i],
                whisper_used=whisper_used.astype(np.float32),
                input_npz=os.path.abspath(args.input_npz),
            )
            print(f"Saved: {out_npz}")

    print(f"[Done] Results are at [{os.path.abspath(out_dir)}]")


if __name__ == "__main__":
    main()

