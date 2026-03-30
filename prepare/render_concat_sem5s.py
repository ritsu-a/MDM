#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from typing import List

import numpy as np


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _ensure_qpos_npz(src_npz: str, dst_npz: str) -> str:
    """
    vis_npz_motion.py expects np.load(npz)['qpos'].
    Our sem5s segments are saved with keys: motion/whisper.
    This function creates a compatible npz at dst_npz and returns its path.
    """
    z = np.load(src_npz, allow_pickle=True)
    if "qpos" in z:
        return src_npz
    if "motion" not in z:
        raise KeyError(f"Expected key 'qpos' or 'motion' in {src_npz}, got {list(z.keys())}")
    qpos = np.asarray(z["motion"], dtype=np.float32)
    if qpos.ndim != 2:
        raise ValueError(f"Expected motion/qpos shape (T,D), got {qpos.shape} from {src_npz}")
    os.makedirs(os.path.dirname(os.path.abspath(dst_npz)), exist_ok=True)
    np.savez(dst_npz, qpos=qpos)
    return dst_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segments_dir",
        type=str,
        default=os.path.join("data", "BEAT_v2", "segment_sem5s", "segments"),
        help="包含 *.npz 与同名 *.wav 的目录",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("semi_videos", "sem5s"),
        help="输出目录（每段 mp4 与 concat.txt）",
    )
    ap.add_argument(
        "--vis_script",
        type=str,
        default=os.path.join("external", "GMR", "scripts", "vis_npz_motion.py"),
        help="渲染脚本路径",
    )
    ap.add_argument(
        "--robot",
        type=str,
        default="g1_brainco",
        help="传给 vis_npz_motion.py 的 --robot（默认 g1_brainco）",
    )
    ap.add_argument("--motion_fps", type=int, default=60)
    ap.add_argument("--ffmpeg", type=str, default="ffmpeg")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（0 表示全部）")
    ap.add_argument("--no_mux_wav", action="store_true", help="不合并 wav，直接拼接纯视频")
    ap.add_argument(
        "--no_concat",
        action="store_true",
        help="不生成 concat.txt / merged_with_wav.mp4，仅输出每段的 mp4",
    )
    args = ap.parse_args()

    segments_dir = os.path.abspath(args.segments_dir)
    out_dir = os.path.abspath(args.out_dir)
    vis_script = os.path.abspath(args.vis_script)

    if not os.path.isdir(segments_dir):
        raise FileNotFoundError(f"segments_dir not found: {segments_dir}")
    if not os.path.isfile(vis_script):
        raise FileNotFoundError(f"vis_script not found: {vis_script}")
    os.makedirs(out_dir, exist_ok=True)

    npzs = sorted(glob.glob(os.path.join(segments_dir, "*.npz")))
    if not npzs:
        raise FileNotFoundError(f"No npz found under: {segments_dir}")
    if args.limit and args.limit > 0:
        npzs = npzs[: int(args.limit)]

    tmp_npz_dir = os.path.join(out_dir, "_tmp_qpos_npz")
    os.makedirs(tmp_npz_dir, exist_ok=True)

    rendered: List[str] = []
    for i, npz_path in enumerate(npzs):
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        wav_path = os.path.join(segments_dir, stem + ".wav")
        mp4_path = os.path.join(out_dir, f"{i:06d}_{stem}.mp4")
        mp4_with_wav = os.path.join(out_dir, f"{i:06d}_{stem}_with_wav.mp4")

        vis_npz = _ensure_qpos_npz(npz_path, os.path.join(tmp_npz_dir, f"{i:06d}_{stem}.npz"))
        _run(
            [
                "python",
                vis_script,
                "--npz_path",
                vis_npz,
                "--video_path",
                mp4_path,
                "--robot",
                str(args.robot),
                "--motion_fps",
                str(int(args.motion_fps)),
            ]
        )

        if args.no_mux_wav or (not os.path.isfile(wav_path)):
            rendered.append(mp4_path)
            continue

        _run(
            [
                args.ffmpeg,
                "-y",
                "-i",
                mp4_path,
                "-i",
                wav_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                mp4_with_wav,
            ]
        )
        rendered.append(mp4_with_wav)

    if args.no_concat:
        print("[Done] (per-segment mp4 only, --no_concat)")
        for p in rendered:
            print(" ", p)
        return

    concat_path = os.path.join(out_dir, "concat.txt")
    with open(concat_path, "w", encoding="utf-8") as f:
        for p in rendered:
            f.write(f"file '{os.path.abspath(p)}'\n")

    merged_path = os.path.join(out_dir, "merged_with_wav.mp4")
    _run([args.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_path, "-c", "copy", merged_path])
    print("[Done]")
    print("  concat:", concat_path)
    print("  merged:", merged_path)


if __name__ == "__main__":
    main()

