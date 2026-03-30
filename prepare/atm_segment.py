#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 BEAT2 的 sem 标注，把 BEAT_v2 原始动作/音频/whisper 特征切成 5s 片段。

输入（默认）：
  - data/BEAT_v2/<take_id>/*.npz                  # 动作，fps=60
  - data/BEAT_v2/<take_id>/*.wav                  # 原始音频
  - data/BEAT_v2/<take_id>/*_whisper_features.npy # whisper 特征，fps=50
  - data/BEAT2/beat_english_v2.0.0/sem/*.txt      # 语义段落

输出：
  - <out_root>/segments/<seg_id>.npz  (keys: motion, whisper)
  - <out_root>/segments/<seg_id>.wav  (尽力写出；缺 soundfile 时跳过)
  - <out_root>/index.tsv              (切分清单)
  - <out_root>/{train,val,all}.txt    (segment id 列表，便于直接喂给 BeatV2Dataset)

切分规则：
  - 每条 sem 中非 "01_beat_align" 的动作事件 => 切出一个 5s 片段
  - 片段以事件中心对齐（center = (t0+t1)/2），窗口长度固定 5s
  - 超出边界会 clamp，并对 motion/whisper/audio 进行补零
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


MOTION_FPS = 60
WHISPER_FPS = 50
SEG_SEC = 5.0
MOTION_LEN = int(round(MOTION_FPS * SEG_SEC))  # 300
WHISPER_LEN = int(round(WHISPER_FPS * SEG_SEC))  # 250


def _safe_makedirs(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _first_existing(paths: Iterable[str]) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _load_motion_npz(npz_path: str) -> np.ndarray:
    npz = np.load(npz_path, allow_pickle=True)
    key = None
    if "qpos" in npz:
        key = "qpos"
    elif "motion" in npz:
        key = "motion"
    else:
        keys = list(npz.keys())
        if keys:
            key = keys[0]
    if key is None:
        raise ValueError(f"Empty npz: {npz_path}")
    arr = np.asarray(npz[key])
    if arr.ndim != 2:
        raise ValueError(f"Expected motion array (T,D), got {arr.shape} from {npz_path} (key={key})")
    return arr.astype(np.float32)


def _load_whisper_npy(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 512:
        raise ValueError(f"Expected whisper feature shape (T,512), got {arr.shape} from {npy_path}")
    return arr


def _pad_slice_2d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """
    Slice x[start:start+length] and pad with zeros to exactly `length`.
    start can be negative; out-of-range is zero-padded.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.shape}")
    T, D = x.shape
    out = np.zeros((length, D), dtype=x.dtype)
    s0 = start
    s1 = start + length
    src0 = max(0, s0)
    src1 = min(T, s1)
    if src1 <= src0:
        return out
    dst0 = src0 - s0
    dst1 = dst0 + (src1 - src0)
    out[dst0:dst1] = x[src0:src1]
    return out


def _hash_to_float01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(16**8)


@dataclass(frozen=True)
class SemEvent:
    label: str
    t0: float
    t1: float
    score: float
    text: str
    raw: str


def _parse_sem_line(line: str) -> Optional[SemEvent]:
    ln = line.strip()
    if not ln:
        return None
    if ln.startswith("#"):
        return None
    parts = ln.split("\t")
    if len(parts) < 5:
        return None
    label = parts[0].strip()
    try:
        t0 = float(parts[1])
        t1 = float(parts[2])
        score = float(parts[4])
    except Exception:
        return None
    text = parts[5].strip() if len(parts) >= 6 else ""
    return SemEvent(label=label, t0=t0, t1=t1, score=score, text=text, raw=ln)


def _segment_start_from_event(t0: float, t1: float, max_sec: float) -> float:
    """
    Choose a 5s window for an event. Default: centered window with boundary clamp.
    """
    center = 0.5 * (t0 + t1)
    start = center - 0.5 * SEG_SEC
    if max_sec <= SEG_SEC:
        return 0.0
    start = max(0.0, min(start, max_sec - SEG_SEC))
    return float(start)


def _read_wav_best_effort(wav_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        return None, None
    try:
        wav, sr = sf.read(wav_path, always_2d=False)
        if wav is None:
            return None, None
        wav = np.asarray(wav)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        return wav, int(sr)
    except Exception:
        return None, None


def _write_wav_best_effort(wav_path: str, audio: np.ndarray, sr: int) -> bool:
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        return False
    try:
        _safe_makedirs(os.path.dirname(os.path.abspath(wav_path)))
        sf.write(wav_path, audio.astype(np.float32), int(sr))
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--beat_root",
        type=str,
        default=os.path.join("data", "BEAT_v2", "1"),
        help="原始 BEAT_v2 take 目录（里面有 *.npz/*.wav/*_whisper_features.npy）",
    )
    ap.add_argument(
        "--sem_root",
        type=str,
        default=os.path.join("data", "BEAT2", "beat_english_v2.0.0", "sem"),
        help="BEAT2 sem 标注目录（*.txt）",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default=os.path.join("data", "BEAT_v2", "segment_sem5s"),
        help="输出根目录",
    )
    ap.add_argument("--skip_label_prefix", type=str, default="01_beat_align", help="跳过该前缀的事件")
    ap.add_argument("--min_score", type=float, default=-1.0, help="只保留 score>=min_score 的事件")
    ap.add_argument("--val_ratio", type=float, default=0.05, help="val 集比例（按 seg_id hash 固定划分）")
    ap.add_argument("--dry_run", action="store_true", help="只生成 index，不写 npz/wav")
    args = ap.parse_args()

    beat_root = os.path.abspath(args.beat_root)
    sem_root = os.path.abspath(args.sem_root)
    out_root = os.path.abspath(args.out_root)
    seg_dir = os.path.join(out_root, "segments")
    _safe_makedirs(seg_dir)

    if not os.path.isdir(beat_root):
        raise FileNotFoundError(f"beat_root not found: {beat_root}")
    if not os.path.isdir(sem_root):
        raise FileNotFoundError(f"sem_root not found: {sem_root}")

    sem_files = [fn for fn in os.listdir(sem_root) if fn.endswith(".txt")]
    sem_files.sort()
    if not sem_files:
        raise FileNotFoundError(f"No sem txt found under: {sem_root}")

    index_path = os.path.join(out_root, "index.tsv")
    all_list: List[str] = []
    train_list: List[str] = []
    val_list: List[str] = []

    with open(index_path, "w", encoding="utf-8") as fw:
        fw.write(
            "\t".join(
                [
                    "seg_id",
                    "src_id",
                    "label",
                    "seg_start",
                    "seg_end",
                    "event_t0",
                    "event_t1",
                    "score",
                    "text",
                ]
            )
            + "\n"
        )

        for sem_fn in sem_files:
            src_id = os.path.splitext(sem_fn)[0]

            motion_path = os.path.join(beat_root, f"{src_id}.npz")
            wav_path = os.path.join(beat_root, f"{src_id}.wav")
            whisper_path = _first_existing(
                [
                    os.path.join(beat_root, f"{src_id}_whisper_features.npy"),
                    os.path.join(beat_root, f"{src_id}_whipser_features.npy"),
                ]
            )

            if not os.path.isfile(motion_path) or whisper_path is None:
                # 必需：motion + whisper 特征（wav 可选但建议存在）
                continue

            motion = _load_motion_npz(motion_path)
            whisper = _load_whisper_npy(whisper_path)
            wav, sr = _read_wav_best_effort(wav_path) if os.path.isfile(wav_path) else (None, None)

            max_sec_motion = motion.shape[0] / float(MOTION_FPS)
            max_sec_whisper = whisper.shape[0] / float(WHISPER_FPS)
            max_sec_wav = (wav.shape[0] / float(sr)) if (wav is not None and sr) else None
            max_sec = min([x for x in [max_sec_motion, max_sec_whisper, max_sec_wav] if x is not None])

            sem_path = os.path.join(sem_root, sem_fn)
            with open(sem_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            ev_i = 0
            for line in lines:
                ev = _parse_sem_line(line)
                if ev is None:
                    continue
                if args.skip_label_prefix and ev.label.startswith(args.skip_label_prefix):
                    continue
                if ev.score < float(args.min_score):
                    continue

                seg_start = _segment_start_from_event(ev.t0, ev.t1, max_sec=max_sec)
                seg_end = seg_start + SEG_SEC

                # Segment id: stable, informative, filesystem-safe
                seg_id = f"{src_id}__{ev.label}__{ev_i:04d}__{seg_start:.3f}"
                seg_id = seg_id.replace(os.sep, "_").replace(" ", "_")
                ev_i += 1

                all_list.append(seg_id)
                if _hash_to_float01(seg_id) < float(args.val_ratio):
                    val_list.append(seg_id)
                else:
                    train_list.append(seg_id)

                fw.write(
                    "\t".join(
                        [
                            seg_id,
                            src_id,
                            ev.label,
                            f"{seg_start:.6f}",
                            f"{seg_end:.6f}",
                            f"{ev.t0:.6f}",
                            f"{ev.t1:.6f}",
                            f"{ev.score:.6f}",
                            ev.text.replace("\t", " ").strip(),
                        ]
                    )
                    + "\n"
                )

                if args.dry_run:
                    continue

                m0 = int(round(seg_start * MOTION_FPS))
                w0 = int(round(seg_start * WHISPER_FPS))
                motion_seg = _pad_slice_2d(motion, m0, MOTION_LEN)
                whisper_seg = _pad_slice_2d(whisper, w0, WHISPER_LEN)

                out_npz = os.path.join(seg_dir, f"{seg_id}.npz")
                np.savez(out_npz, motion=motion_seg.astype(np.float32), whisper=whisper_seg.astype(np.float32))

                if wav is not None and sr is not None:
                    s0 = int(round(seg_start * sr))
                    s1 = int(round((seg_start + SEG_SEC) * sr))
                    audio_seg = np.zeros((int(round(SEG_SEC * sr)),), dtype=np.float32)
                    a0 = max(0, s0)
                    a1 = min(wav.shape[0], s1)
                    if a1 > a0:
                        dst0 = a0 - s0
                        dst1 = dst0 + (a1 - a0)
                        audio_seg[dst0:dst1] = wav[a0:a1]
                    out_wav = os.path.join(seg_dir, f"{seg_id}.wav")
                    _write_wav_best_effort(out_wav, audio_seg, sr)

    def _write_list(path: str, items: List[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(items) + ("\n" if items else ""))

    _write_list(os.path.join(out_root, "all.txt"), all_list)
    _write_list(os.path.join(out_root, "train.txt"), train_list)
    _write_list(os.path.join(out_root, "val.txt"), val_list)

    print(f"[Done] segments={len(all_list)} train={len(train_list)} val={len(val_list)}")
    print(f"  out_root: {out_root}")
    print(f"  index:    {index_path}")


if __name__ == "__main__":
    main()

