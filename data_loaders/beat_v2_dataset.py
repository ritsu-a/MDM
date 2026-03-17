import os
from os.path import join as pjoin
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils import data


# 固定的音频时序长度（帧数），用于裁剪 / 补零到统一长度
AUDIO_MAX_LEN = 250


def _read_id_list(split_file: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f.readlines()]
    return [x for x in ids if x]


def _load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expect a single BEAT_v2 npz with:
      - motion: (T_m, D)  e.g. (300, 60)
      - whisper: (T_a, 512)
    """
    npz = np.load(npz_path, allow_pickle=True)
    if "motion" not in npz or "whisper" not in npz:
        raise KeyError(f"Expected 'motion' and 'whisper' in {npz_path}, got {list(npz.keys())}")
    motion = np.asarray(npz["motion"], dtype=np.float32)
    whisper = np.asarray(npz["whisper"], dtype=np.float32)
    if motion.ndim != 2:
        raise ValueError(f"Expected motion shape (T,D), got {motion.shape} from {npz_path}")
    if whisper.ndim != 2 or whisper.shape[1] != 512:
        raise ValueError(f"Expected whisper shape (T,512), got {whisper.shape} from {npz_path}")
    return motion, whisper


def compute_mean_std(data_root: str, split_file: str) -> Tuple[np.ndarray, np.ndarray]:
    mean_path = pjoin(data_root, "Mean.npy")
    std_path = pjoin(data_root, "Std.npy")
    if os.path.isfile(mean_path) and os.path.isfile(std_path):
        return np.load(mean_path), np.load(std_path)

    ids = _read_id_list(split_file)
    motions: List[np.ndarray] = []
    for sid in ids:
        npz_path = pjoin(data_root, f"{sid}.npz")
        motion, _ = _load_npz(npz_path)
        motions.append(motion)

    concat = np.concatenate(motions, axis=0)  # (sum_T, D)
    mean = np.mean(concat, axis=0).astype(np.float32)
    std = np.std(concat, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    os.makedirs(data_root, exist_ok=True)
    np.save(mean_path, mean)
    np.save(std_path, std)
    return mean, std


class BeatV2Dataset(data.Dataset):
    """
    Simple audio-to-motion dataset for BEAT_v2 whisper features.

    Expected layout under data_root:
      - {ID}.npz with keys:
          * motion: (T_m, D) e.g. (300, 60)
          * whisper: (T_a, 512)
      - train.txt / val.txt listing IDs (without suffix)
      - Mean.npy / Std.npy (auto-computed from train split if missing)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        fixed_len: int = 300,
        max_motion_length: int = 300,
        seed: int = 0,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.fixed_len = fixed_len
        self.max_motion_length = max_motion_length

        split_file = pjoin(self.data_root, f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        self.ids = _read_id_list(split_file)

        self.mean, self.std = compute_mean_std(self.data_root, pjoin(self.data_root, "train.txt"))

        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def _get_item(self, sid: str) -> Dict:
        npz_path = pjoin(self.data_root, f"{sid}.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"BEAT_v2 npz not found: {npz_path}")

        motion, whisper = _load_npz(npz_path)  # (T_m,D), (T_a,512)

        # motion length & cropping/padding
        m_length = motion.shape[0]
        use_len = self.fixed_len if self.fixed_len > 0 else m_length
        use_len = min(use_len, self.max_motion_length)

        if motion.shape[0] >= use_len:
            if motion.shape[0] == use_len:
                start = 0
            else:
                start = self.rng.randint(0, motion.shape[0] - use_len)
            motion_cut = motion[start : start + use_len]
            length = use_len
        else:
            pad = np.zeros((use_len - motion.shape[0], motion.shape[1]), dtype=motion.dtype)
            motion_cut = np.concatenate([motion, pad], axis=0)
            length = motion.shape[0]

        motion_norm = (motion_cut - self.mean) / self.std
        if motion_norm.shape[0] < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - motion_norm.shape[0], motion_norm.shape[1]), dtype=motion_norm.dtype)
            motion_norm = np.concatenate([motion_norm, pad], axis=0)

        # audio feature: 保留 whisper 的时序特征 (T_a, 512)，并裁剪 / 补零到固定长度 AUDIO_MAX_LEN
        T_a = whisper.shape[0]
        if T_a >= AUDIO_MAX_LEN:
            # 这里简单从开头截取 AUDIO_MAX_LEN 帧；如需随机截取可改为随机 start
            start = 0
            whisper_cut = whisper[start:start + AUDIO_MAX_LEN]
        else:
            pad = np.zeros((AUDIO_MAX_LEN - T_a, whisper.shape[1]), dtype=whisper.dtype)
            whisper_cut = np.concatenate([whisper, pad], axis=0)

        audio_seq = whisper_cut.astype(np.float32)  # (AUDIO_MAX_LEN, 512)

        return {
            "audio_feat": audio_seq,
            "motion": motion_norm.astype(np.float32),
            "length": int(length),
            "key": sid,
        }

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        item = self._get_item(sid)
        return (
            item["audio_feat"],  # 0: (512,)
            item["motion"],      # 1: (T,D) normalized
            item["length"],      # 2: scalar
            item["key"],         # 3: id
        )


class BeatV2(data.Dataset):
    """
    Wrapper for BEAT_v2 audio-to-motion to match get_data API.
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = 300,
        mode: str = "train",
        abs_path: str = ".",
        fixed_len: int = 300,
        device=None,
        autoregressive: bool = False,
        data_dir: str = "",
        cache_path: Optional[str] = None,
    ):
        super().__init__()
        _ = (num_frames, mode, abs_path, device, autoregressive, cache_path)
        # 默认使用你提供的 BEAT_v2 子目录：data/BEAT_v2/segment_1wayne
        data_root = data_dir if data_dir else pjoin(abs_path, "data", "BEAT_v2", "segment_1wayne")
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"BEAT_v2 data root not found: {data_root}")

        self.dataset = BeatV2Dataset(
            data_root=data_root,
            split=split,
            fixed_len=fixed_len if fixed_len > 0 else 300,
            max_motion_length=300,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

