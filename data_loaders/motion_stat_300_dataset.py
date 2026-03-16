import os
import random
from os.path import join as pjoin
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch
from torch.utils import data

from data_loaders.humanml.utils.word_vectorizer import WordVectorizer


def _read_id_list(split_file: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f.readlines()]
    return [x for x in ids if x]


def _load_motion_npz(motion_npz_path: str) -> np.ndarray:
    npz = np.load(motion_npz_path)
    if "qpos" in npz:
        arr = npz["qpos"]
    elif "motion" in npz:
        arr = npz["motion"]
    else:
        # fall back to the first array key
        keys = list(npz.keys())
        if len(keys) == 0:
            raise ValueError(f"Empty npz file: {motion_npz_path}")
        arr = npz[keys[0]]
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected motion array with shape (T, D), got {arr.shape} from {motion_npz_path}")
    return arr.astype(np.float32)


def compute_mean_std(data_root: str, split_file: str, cache_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    mean_path = pjoin(data_root, "Mean.npy")
    std_path = pjoin(data_root, "Std.npy")
    if os.path.isfile(mean_path) and os.path.isfile(std_path):
        return np.load(mean_path), np.load(std_path)

    ids = _read_id_list(split_file)
    motions: List[np.ndarray] = []
    for sid in ids:
        motion_path = pjoin(data_root, f"{sid}_motion.npz")
        motions.append(_load_motion_npz(motion_path))

    concat = np.concatenate(motions, axis=0)  # (sum_T, D)
    mean = np.mean(concat, axis=0).astype(np.float32)
    std = np.std(concat, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    os.makedirs(data_root, exist_ok=True)
    np.save(mean_path, mean)
    np.save(std_path, std)
    return mean, std


def _tokenize_caption(nlp, caption: str) -> List[str]:
    doc = nlp(caption.strip())
    tokens: List[str] = []
    for t in doc:
        if t.is_space:
            continue
        w = t.text.lower()
        pos = t.pos_ if t.pos_ else "OTHER"
        tokens.append(f"{w}/{pos}")
    return tokens


class MotionStat300Dataset(data.Dataset):
    """
    Expected data layout under data_root:
      - {ID}.txt (caption)
      - {ID}_motion.npz (motion), with key 'qpos' preferred, shape (T, 60) where T=300
      - train.txt / val.txt listing IDs (without suffixes)
      - Mean.npy / Std.npy (auto computed from train split if missing)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        fixed_len: int = 300,
        max_motion_length: int = 300,
        cache_path: Optional[str] = None,
        glove_root: str = "./glove",
        glove_prefix: str = "our_vab",
        seed: int = 0,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.fixed_len = fixed_len
        self.max_motion_length = max_motion_length
        self.rng = random.Random(seed)

        split_file = pjoin(self.data_root, f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        self.ids = _read_id_list(split_file)

        self.mean, self.std = compute_mean_std(self.data_root, pjoin(self.data_root, "train.txt"))

        self.w_vectorizer = WordVectorizer(glove_root, glove_prefix)
        self.max_text_len = 20
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")

        self._cache_path = cache_path
        self._cached: Optional[Dict[str, Dict]] = None
        if cache_path and os.path.isfile(cache_path):
            cached = np.load(cache_path, allow_pickle=True)
            self._cached = cached.item() if isinstance(cached, np.ndarray) else cached

    def inv_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse normalization.
        Tries to coerce common layout variants to (B, T, 60):
          - (B, T, 60)
          - (B, T, 1, 60)
          - (B, 1, T, 60)
          - (B, 60, T, 1)  # e.g. (B, C, T, 1)
        """
        if x.ndim == 4:
            # (B, T, 1, 60) -> (B, T, 60)
            if x.shape[2] == 1 and x.shape[3] == 60:
                x = x.squeeze(2)
            # (B, 1, T, 60) -> (B, T, 60)
            elif x.shape[1] == 1 and x.shape[3] == 60:
                x = x.squeeze(1)
            # (B, 60, T, 1) -> (B, T, 60)
            elif x.shape[1] == 60 and x.shape[3] == 1:
                x = x.permute(0, 2, 1, 3).squeeze(3)
            else:
                raise ValueError(
                    f"Expected normalized motion with shape (B,T,60) or (B,T,1,60), "
                    f"or a simple layout variant, got {tuple(x.shape)}"
                )
        if x.ndim != 3:
            raise ValueError(f"Expected normalized motion with shape (B,T,60) or (B,T,1,60), got {tuple(x.shape)}")
        mean = torch.from_numpy(self.mean).to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        std = torch.from_numpy(self.std).to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        return x * std + mean

    def __len__(self):
        return len(self.ids)

    def _get_item_uncached(self, sid: str) -> Dict:
        txt_path = pjoin(self.data_root, f"{sid}.txt")
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"Caption file not found: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        if not caption:
            caption = "a person moves."

        motion_path = pjoin(self.data_root, f"{sid}_motion.npz")
        motion = _load_motion_npz(motion_path)  # (T,D)
        m_length = motion.shape[0]

        tokens = _tokenize_caption(self.nlp, caption)
        if len(tokens) == 0:
            tokens = ["unk/OTHER"]
        sent_len = min(len(tokens), self.max_text_len)
        tokens = tokens[: self.max_text_len]
        if len(tokens) < self.max_text_len:
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len - len(tokens))

        word_embeddings = []
        pos_one_hots = []
        for tok in tokens:
            wv, pv = self.w_vectorizer[tok]
            word_embeddings.append(wv)
            pos_one_hots.append(pv)
        word_embeddings = np.stack(word_embeddings, axis=0).astype(np.float32)
        pos_one_hots = np.stack(pos_one_hots, axis=0).astype(np.float32)

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

        return {
            "word_embeddings": word_embeddings,
            "pos_one_hots": pos_one_hots,
            "caption": caption,
            "sent_len": sent_len,
            "motion": motion_norm.astype(np.float32),
            "length": int(length),
            "tokens_str": tokens,
            "key": sid,
        }

    def __getitem__(self, idx):
        sid = self.ids[idx]
        if self._cached is not None and sid in self._cached:
            item = self._cached[sid]
        else:
            item = self._get_item_uncached(sid)
        return (
            item["word_embeddings"],
            item["pos_one_hots"],
            item["caption"],
            item["sent_len"],
            item["motion"],
            item["length"],
            item["tokens_str"],
            item["key"],
        )


class MotionStat300(data.Dataset):
    """
    Wrapper to match existing get_data API expectations.
    Exposes .t2m_dataset for inv_transform usage in sampling code.
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
        _ = (num_frames, mode, abs_path, device, autoregressive)  # kept for signature compatibility
        data_root = data_dir if data_dir else pjoin(abs_path, "data", "motion_stat_300")
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"motion_stat_300 data root not found: {data_root}")

        if cache_path is None:
            os.makedirs(pjoin(abs_path, "dataset"), exist_ok=True)
            cache_path = pjoin(abs_path, "dataset", f"motion_stat_300_{split}.npy")

        self.t2m_dataset = MotionStat300Dataset(
            data_root=data_root,
            split=split,
            fixed_len=fixed_len if fixed_len > 0 else 300,
            max_motion_length=300,
            cache_path=cache_path,
            glove_root="./glove",
            glove_prefix="our_vab",
        )

    def __len__(self):
        return len(self.t2m_dataset)

    def __getitem__(self, idx):
        return self.t2m_dataset[idx]

