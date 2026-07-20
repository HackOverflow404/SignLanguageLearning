"""Phase 1 loader: join cached pose sequences to phonology + signer-independent splits.

Given an extractor name and a split, yields
``(pose_sequence, id_gloss, phonological_features, signer_id)`` by joining
``data/cache/{extractor}/_manifest.csv`` (written by scripts/extract_landmarks.py)
with ``data/phonology.csv`` (written by tools/join_phonology.py) on ``asllex_code``.

Also provides:
  * ``LabelEncoder`` -- id_gloss <-> int, deterministic (sorted).
  * ``build_torch_dataset`` -- a fixed-length ``torch.utils.data.Dataset`` that
    pads/crops each sequence and exposes the label encoder. torch is imported
    lazily inside that factory, so importing this module never requires torch.

Raw keypoints are returned in the extractor's native topology; normalization is a
read-time concern for features.py, not here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "cache"
PHONOLOGY_CSV = REPO / "data" / "phonology.csv"
SPLITS = ("train", "val", "test")


@dataclass
class PoseSequence:
    """A cached pose clip in the extractor's native topology."""

    keypoints: np.ndarray            # (T, K, 2) pixel coords
    scores: np.ndarray               # (T, K) confidence
    blendshapes: np.ndarray | None   # (T, 52) MediaPipe facial coeffs, else None
    fps: float
    video_id: str

    @property
    def n_frames(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def n_keypoints(self) -> int:
        return int(self.keypoints.shape[1])


# ---- CSV joins (no torch, no npz load) --------------------------------------


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_phonology() -> dict[str, dict]:
    """{asllex_code: phonology row}. Raises if join_phonology.py hasn't been run."""
    try:
        return {r["asllex_code"]: r for r in _read_csv(PHONOLOGY_CSV)}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{PHONOLOGY_CSV} not found -- run `tools/join_phonology.py` first")


def manifest_path(extractor: str) -> Path:
    return CACHE / extractor / "_manifest.csv"


def load_index(extractor: str, split: str | None = None) -> list[dict]:
    """Manifest rows for an extractor (optionally one split), each joined to its
    phonology row under key ``phonology``. Cheap -- does not open any .npz."""
    mpath = manifest_path(extractor)
    try:
        rows = _read_csv(mpath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{mpath} not found -- run "
            f"`scripts/extract_landmarks.py --extractor {extractor}` first")
    phon = load_phonology()
    out = []
    for r in rows:
        if split is not None and r["split"] != split:
            continue
        rec = dict(r)
        rec["phonology"] = phon.get(r["asllex_code"])
        out.append(rec)
    return out


def load_pose(extractor: str, video_id: str) -> PoseSequence:
    npz = CACHE / extractor / f"{video_id}.npz"
    with np.load(npz) as d:
        bs = d["blendshapes"] if "blendshapes" in d.files else None
        return PoseSequence(
            keypoints=d["keypoints"], scores=d["scores"], blendshapes=bs,
            fps=float(d["fps"]), video_id=video_id,
        )


def iter_records(extractor: str, split: str):
    """Yield ``(pose_sequence, id_gloss, phonological_features, signer_id)`` for a
    split, loading each clip's .npz lazily."""
    for r in load_index(extractor, split):
        pose = load_pose(extractor, r["video_id"])
        yield pose, r["id_gloss"], r["phonology"], r["signer_id"]


# ---- label encoding ---------------------------------------------------------


class LabelEncoder:
    """Deterministic id_gloss <-> int mapping (classes sorted)."""

    def __init__(self, labels):
        self.classes_ = sorted(set(labels))
        self._to_idx = {c: i for i, c in enumerate(self.classes_)}

    def encode(self, label: str) -> int:
        return self._to_idx[label]

    def decode(self, idx: int) -> str:
        return self.classes_[idx]

    def __len__(self) -> int:
        return len(self.classes_)


def _fixed_length(pose: PoseSequence, max_len: int, with_scores: bool) -> np.ndarray:
    """Pad (zeros) or centre-crop a sequence to exactly max_len frames.

    Zero-padding matches the extractor's "no detection = zeros" convention, so
    padded frames are indistinguishable from missed frames (both masked by score 0).
    Returns (max_len, K, 3) as [x, y, score] if with_scores else (max_len, K, 2)."""
    kp = pose.keypoints.astype(np.float32)
    if with_scores:
        feat = np.concatenate([kp, pose.scores.astype(np.float32)[..., None]], axis=-1)
    else:
        feat = kp
    t = feat.shape[0]
    if t == max_len:
        return feat
    if t > max_len:
        start = (t - max_len) // 2            # centre crop
        return feat[start:start + max_len]
    pad = np.zeros((max_len - t, *feat.shape[1:]), dtype=np.float32)
    return np.concatenate([feat, pad], axis=0)


def build_torch_dataset(extractor: str, split: str, max_len: int = 128,
                        with_scores: bool = True, label_encoder: "LabelEncoder | None" = None):
    """A fixed-length ``torch.utils.data.Dataset`` over one split.

    Each item is ``(features, label)`` where features is a float tensor of shape
    ``(max_len, K, 3)`` ([x, y, score]; ``with_scores=False`` -> (max_len, K, 2))
    and label is a long tensor. ``.label_encoder`` is exposed on the dataset.

    torch is imported here (lazily) so importing aslcv.dataset never needs torch.
    The label encoder is fit over ALL splits by default, so codes are stable when
    train/val/test datasets are built separately.
    """
    import torch  # noqa: local import by design

    index = load_index(extractor, split)
    if label_encoder is None:
        label_encoder = LabelEncoder(r["id_gloss"] for r in load_index(extractor))

    class PoseDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.extractor = extractor
            self.split = split
            self.index = index
            self.max_len = max_len
            self.with_scores = with_scores
            self.label_encoder = label_encoder

        def __len__(self):
            return len(self.index)

        def __getitem__(self, i):
            r = self.index[i]
            pose = load_pose(self.extractor, r["video_id"])
            feat = _fixed_length(pose, self.max_len, self.with_scores)
            label = self.label_encoder.encode(r["id_gloss"])
            return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    return PoseDataset()
