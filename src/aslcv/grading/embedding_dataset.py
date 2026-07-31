"""Dataset/dataloader glue for training the Phase 4 embedding grader.

Bridges `dataset.py` (manifest + cache loading), `features.py` (assembly +
standardization + `hand_motion_energy`), and `phonology_labels.py` (per-parameter
label encoders + support) into one in-memory torch Dataset, plus the PK batch sampler
batch-hard triplet training needs (P signs x K clips per batch) and its collate_fn.

Everything is precomputed once at construction time rather than re-featurized every
epoch: the whole 60-sign train split is <=~900 short clips, comfortably small enough
to hold as already-assembled (T, F) arrays in memory.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .. import dataset as dataset_mod
from ..extractor.base import Pose
from ..features import FeaturePipeline, Standardizer, hand_motion_energy
from .phonology_labels import CATEGORICAL_PARAMETERS, PhonologyLabels


def _load_poses_npz(npz_path) -> list[Pose]:
    """Raw per-frame Pose objects for one cached clip -- the same reconstruction
    FeaturePipeline.assemble_clip does internally, needed here as a standalone list
    because hand_motion_energy() takes poses directly, not an assembled FeatureClip."""
    with np.load(Path(npz_path)) as d:
        kp, sc = d["keypoints"], d["scores"]
        bs = d["blendshapes"] if "blendshapes" in d.files else None
        n = kp.shape[0]
        return [
            Pose(kp[t], sc[t], blendshapes=None if bs is None else bs[t])
            for t in range(n)
        ]


def _tempo_features(energy: np.ndarray) -> np.ndarray:
    """2 scalars -- (peak autocorrelation lag / clip length, peak height) -- the
    explicit periodicity signal `repeated` needs (CLAUDE.md known issue #5: DTW's
    length-normalization smears cyclic movement; geometry-only features inherit that).

    Computed via FFT autocorrelation of the hand motion-energy signal. Lags 0-1 are
    excluded from the peak search: energy is smooth frame-to-frame regardless of
    repetition, so the trivial near-lag-0 autocorrelation peak would otherwise dominate
    and say nothing about whether the motion actually repeats.
    """
    n = len(energy)
    if n < 4 or not np.any(energy):
        return np.zeros(2, dtype=np.float32)
    x = energy.astype(np.float64) - energy.mean()
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 1e-8:
        return np.zeros(2, dtype=np.float32)
    acf = acf / acf[0]
    search = acf[2:]
    if len(search) == 0:
        return np.zeros(2, dtype=np.float32)
    peak_idx = int(np.argmax(search))
    peak_lag = (peak_idx + 2) / n
    peak_height = float(np.clip(search[peak_idx], -1.0, 1.0))
    return np.array([peak_lag, peak_height], dtype=np.float32)


def _rows_for(extractor: str, split: str, signs) -> list[dict]:
    wanted = set(signs)
    return [r for r in dataset_mod.load_index(extractor, split) if r["id_gloss"] in wanted]


def fit_standardizer(pipeline: FeaturePipeline, extractor: str, signs) -> Standardizer:
    """Fit a Standardizer on the TRAIN split's raw features only (never val/test --
    matches DTWGrader.build's convention exactly)."""
    cache_dir = dataset_mod.CACHE / extractor
    rows = _rows_for(extractor, "train", signs)
    if not rows:
        raise RuntimeError(f"no train clips found for extractor={extractor!r} over given signs")
    feats = [pipeline.assemble_npz(cache_dir / f"{r['video_id']}.npz").features for r in rows]
    return Standardizer.fit(feats)


class EmbeddingClipDataset(torch.utils.data.Dataset):
    """One split's clips, pre-featurized + tempo-featured + phonology-labeled.

    `pipeline` must NOT carry a standardizer itself (pass raw features out); apply
    `standardizer` here explicitly so callers can fit it once on train and reuse the
    same fitted stats for val/test without accidentally re-fitting.
    """

    def __init__(self, extractor: str, split: str, pipeline: FeaturePipeline, signs,
                 id_gloss_encoder: "dataset_mod.LabelEncoder", phon_labels: PhonologyLabels,
                 standardizer: "Standardizer | None" = None):
        cache_dir = dataset_mod.CACHE / extractor
        rows = _rows_for(extractor, split, signs)
        if not rows:
            raise RuntimeError(f"no clips found for split={split!r} extractor={extractor!r}")

        self.items: list[dict] = []
        for r in rows:
            npz = cache_dir / f"{r['video_id']}.npz"
            clip = pipeline.assemble_npz(npz)
            feats = clip.features if standardizer is None else standardizer.transform(clip.features)
            poses = _load_poses_npz(npz)
            energy = hand_motion_energy(pipeline.normalizer, pipeline.skeleton, poses)
            tempo = _tempo_features(energy)

            gloss = r["id_gloss"]
            item = {
                "features": feats,
                "tempo": tempo,
                "id_gloss": id_gloss_encoder.encode(gloss),
                "id_gloss_name": gloss,
                "video_id": r["video_id"],
                "repeated_movement": 1.0 if phon_labels.repeated_bool(gloss) else 0.0,
            }
            for p in CATEGORICAL_PARAMETERS:
                item[p] = phon_labels.encoders[p].encode(phon_labels.label_for(gloss, p))
            self.items.append(item)

        self.blocks = clip.blocks  # same for every clip given a fixed pipeline config

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]

    @property
    def id_gloss_indices(self) -> list[int]:
        return [it["id_gloss"] for it in self.items]


def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch's variable-length (T, F) arrays to the batch's own max length."""
    lengths = torch.tensor([b["features"].shape[0] for b in batch], dtype=torch.int64)
    max_t = int(lengths.max())
    f_dim = batch[0]["features"].shape[1]
    feats = torch.zeros(len(batch), max_t, f_dim, dtype=torch.float32)
    for i, b in enumerate(batch):
        t = b["features"].shape[0]
        feats[i, :t] = torch.from_numpy(np.asarray(b["features"], dtype=np.float32))

    out = {
        "features": feats,
        "lengths": lengths,
        "tempo": torch.stack([torch.from_numpy(b["tempo"]) for b in batch]),
        "id_gloss": torch.tensor([b["id_gloss"] for b in batch], dtype=torch.int64),
        "repeated": torch.tensor([b["repeated_movement"] for b in batch], dtype=torch.float32),
    }
    for p in CATEGORICAL_PARAMETERS:
        out[p] = torch.tensor([b[p] for b in batch], dtype=torch.int64)
    return out


class PKBatchSampler(torch.utils.data.Sampler):
    """Yields P-signs x K-clips-per-sign index batches for batch-hard triplet training.

    Every anchor in a PK batch has >=1 same-label positive by construction (as long as
    K >= 2), which batch_hard_triplet_loss requires. If a sign has fewer than K clips
    in this split, samples with replacement for that sign only (degrades gracefully;
    the 60-sign train split's minimum is 13 clips/sign, so this never triggers there).
    """

    def __init__(self, id_gloss_indices: list[int], p: int, k: int, batches_per_epoch: int, seed: int = 0):
        self.by_sign: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(id_gloss_indices):
            self.by_sign[s].append(i)
        self.signs = list(self.by_sign)
        if p > len(self.signs):
            raise ValueError(f"p={p} exceeds the number of distinct signs in this split ({len(self.signs)})")
        self.p = p
        self.k = k
        self.batches_per_epoch = batches_per_epoch
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            signs = self.rng.sample(self.signs, self.p)
            batch: list[int] = []
            for s in signs:
                pool = self.by_sign[s]
                if len(pool) >= self.k:
                    batch.extend(self.rng.sample(pool, self.k))
                else:
                    batch.extend(self.rng.choices(pool, k=self.k))
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
