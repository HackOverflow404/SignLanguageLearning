"""features.py -- assemble a per-frame pose sequence into a fixed-shape (T, F) array.

This module OWNS feature assembly. The normalizer owns "what reference frame is each
point in" (normalizer/shoulder.py); nothing here reaches back into normalization
geometry. Given a clip as a sequence of raw `Pose`s plus an injected `Normalizer`
and the extractor's `Skeleton`, the pipeline is:

  1. normalize      -- run the injected normalizer per frame -> NormalizedPose.
  2. selection      -- keep the signing blocks (global + local hands); drop the
                       global block's face points unless `face=True`.
  3. confidence     -- append per-point confidence as a channel (x, y, conf).
                       presence = score > 0; undetected points are zeroed. The conf
                       MODE reconciles backends: "graded" keeps scores as-is,
                       "binary" thresholds to {0,1} (Phase 3 passes "binary" so the
                       graded DWPose and the near-binary MediaPipe compare fairly).
  4. concat         -- lay blocks end-to-end in a FIXED order into one per-frame
                       vector, recording a name->slice map so Phase 4's phonological
                       heads can read e.g. the hand slice vs the body slice.
  5. velocity       -- (default on) append per-point (dx, dy) vs the previous frame;
                       movement is a phonological parameter. First frame delta = 0.
  6. stack          -- pile per-frame vectors -> (T, F). DTW needs no fixed length,
                       so raw variable length is the default; `to_fixed_length` is
                       available when a model needs it.
  7. standardize    -- (optional) per-feature (x - mean) / std, with stats FIT ON
                       TRAIN ONLY (`Standardizer`) and loaded/applied to val/test.

numpy only; no torch.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .extractor.base import Pose, Skeleton
from .normalizer.base import Normalizer

DEFAULT_BLOCK_ORDER = ("global", "left_hand", "right_hand")


@dataclass
class FeatureClip:
    """Assembled features for one clip.

    features : (T, F) float32
    blocks   : block name -> its contiguous slice in the F axis (covers all of F).
    channels_per_point : 3 (x, y, conf) or 5 (+ dx, dy) when velocity is on.
    """

    features: np.ndarray
    blocks: dict[str, slice]
    channels_per_point: int


class Standardizer:
    """Per-feature (x - mean) / std. Fit on the TRAIN split only, then applied to
    val/test -- fitting on val/test would leak. Persisted as a small .npz."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, feature_arrays) -> "Standardizer":
        """`feature_arrays`: iterable of (T, F) arrays (or FeatureClips) from TRAIN."""
        frames = np.concatenate(
            [a.features if isinstance(a, FeatureClip) else np.asarray(a) for a in feature_arrays],
            axis=0,
        )
        mean = frames.mean(axis=0)
        std = frames.std(axis=0)
        std[std < 1e-6] = 1.0  # constant features -> leave unscaled, don't divide by ~0
        return cls(mean, std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)

    def save(self, path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path) -> "Standardizer":
        with np.load(path) as d:
            return cls(d["mean"], d["std"])


def to_fixed_length(features: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad (at the end) or centre-crop a (T, F) array to exactly `length` frames.
    Optional -- DTW handles variable length and should use the raw features."""
    t = features.shape[0]
    if t == length:
        return features
    if t > length:
        start = (t - length) // 2
        return features[start:start + length]
    pad = np.zeros((length - t, features.shape[1]), dtype=features.dtype)
    return np.concatenate([features, pad], axis=0)


class FeaturePipeline:
    """Turns a clip (sequence of raw `Pose`s) into a `FeatureClip`.

    The `normalizer` is injected -- Phase 3 ablates normalizers behind this one seam
    without touching assembly. Config toggles live here; the normalizer's own config
    (e.g. `local_hand`) decides which blocks exist, and this pipeline simply assembles
    whatever blocks the normalizer emits (in `block_order`), so toggling the
    normalizer's `local_hand` changes F predictably.
    """

    def __init__(
        self,
        normalizer: Normalizer,
        skeleton: Skeleton,
        *,
        block_order=DEFAULT_BLOCK_ORDER,
        face: bool = False,
        confidence: str = "graded",
        binary_threshold: float = 0.5,
        velocity: bool = True,
        standardizer: "Standardizer | None" = None,
    ):
        if confidence not in ("graded", "binary"):
            raise ValueError(f"confidence must be 'graded' or 'binary', got {confidence!r}")
        self.normalizer = normalizer
        self.skeleton = skeleton
        self.block_order = tuple(block_order)
        self.face = face
        self.confidence = confidence
        self.binary_threshold = binary_threshold
        self.velocity = velocity
        self.standardizer = standardizer

    # -- selection -----------------------------------------------------------

    def _kept_rows(self, block_name: str, n_points: int) -> np.ndarray:
        """Row indices to keep within a block. Only the global block is filtered
        (its face points are dropped unless `face=True`); a hand block is kept whole.
        The global block is every keypoint in skeleton order, so row i names point i."""
        if block_name == "global" and not self.face:
            names = self.skeleton.names
            return np.array(
                [i for i in range(n_points) if not names[i].startswith("face")],
                dtype=int,
            )
        return np.arange(n_points)

    # -- assembly ------------------------------------------------------------

    def assemble(self, poses) -> FeatureClip:
        poses = list(poses)
        if not poses:
            raise ValueError("assemble() needs at least one frame")
        n_frames = len(poses)
        channels = 5 if self.velocity else 3

        # Layout is fixed for the clip: derive it from frame 0 (the normalizer emits
        # the same block set every frame for a given config).
        first = self.normalizer.normalize(poses[0], self.skeleton)
        present = [b for b in self.block_order if b in first.blocks]
        kept = {b: self._kept_rows(b, first.block(b)[0].shape[0]) for b in present}

        blocks: dict[str, slice] = {}
        p = 0
        for b in present:
            m = len(kept[b])
            blocks[b] = slice(p * channels, (p + m) * channels)
            p += m
        n_points = p

        # base[t] = [x, y, conf] per selected point
        base = np.zeros((n_frames, n_points, 3), dtype=np.float32)
        for t, pose in enumerate(poses):
            npose = first if t == 0 else self.normalizer.normalize(pose, self.skeleton)
            off = 0
            for b in present:
                kp, sc = npose.block(b)
                rows = kept[b]
                kp_sel, sc_sel = kp[rows], sc[rows]
                if self.confidence == "graded":
                    conf = sc_sel
                else:
                    conf = (sc_sel > self.binary_threshold).astype(np.float32)
                pt = np.concatenate([kp_sel, conf[:, None]], axis=1)  # (m, 3)
                pt[sc_sel <= 0] = 0.0  # presence = score > 0; absent points are zeros
                m = len(rows)
                base[t, off:off + m] = pt
                off += m

        if self.velocity:
            pos = base[:, :, :2]
            vel = np.zeros_like(pos)
            vel[1:] = pos[1:] - pos[:-1]  # first-frame delta stays 0
            frame_feats = np.concatenate([base, vel], axis=2)  # (T, P, 5)
        else:
            frame_feats = base  # (T, P, 3)

        features = frame_feats.reshape(n_frames, n_points * channels).astype(np.float32)
        if self.standardizer is not None:
            features = self.standardizer.transform(features)
        return FeatureClip(features=features, blocks=blocks, channels_per_point=channels)

    def assemble_clip(self, keypoints, scores, blendshapes=None) -> FeatureClip:
        """Assemble from raw per-clip arrays (T, K, 2) / (T, K) [/ (T, 52)]."""
        n = keypoints.shape[0]
        poses = [
            Pose(
                keypoints[t],
                scores[t],
                blendshapes=None if blendshapes is None else blendshapes[t],
            )
            for t in range(n)
        ]
        return self.assemble(poses)

    def assemble_npz(self, npz_path) -> FeatureClip:
        """Assemble a cached clip from data/cache/{extractor}/{video_id}.npz."""
        with np.load(Path(npz_path)) as d:
            bs = d["blendshapes"] if "blendshapes" in d.files else None
            return self.assemble_clip(d["keypoints"], d["scores"], bs)
