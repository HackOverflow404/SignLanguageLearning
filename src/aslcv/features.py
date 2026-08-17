"""features.py -- assemble a per-frame pose sequence into a fixed-shape (T, F) array.

This module OWNS feature assembly. The normalizer owns "what reference frame is each
point in" (normalizer/shoulder.py); nothing here reaches back into normalization
geometry. Given a clip as a sequence of raw `Pose`s plus an injected `Normalizer`
and the extractor's `Skeleton`, the pipeline is:

  1. trim           -- (optional, default OFF until measured) drop leading/trailing
                       frames whose hand motion energy never clears a threshold --
                       ASL Citizen clips are recorded rest -> sign -> rest, and DTW
                       normalizes cost by path length, so a run of near-identical
                       rest frames aligns cheaply against EVERY candidate sign and
                       dilutes the gap between the right answer and the wrong ones.
                       Uses `hand_motion_energy()` (below), the SAME signal Phase
                       6's live boundary detector is meant to use -- one
                       implementation, two consumers.
  2. normalize      -- run the injected normalizer per frame -> NormalizedPose.
  3. selection      -- keep the signing blocks (global + local hands); drop the
                       global block's face points unless `face=True`, and its
                       leg/foot points unless `legs_feet=True` (both resolved via
                       `Skeleton.region()`, so the drop works by MEANING across
                       topologies -- see normalizer/base.py:REQUIRED_REGIONS).
  4. confidence     -- append per-point confidence as a channel (x, y, conf).
                       presence = score > 0; undetected points are zeroed. The conf
                       MODE reconciles backends: "graded" keeps scores as-is,
                       "binary" thresholds to {0,1} (Phase 3 passes "binary" so the
                       graded DWPose and the near-binary MediaPipe compare fairly).
  5. concat         -- lay blocks end-to-end in a FIXED order into one per-frame
                       vector, recording a name->slice map so Phase 4's phonological
                       heads can read e.g. the hand slice vs the body slice.
  6. velocity       -- (default on) append per-point (dx, dy) vs the previous frame;
                       movement is a phonological parameter. First frame delta = 0.
                       A point absent (score == 0) on either side of a delta gets a
                       zero delta instead -- signing occludes hands constantly, and
                       differencing against a zero-fill would read as a fake jump.
  7. depth_proxies  -- (optional, default OFF until measured) append 4 trailing
                       SCALAR columns -- left/right hand apparent size and
                       left/right arm foreshortening, each relative to shoulder
                       width -- read from `NormalizedPose.scales`. A monocular
                       depth cue 2D (x, y) keypoints are otherwise blind to,
                       computed for free from distances the normalizer already
                       divides by. Lives in its own `blocks["depth_proxies"]`
                       slice, NOT part of the per-point (x,y,conf[,dx,dy]) layout.
  8. stack          -- pile per-frame vectors -> (T, F). DTW needs no fixed length,
                       so raw variable length is the default; `to_fixed_length` is
                       available when a model needs it.
  9. standardize    -- (optional) per-feature (x - mean) / std, with stats FIT ON
                       TRAIN ONLY (`Standardizer`) and loaded/applied to val/test.

numpy only; no torch.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .extractor.base import Pose, Skeleton
from .normalizer.base import Normalizer

DEFAULT_BLOCK_ORDER = ("global", "left_hand", "right_hand")

# Fixed order for the depth_proxies trailing columns (see FeaturePipeline.assemble).
# A normalizer that doesn't compute a given key contributes 0.0 for it every frame
# (NormalizedPose.scales.get default) -- same zero-fill convention as everywhere
# else in this pipeline, not a special case.
DEPTH_PROXY_KEYS = ("left_hand", "right_hand", "left_arm", "right_arm")


def hand_motion_energy(normalizer: Normalizer, skeleton: Skeleton, poses) -> np.ndarray:
    """Per-frame motion energy: mean velocity MAGNITUDE over both hands' points,
    in the normalizer's local-hand (wrist-centered, wrist->knuckle-scaled) frame --
    so energy is comparable across signers/distances, exactly like every other
    normalized quantity in this pipeline. Returns a (T,) float32 array.

    ONE IMPLEMENTATION, TWO CONSUMERS: this is the signal FeaturePipeline's
    `trim_to_motion` toggle uses to find where a sign starts/stops (ASL Citizen
    clips are rest -> sign -> rest; DTW dilutes on rest frames -- see the module
    docstring's step 1), and it is meant to be the SAME signal Phase 6's live
    boundary detector uses online. Import and call this directly for that use
    rather than re-deriving "is the signer moving their hands" a second way.

    Independent of FeaturePipeline's face/legs_feet/confidence/velocity/
    depth_proxies toggles -- none of those affect hand velocity, so this runs its
    own minimal normalize + hand-position + velocity pass rather than requiring a
    fully-configured pipeline. Needs `normalizer` to emit `left_hand`/`right_hand`
    blocks (i.e. local_hand=True if it's a ShoulderNormalizer) -- raises
    ValueError otherwise, since "motion energy" is undefined with no hand points.
    """
    poses = list(poses)
    n_frames = len(poses)
    if n_frames == 0:
        raise ValueError("hand_motion_energy() needs at least one frame")

    first = normalizer.normalize(poses[0], skeleton)
    hand_blocks = [b for b in ("left_hand", "right_hand") if b in first.blocks]
    if not hand_blocks:
        raise ValueError(
            "hand_motion_energy() needs a normalizer that emits left_hand/right_hand "
            "blocks (e.g. ShoulderNormalizer(local_hand=True)); got blocks="
            f"{list(first.blocks)}")

    positions, presence = [], []
    for t, pose in enumerate(poses):
        npose = first if t == 0 else normalizer.normalize(pose, skeleton)
        kp_parts = [npose.block(b)[0] for b in hand_blocks]
        sc_parts = [npose.block(b)[1] for b in hand_blocks]
        positions.append(np.concatenate(kp_parts, axis=0))
        presence.append(np.concatenate(sc_parts, axis=0) > 0)
    positions = np.stack(positions).astype(np.float32)  # (T, P, 2)
    presence = np.stack(presence)                        # (T, P)

    if n_frames < 2:
        return np.zeros(n_frames, dtype=np.float32)

    vel = np.zeros_like(positions)
    vel[1:] = positions[1:] - positions[:-1]
    both_present = presence[1:] & presence[:-1]
    vel[1:][~both_present] = 0.0  # same presence-gap zero-fill as assemble()'s velocity step

    mag = np.linalg.norm(vel, axis=2)  # (T, P)
    # Average over PRESENT points only. Absent points are exactly 0 by the
    # zero-fill convention above, which would silently pull the mean toward 0 and
    # read as "low motion" when the real story is "hand not detected" -- a
    # different phenomenon that must not be confused with a genuine rest frame.
    present_count = presence.sum(axis=1)
    energy = np.zeros(n_frames, dtype=np.float32)
    has_any = present_count > 0
    energy[has_any] = mag[has_any].sum(axis=1) / present_count[has_any]
    return energy


def motion_active_span(energy: np.ndarray, threshold: float, pad_frames: int = 0) -> tuple[int, int]:
    """[start, stop) covering every frame whose motion energy exceeds `threshold`,
    padded by `pad_frames` on each side (clipped to the valid range). Never trims
    to empty: if no frame clears the threshold, returns the full range unchanged --
    a clip that never registers as "moving" is a detection problem, not something
    to discard."""
    active = energy > threshold
    if not active.any():
        return 0, len(energy)
    idx = np.nonzero(active)[0]
    start = max(0, int(idx[0]) - pad_frames)
    stop = min(len(energy), int(idx[-1]) + 1 + pad_frames)
    return start, stop


def live_capture_span(energy: np.ndarray, threshold: float, preroll: int, settle_frames: int) -> tuple[int, int]:
    """[start, stop) approximating what `capture.CaptureBuffer` would ACTUALLY
    capture live for this clip's content, for a cached rest->sign->rest clip
    that (like every clip in this dataset) has exactly one coherent motion
    burst: the motion-active span (`pad_frames=0` -- CaptureBuffer's own
    margins ARE the padding, not `motion_active_span`'s) extended by
    `preroll` frames before motion starts and `settle_frames` frames after it
    ends, matching CaptureBuffer's idle-state preroll trim and its
    settle-window completion rule.

    This is an offline APPROXIMATION of CaptureBuffer's real frame-by-frame
    state machine (see capture.py), not a byte-for-byte replay -- it doesn't
    need to be one, since it exists to answer "roughly how much rest padding
    will live capture keep around this sign," and CaptureBuffer's own
    behavior reduces to exactly this for the single-clean-motion-burst case
    every cached clip is. Built to close a REAL, measured train/serve
    mismatch: ASL Citizen's raw clips carry far more rest padding than a live
    capture ever will (see project_workflow.md's Phase 4 grader-accuracy
    investigation) -- `embedding_dataset.py` uses this to shape TRAINING data
    to match what the model will actually be fed live, instead of training
    on a rest-padding distribution live capture can't reproduce."""
    start, stop = motion_active_span(energy, threshold, pad_frames=0)
    return max(0, start - preroll), min(len(energy), stop + settle_frames)


@dataclass
class FeatureClip:
    """Assembled features for one clip.

    features : (T, F) float32
    blocks   : block name -> its contiguous slice in the F axis (covers all of F).
    channels_per_point : 3 (x, y, conf) or 5 (+ dx, dy) when velocity is on. Applies
        to every block EXCEPT `depth_proxies` (present only when that toggle is on),
        which is 4 trailing scalar columns, not per-point channels -- code that does
        channel-strided indexing across the full `features` width (e.g. pulling
        every point's confidence via `features[:, 2::channels_per_point]`) must
        exclude `blocks["depth_proxies"]` first.
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
        legs_feet: bool = False,
        confidence: str = "graded",
        binary_threshold: float = 0.5,
        velocity: bool = True,
        depth_proxies: bool = False,
        trim_to_motion: bool = False,
        motion_threshold: float = 0.02,
        motion_pad_frames: int = 3,
        standardizer: "Standardizer | None" = None,
    ):
        if confidence not in ("graded", "binary"):
            raise ValueError(f"confidence must be 'graded' or 'binary', got {confidence!r}")
        self.normalizer = normalizer
        self.skeleton = skeleton
        self.block_order = tuple(block_order)
        self.face = face
        self.legs_feet = legs_feet
        self.confidence = confidence
        self.binary_threshold = binary_threshold
        self.velocity = velocity
        self.depth_proxies = depth_proxies
        # trim_to_motion: drop leading/trailing REST frames before assembly, using
        # hand_motion_energy() (module-level, see above) -- off by default until
        # measured (see scripts/measure_motion_energy.py); motion_threshold is in
        # the SAME normalized units hand_motion_energy() returns (local-hand-frame
        # units per frame, scale-invariant), not raw pixels.
        self.trim_to_motion = trim_to_motion
        self.motion_threshold = motion_threshold
        self.motion_pad_frames = motion_pad_frames
        self.standardizer = standardizer

    # -- selection -----------------------------------------------------------

    def _kept_rows(self, block_name: str, n_points: int) -> np.ndarray:
        """Row indices to keep within a block. Only the global block is filtered --
        its face points are dropped unless `face=True`, and its leg/foot points are
        dropped unless `legs_feet=True`, both resolved via `Skeleton.region()` so
        the drop works by MEANING across topologies, not name-prefix matching; a
        hand block is kept whole. The global block is every keypoint in skeleton
        order, so row i names point i."""
        if block_name != "global":
            return np.arange(n_points)
        drop: set[int] = set()
        if not self.face:
            drop |= set(self.skeleton.region("face"))
        if not self.legs_feet:
            drop |= set(self.skeleton.region("legs_feet"))
        return np.array([i for i in range(n_points) if i not in drop], dtype=int)

    # -- assembly ------------------------------------------------------------

    def assemble(self, poses) -> FeatureClip:
        poses = list(poses)
        if not poses:
            raise ValueError("assemble() needs at least one frame")
        if self.trim_to_motion:
            energy = hand_motion_energy(self.normalizer, self.skeleton, poses)
            start, stop = motion_active_span(energy, self.motion_threshold, self.motion_pad_frames)
            poses = poses[start:stop]
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
        # presence[t, i] = point i was actually detected in frame t (score > 0),
        # tracked separately from the conf channel above: in "binary" mode conf is
        # thresholded at binary_threshold, so a low-but-nonzero score reads as
        # conf==0 without being ABSENT. Velocity needs the real presence bit, not
        # the confidence-mode-dependent channel.
        presence = np.zeros((n_frames, n_points), dtype=bool)
        # depth_proxies: monocular-depth scale ratios (apparent hand size / arm
        # foreshortening, both relative to shoulder width) the normalizer computed
        # anyway and would otherwise discard -- see NormalizedPose.scales. Filled in
        # the SAME per-frame loop as everything else so normalize() is called once
        # per frame, not twice.
        scale_arr = np.zeros((n_frames, len(DEPTH_PROXY_KEYS)), dtype=np.float32) if self.depth_proxies else None
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
                presence[t, off:off + m] = sc_sel > 0
                off += m
            if scale_arr is not None:
                for i, key in enumerate(DEPTH_PROXY_KEYS):
                    scale_arr[t, i] = npose.scales.get(key, 0.0)

        if self.velocity:
            pos = base[:, :, :2]
            vel = np.zeros_like(pos)
            vel[1:] = pos[1:] - pos[:-1]  # first-frame delta stays 0
            # A point absent on either side of a delta was zero-filled there, not
            # actually at that position, so differencing against it is a fake jump
            # (real -> 0 -> real reads as two violent moves). Zero the delta at
            # ANY frame touching an absence -- both the frame it disappears on and
            # the frame it reappears on -- rather than differencing against the
            # zero-fill.
            both_present = presence[1:] & presence[:-1]
            vel[1:][~both_present] = 0.0
            frame_feats = np.concatenate([base, vel], axis=2)  # (T, P, 5)
        else:
            frame_feats = base  # (T, P, 3)

        features = frame_feats.reshape(n_frames, n_points * channels).astype(np.float32)
        if scale_arr is not None:
            # Appended as trailing SCALAR columns, not per-point (x,y,conf[,dx,dy])
            # channels -- code that does channel-strided indexing across the whole
            # feature vector (e.g. `features[:, 2::channels]` to pull every point's
            # confidence) must not run over this block. Use blocks["depth_proxies"]
            # to isolate it, same as any other named block.
            F = features.shape[1]
            features = np.concatenate([features, scale_arr], axis=1)
            blocks["depth_proxies"] = slice(F, F + len(DEPTH_PROXY_KEYS))
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
