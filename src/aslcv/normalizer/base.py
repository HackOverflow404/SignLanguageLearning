"""The pose-normalizer contract, mirroring extractor/base.py.

A normalizer repositions a `Pose`'s keypoints into reference frames so the same
sign from different signers, distances, or camera positions yields comparable
features. It runs at read time inside features.py, downstream of the raw pose
cache -- never at extraction (the cache stays raw so re-normalizing never forces
a re-extract).

Normalization is the *swappable strategy* the Phase 3 grid ablates, so features.py
injects any `Normalizer` through the single `normalize()` contract without knowing
which concrete one it holds. That injection seam is the whole reason this file
exists; it is deliberately minimal -- one abstract method, one return type.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..extractor.base import Pose, Skeleton

# The cross-topology anchor vocabulary the normalizer relies on: neutral names that
# every Skeleton must resolve via `Skeleton.anchor()` to its own index. It lives here
# (not inside any one topology) so it is the single contract; each skeleton maps these
# names to its own points, and tests/test_anchors.py asserts every skeleton covers
# every name. The hand anchors give ShoulderNormalizer a per-hand origin
# (`*_hand_wrist`) and scale reference (`*_hand_wrist` -> `*_hand_middle_mcp`).
REQUIRED_ANCHORS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_hand_wrist",
    "right_hand_wrist",
    "left_hand_middle_mcp",
    "right_hand_middle_mcp",
)


@dataclass
class NormalizedPose:
    """One frame's keypoints after normalization, grouped into reference-frame blocks.

    A normalizer is not "one pose in, one pose out": a single frame can yield points
    in DIFFERENT coordinate frames. ShoulderNormalizer, for example, emits a *global*
    block (body / arm / hand positions, shoulder-anchored) plus per-hand *local*
    blocks (each hand re-centred on its own wrist, so handshape is described
    independent of where the hand is). The same physical hand keypoint can therefore
    appear twice -- once as a global position, once as a local shape. A flat `Pose`
    cannot express "these rows are in shoulder-frame and those in wrist-frame," so
    this type pairs the keypoints with a map of which rows belong to which block.

    Fields
    ------
    keypoints : np.ndarray, shape (N, 2)
        Normalized coordinates for every point across every block, concatenated in
        block order. N is the sum of the block sizes, which may EXCEED the source
        Pose's keypoint count when a point is emitted in more than one frame.
    scores : np.ndarray, shape (N,)
        Per-point confidence, row-aligned to `keypoints` (a duplicated point carries
        its source score in each block). features.py does presence / confidence
        handling on this (presence = score > 0; see how mediapipe.py and
        rtmlib_base.py set scores); the normalizer only carries it through.
    blocks : dict[str, slice]
        Block name -> the contiguous row range it occupies in `keypoints` / `scores`,
        e.g. ``{"global": slice(0, G), "left_hand": slice(G, G + H), ...}``. Insertion
        order is the concatenation order. The reference frame is implied by the block
        name -- a documented convention between the normalizer and features.py -- which
        is all any consumer needs; nothing downstream reads the frame as data. (If that
        ever changes, add a parallel ``frames: dict[str, str]`` without disturbing the
        array layout.)
    blendshapes : np.ndarray | None
        The source Pose's (52,) facial-action coefficients, passed through unchanged
        (they are not spatial, so normalization does not touch them). Kept here so a
        frame's full signal -- geometry + non-manual markers -- lives on one object
        for the Phase 4 NMM heads. None when the backend supplies none (e.g. DWPose).

    Deliberately NOT carried: `width` / `height`. After normalization the coordinates
    are frame-relative and unitless, so the source pixel dimensions are meaningless
    here and are dropped to keep anyone from mixing them with normalized coords.

    Division of labor
    -----------------
    The normalizer owns *what frame each point is in* (the blocks). features.py owns
    everything after: block selection, concatenation into ``[body | L-hand | R-hand |
    face?]``, the confidence channel, velocity deltas, sequence stacking, and
    standardization. Keep those out of here and out of the normalizer.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    blocks: dict[str, slice]
    blendshapes: np.ndarray | None = None

    def block(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """The normalized (keypoints, scores) views for one block by name.

        Raises KeyError (listing the blocks present) if `name` is absent, mirroring
        `Skeleton.anchor`. Returns array views, not copies.
        """
        if name not in self.blocks:
            raise KeyError(
                f"{name!r} is not a block of this NormalizedPose; have "
                f"{list(self.blocks)}")
        sl = self.blocks[name]
        return self.keypoints[sl], self.scores[sl]


class Normalizer(ABC):
    """Interface for pose normalizers -- a swappable, read-time strategy.

    Turns a raw `Pose` (in the extractor's native topology) into a `NormalizedPose`
    whose keypoints sit in one or more reference frames, so the same sign is
    comparable across signers and distances. features.py injects one of these and
    calls `normalize()` without knowing which concrete strategy it is (Phase 3
    ablates normalizers); this contract is that injection seam.

    Configuration -- which blocks to emit (local-hand on/off, a future face on/off),
    any thresholds -- is a CONSTRUCTOR argument on the concrete normalizer, exactly
    as `Extractor` takes `running_mode`. The abstract `normalize` signature stays
    clean so every strategy is called identically.

    Rotation is deliberately NOT normalized. A tilted head is grammatical (topic
    marking, negation) and hand orientation is itself a phonological parameter, so
    both must be preserved -- normalize translation and scale only. Do not add
    rotation normalization "helpfully."
    """

    @abstractmethod
    def normalize(self, pose: Pose, skeleton: Skeleton) -> NormalizedPose:
        """Reposition `pose`'s keypoints into reference frames -> a `NormalizedPose`.

        `skeleton` supplies the semantic anchors by meaning
        (`skeleton.anchor("left_shoulder")`, and whichever other anchors the concrete
        strategy needs), so one normalizer works across all four extractor topologies
        without hard-coding indices. See `NormalizedPose` for the block / layout
        contract the return value must satisfy.
        """
        ...
