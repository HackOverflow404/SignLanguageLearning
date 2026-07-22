"""ShoulderNormalizer -- the one concrete Normalizer (see normalizer/base.py).

From a single Pose it produces:
  * a **global** block: every keypoint placed in a body frame -- origin at the
    shoulder midpoint, scale = shoulder width -- so body / arm / hand POSITIONS are
    comparable across signers and distances (carries location + movement).
    features.py selects which of these to keep (upper body + hands, drop legs/feet).
  * **left_hand** / **right_hand** blocks (when local_hand=True): each hand re-placed
    in its OWN frame -- origin at the hand wrist, scale = wrist -> middle-finger MCP
    -- so handshape is described independent of where the hand sits in space.

All reference points are read by MEANING via `Skeleton.anchor()` ("left_shoulder",
"{side}_hand_wrist", "{side}_hand_middle_mcp"), so one normalizer serves every
extractor topology. Anchors give single points, not sets, so the *members* of each
hand are found by name (see `_hand_point_indices`) -- never by hardcoded offsets.

Frames are translation- and scale-invariant only. Rotation is deliberately NOT
normalized (a tilted head is grammatical; hand orientation is a phonological
parameter) -- see the `Normalizer` base class.

Degenerate inputs never divide by zero. A block whose frame cannot be built --
shoulders (or a hand's wrist/knuckle) undetected (score 0) or coincident (scale
<= eps) -- is emitted at its normal size but ZERO-FILLED (keypoints 0, scores 0).
Emitting at a fixed size keeps features.py's per-frame layout stable across frames;
the zero scores are the absence flag its presence check (score > 0) already reads,
matching the extractor's "no detection = zeros" convention.
"""
import numpy as np

from ..extractor.base import Pose, Skeleton
from .base import Normalizer, NormalizedPose

# Name fragments marking a keypoint as part of a hand, across topologies:
# COCO-WholeBody ("left_hand_root", "left_middle_finger1", "left_thumb1") and
# MediaPipe ("left_hand_9"). With the "{side}_" prefix this selects exactly the hand
# points on either backend and nothing else (a body "left_wrist" matches none).
_HAND_NAME_FRAGMENTS = ("hand", "finger", "thumb")


def _hand_point_indices(skeleton: Skeleton, side: str) -> list[int]:
    """Indices of the `side` ("left"/"right") hand's keypoints, resolved by name."""
    prefix = f"{side}_"
    return [
        i
        for i, name in enumerate(skeleton.names)
        if name.startswith(prefix) and any(f in name for f in _HAND_NAME_FRAGMENTS)
    ]


class ShoulderNormalizer(Normalizer):
    """Shoulder-anchored global frame + optional per-hand wrist frames.

    Parameters
    ----------
    local_hand : bool
        Emit the per-hand local (handshape) blocks. False -> only the global block.
    eps : float
        A frame scale (shoulder width or wrist->knuckle distance) at or below this is
        treated as degenerate -- the block is zero-filled instead of divided by.
    """

    def __init__(self, local_hand: bool = True, eps: float = 1e-6):
        self.local_hand = local_hand
        self.eps = eps

    def normalize(self, pose: Pose, skeleton: Skeleton) -> NormalizedPose:
        kp = np.asarray(pose.keypoints, dtype=np.float32)
        sc = np.asarray(pose.scores, dtype=np.float32)
        n_total = kp.shape[0]

        block_kp: list[np.ndarray] = []
        block_sc: list[np.ndarray] = []
        blocks: dict[str, slice] = {}
        cursor = 0

        # --- global block: every point in the shoulder frame -------------------
        ls, rs = skeleton.anchor("left_shoulder"), skeleton.anchor("right_shoulder")
        width = float(np.linalg.norm(kp[ls] - kp[rs]))
        if bool(sc[ls] > 0) and bool(sc[rs] > 0) and width > self.eps:
            origin = (kp[ls] + kp[rs]) / 2.0
            g_kp = (kp - origin) / width
            g_sc = sc.copy()
        else:
            g_kp = np.zeros((n_total, 2), dtype=np.float32)
            g_sc = np.zeros(n_total, dtype=np.float32)
        block_kp.append(g_kp)
        block_sc.append(g_sc)
        blocks["global"] = slice(cursor, cursor + n_total)
        cursor += n_total

        # --- local hand blocks: each hand in its own wrist frame ---------------
        if self.local_hand:
            for side in ("left", "right"):
                idx = _hand_point_indices(skeleton, side)
                if not idx:
                    continue  # skeleton has no hand block for this side
                n = len(idx)
                wrist = skeleton.anchor(f"{side}_hand_wrist")
                mcp = skeleton.anchor(f"{side}_hand_middle_mcp")
                scale = float(np.linalg.norm(kp[wrist] - kp[mcp]))
                if bool(sc[wrist] > 0) and bool(sc[mcp] > 0) and scale > self.eps:
                    h_kp = (kp[idx] - kp[wrist]) / scale
                    h_sc = sc[idx].copy()
                else:
                    h_kp = np.zeros((n, 2), dtype=np.float32)
                    h_sc = np.zeros(n, dtype=np.float32)
                block_kp.append(h_kp)
                block_sc.append(h_sc)
                blocks[f"{side}_hand"] = slice(cursor, cursor + n)
                cursor += n

        return NormalizedPose(
            keypoints=np.concatenate(block_kp, axis=0),
            scores=np.concatenate(block_sc, axis=0),
            blocks=blocks,
            blendshapes=pose.blendshapes,
        )
