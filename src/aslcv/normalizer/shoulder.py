"""ShoulderNormalizer -- the one concrete Normalizer (see normalizer/base.py).

From a single Pose it produces:
  * a **global** block: every keypoint placed in a body frame -- origin at the
    shoulder midpoint, scale = shoulder width -- so body / arm / hand POSITIONS are
    comparable across signers and distances (carries location + movement).
    features.py selects which of these to keep (upper body + hands, drop legs/feet).
  * **left_hand** / **right_hand** blocks (when local_hand=True): each hand re-placed
    in its OWN frame -- origin at the hand wrist, scale = wrist -> middle-finger MCP
    -- so handshape is described independent of where the hand sits in space.
  * **scales**: the divisors above, kept instead of discarded -- see NormalizedPose.
    Building the local-hand frame already computes wrist->knuckle distance to divide
    by; dividing THAT by shoulder width turns "apparent hand size in pixels" (meaning-
    less -- depends on camera distance) into "apparent hand size relative to the
    signer's own shoulder width" (dimensionless, signer- and distance-invariant) --
    the classic monocular depth cue, computed for free from points already in hand.
    Also computed: elbow->wrist (arm-chain, not hand-block) distance / shoulder
    width, for arm foreshortening. Both are read by features.py's `depth_proxies`
    toggle (default off until measured).

All reference points are read by MEANING via `Skeleton.anchor()` ("left_shoulder",
"{side}_hand_wrist", "{side}_hand_middle_mcp", "{side}_elbow", "{side}_body_wrist"),
so one normalizer serves every extractor topology. Anchors give single points, not
sets, so the *members* of each hand are found via `Skeleton.region()` (see
`_hand_point_indices`) -- never by hardcoded offsets or name-prefix matching.

Frames are translation- and scale-invariant only. Rotation is deliberately NOT
normalized (a tilted head is grammatical; hand orientation is a phonological
parameter) -- see the `Normalizer` base class.

Degenerate inputs never divide by zero. A block whose frame cannot be built --
shoulders (or a hand's wrist/knuckle) undetected (score 0) or coincident (scale
<= eps) -- is emitted at its normal size but ZERO-FILLED (keypoints 0, scores 0).
Emitting at a fixed size keeps features.py's per-frame layout stable across frames;
the zero scores are the absence flag its presence check (score > 0) already reads,
matching the extractor's "no detection = zeros" convention. Scale ratios follow the
same convention: a ratio that can't be computed (either distance's points absent, or
shoulder width itself degenerate) is recorded as 0.0, never NaN or omitted.
"""
import numpy as np

from ..extractor.base import Pose, Skeleton
from .base import Normalizer, NormalizedPose


def _hand_point_indices(skeleton: Skeleton, side: str) -> list[int]:
    """Indices of the `side` ("left"/"right") hand's keypoints, resolved by REGION
    (the canonical cross-topology vocabulary, normalizer.base.REQUIRED_REGIONS) --
    not by name-prefix matching, so this stays correct for any skeleton that
    declares `left_hand`/`right_hand` regardless of its own naming convention."""
    return list(skeleton.region(f"{side}_hand"))


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
        scales: dict[str, float] = {}
        cursor = 0

        # --- global block: every point in the shoulder frame -------------------
        ls, rs = skeleton.anchor("left_shoulder"), skeleton.anchor("right_shoulder")
        width = float(np.linalg.norm(kp[ls] - kp[rs]))
        width_valid = bool(sc[ls] > 0) and bool(sc[rs] > 0) and width > self.eps
        if width_valid:
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
        for side in ("left", "right"):
            idx = _hand_point_indices(skeleton, side)
            wrist = skeleton.anchor(f"{side}_hand_wrist")
            mcp = skeleton.anchor(f"{side}_hand_middle_mcp")
            hand_scale = float(np.linalg.norm(kp[wrist] - kp[mcp]))
            hand_scale_valid = bool(sc[wrist] > 0) and bool(sc[mcp] > 0) and hand_scale > self.eps
            # Apparent hand size relative to shoulder width -- a monocular depth
            # proxy computed from the SAME distance the local-hand frame already
            # divides by, just not thrown away. Needs width to ALSO be valid: a
            # ratio against a degenerate denominator is meaningless, not just noisy.
            scales[f"{side}_hand"] = (hand_scale / width) if (hand_scale_valid and width_valid) else 0.0

            if self.local_hand and idx:
                if hand_scale_valid:
                    h_kp = (kp[idx] - kp[wrist]) / hand_scale
                    h_sc = sc[idx].copy()
                else:
                    h_kp = np.zeros((len(idx), 2), dtype=np.float32)
                    h_sc = np.zeros(len(idx), dtype=np.float32)
                block_kp.append(h_kp)
                block_sc.append(h_sc)
                blocks[f"{side}_hand"] = slice(cursor, cursor + len(idx))
                cursor += len(idx)

        # --- arm-foreshortening scale (elbow -> arm-chain wrist / shoulder width) --
        # A second, independent depth proxy: the forearm foreshortens the same way
        # a hand does when it moves along the camera axis. `*_body_wrist` is the
        # ARM-CHAIN wrist, deliberately not `*_hand_wrist` (the hand-block root) --
        # see the module docstring and coco_wholebody.py's anchor comment.
        for side in ("left", "right"):
            elbow = skeleton.anchor(f"{side}_elbow")
            body_wrist = skeleton.anchor(f"{side}_body_wrist")
            arm_scale = float(np.linalg.norm(kp[elbow] - kp[body_wrist]))
            arm_scale_valid = bool(sc[elbow] > 0) and bool(sc[body_wrist] > 0) and arm_scale > self.eps
            scales[f"{side}_arm"] = (arm_scale / width) if (arm_scale_valid and width_valid) else 0.0

        return NormalizedPose(
            keypoints=np.concatenate(block_kp, axis=0),
            scores=np.concatenate(block_sc, axis=0),
            blocks=blocks,
            blendshapes=pose.blendshapes,
            scales=scales,
        )
