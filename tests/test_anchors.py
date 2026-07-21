"""Cross-topology anchor coverage + a hand-anchor sanity eyeball.

Runs under pytest OR as a plain script (`python tests/test_anchors.py`).

Every skeleton must resolve every name in REQUIRED_ANCHORS to a valid in-range
index, and the hand anchors must land in that skeleton's own left/right hand block.
The last test prints, for one real cached clip per topology, the pixel coords of the
hand wrist (should sit on the body wrist) and the middle-finger MCP (should sit
between wrist and fingertip, not at the tip) -- a human check that the knuckle isn't
the fingertip.
"""
import glob
from pathlib import Path

import numpy as np

from aslcv.normalizer.base import REQUIRED_ANCHORS
from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY, LEFT_HAND, RIGHT_HAND
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC

REPO = Path(__file__).resolve().parents[1]
SKELETONS = {"coco_wholebody": COCO_WHOLEBODY, "mediapipe_holistic": MEDIAPIPE_HOLISTIC}


def _hand_blocks(name, sk):
    """(left indices, right indices) for a skeleton's hand keypoints."""
    if name == "coco_wholebody":
        return set(LEFT_HAND), set(RIGHT_HAND)
    left = {i for i, n in enumerate(sk.names) if n.startswith("left_hand")}
    right = {i for i, n in enumerate(sk.names) if n.startswith("right_hand")}
    return left, right


def test_all_skeletons_provide_required_anchors():
    for name, sk in SKELETONS.items():
        for anchor in REQUIRED_ANCHORS:
            idx = sk.anchor(anchor)  # raises KeyError if the skeleton lacks it
            assert 0 <= idx < len(sk.names), f"{name}: {anchor} -> {idx} out of range"


def test_hand_anchors_land_in_their_hand_block():
    for name, sk in SKELETONS.items():
        left, right = _hand_blocks(name, sk)
        assert sk.anchor("left_hand_wrist") in left, f"{name}: left_hand_wrist not in left block"
        assert sk.anchor("left_hand_middle_mcp") in left, f"{name}: left_hand_middle_mcp not in left block"
        assert sk.anchor("right_hand_wrist") in right, f"{name}: right_hand_wrist not in right block"
        assert sk.anchor("right_hand_middle_mcp") in right, f"{name}: right_hand_middle_mcp not in right block"


# (cache dir, skeleton, body-wrist point name, middle-fingertip point name)
_EYEBALL = [
    ("dwpose", COCO_WHOLEBODY, "left_wrist", "left_middle_finger4"),
    ("mediapipe", MEDIAPIPE_HOLISTIC, "pose_15", "left_hand_12"),  # BlazePose L wrist; MP L middle tip
]


def _best_left_hand_frame(extractor, left_idx):
    """The single frame (across clips) with the most confident left-hand detection,
    so the eyeball sees a real spread-out hand rather than a collapsed low-score one."""
    left_idx = sorted(left_idx)
    best = (-1.0, None, None)  # (mean left-hand score, clip, frame keypoints)
    for npz in sorted(glob.glob(str(REPO / f"data/cache/{extractor}/*.npz")))[:60]:
        with np.load(npz) as d:
            scores, keypoints = d["scores"], d["keypoints"]
        per_frame = scores[:, left_idx].mean(axis=1)
        f = int(np.argmax(per_frame))
        if per_frame[f] > best[0]:
            best = (float(per_frame[f]), Path(npz).name, keypoints[f])
    return best


def test_print_hand_anchor_pixels():
    for extractor, sk, body_wrist_name, tip_name in _EYEBALL:
        w = sk.anchor("left_hand_wrist")
        mcp = sk.anchor("left_hand_middle_mcp")
        bw = sk.names.index(body_wrist_name)
        tip = sk.names.index(tip_name)
        left_idx, _ = _hand_blocks("coco_wholebody" if sk is COCO_WHOLEBODY else "mp", sk)
        conf, clip, kp = _best_left_hand_frame(extractor, left_idx)
        print(f"\n[{extractor}] {clip} (left-hand conf {conf:.2f})")
        if kp is None:
            continue
        print(f"   left_hand_wrist  {np.round(kp[w], 1)}")
        print(f"   body left_wrist  {np.round(kp[bw], 1)}   (should ~= left_hand_wrist)")
        print(f"   middle MCP       {np.round(kp[mcp], 1)}   (should be between wrist and tip)")
        print(f"   middle tip       {np.round(kp[tip], 1)}")
        # quantitative: MCP must sit far from the tip relative to wrist->tip span
        wrist_to_tip = float(np.linalg.norm(kp[tip] - kp[w]) + 1e-6)
        mcp_to_tip = float(np.linalg.norm(kp[tip] - kp[mcp]))
        print(f"   mcp_to_tip / wrist_to_tip = {mcp_to_tip / wrist_to_tip:.2f}  "
              f"(near 0 would mean MCP == fingertip -> wrong anchor)")


if __name__ == "__main__":
    passed = failed = 0
    for _name, fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {_name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
