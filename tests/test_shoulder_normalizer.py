"""ShoulderNormalizer: frame invariances + degeneracy, on both topologies.

Runs under pytest OR as a plain script (`python tests/test_shoulder_normalizer.py`).

The invariances are the point:
  * global block is translation + uniform-scale invariant (location/size of the whole
    body must not matter),
  * local hand block is invariant to translating the whole hand (handshape must not
    depend on where the hand is),
and they are asserted SEPARATELY so one can't mask the other. Both run against a
COCO-WholeBody and a MediaPipe skeleton to prove the anchor-based indexing is
cross-backend.
"""
import numpy as np

from aslcv.extractor.base import Pose
from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.normalizer.shoulder import ShoulderNormalizer, _hand_point_indices

SKELETONS = {"coco_wholebody": COCO_WHOLEBODY, "mediapipe_holistic": MEDIAPIPE_HOLISTIC}
ATOL = 1e-4


def _synthetic_pose(skeleton, seed=0):
    """Random keypoints, but with the anchor points pinned to non-degenerate spots
    (shoulder width 2, wrist->knuckle distance 1, elbow->body_wrist distance 1.5)
    so frames are always buildable."""
    rng = np.random.default_rng(seed)
    k = len(skeleton.names)
    kp = rng.uniform(0.0, 4.0, size=(k, 2)).astype(np.float32)
    kp[skeleton.anchor("left_shoulder")] = (1.0, 5.0)
    kp[skeleton.anchor("right_shoulder")] = (3.0, 5.0)  # width = 2
    for side in ("left", "right"):
        kp[skeleton.anchor(f"{side}_hand_wrist")] = (2.0, 1.0)
        kp[skeleton.anchor(f"{side}_hand_middle_mcp")] = (2.0, 2.0)  # scale = 1
        kp[skeleton.anchor(f"{side}_elbow")] = (2.0, 4.5)
        kp[skeleton.anchor(f"{side}_body_wrist")] = (2.0, 3.0)  # elbow->wrist = 1.5
    return Pose(keypoints=kp, scores=np.ones(k, dtype=np.float32))


def test_global_invariance_translation_and_scale():
    for name, sk in SKELETONS.items():
        norm = ShoulderNormalizer(local_hand=True)
        pose = _synthetic_pose(sk)
        g1, _ = norm.normalize(pose, sk).block("global")

        s, t = 1.7, np.array([2.5, -1.5], dtype=np.float32)
        moved = Pose(keypoints=(pose.keypoints * s + t).astype(np.float32), scores=pose.scores)
        g2, _ = norm.normalize(moved, sk).block("global")

        np.testing.assert_allclose(
            g1, g2, atol=ATOL, err_msg=f"{name}: global block not translation+scale invariant")


def test_local_hand_invariance_translation():
    for name, sk in SKELETONS.items():
        norm = ShoulderNormalizer(local_hand=True)
        pose = _synthetic_pose(sk)
        l1, _ = norm.normalize(pose, sk).block("left_hand")

        # move ONLY the left hand (its whole frame, incl. wrist + knuckle, shifts)
        idx = _hand_point_indices(sk, "left")
        kp = pose.keypoints.copy()
        kp[idx] += np.array([10.0, -7.0], dtype=np.float32)
        moved = Pose(keypoints=kp, scores=pose.scores)
        out = norm.normalize(moved, sk)
        l2, _ = out.block("left_hand")

        np.testing.assert_allclose(
            l1, l2, atol=ATOL, err_msg=f"{name}: local hand block not translation invariant")
        # different property from the global test: moving the hand DID shift its
        # global position (so the two invariances are genuinely distinct)
        g0, _ = norm.normalize(pose, sk).block("global")
        g_moved, _ = out.block("global")
        assert not np.allclose(g0[idx], g_moved[idx]), f"{name}: hand's global position should have moved"


def test_degenerate_inputs_zero_fill_no_crash():
    for name, sk in SKELETONS.items():
        norm = ShoulderNormalizer(local_hand=True)
        pose = _synthetic_pose(sk)
        ls, rs = sk.anchor("left_shoulder"), sk.anchor("right_shoulder")

        # (a) zero shoulder width -> coincident shoulders
        kp = pose.keypoints.copy(); kp[rs] = kp[ls]
        g, gs = norm.normalize(Pose(kp, pose.scores.copy()), sk).block("global")
        assert np.all(g == 0) and np.all(gs == 0), f"{name}: zero-width global not zero-filled"

        # (b) undetected shoulders (score 0)
        sc = pose.scores.copy(); sc[ls] = 0.0; sc[rs] = 0.0
        g, gs = norm.normalize(Pose(pose.keypoints.copy(), sc), sk).block("global")
        assert np.all(g == 0) and np.all(gs == 0), f"{name}: score-0 shoulders global not zero-filled"

        # (c) undetected hand (wrist + knuckle score 0) -> that hand's block zero-filled
        sc = pose.scores.copy()
        sc[sk.anchor("left_hand_wrist")] = 0.0
        sc[sk.anchor("left_hand_middle_mcp")] = 0.0
        lh, lhs = norm.normalize(Pose(pose.keypoints.copy(), sc), sk).block("left_hand")
        assert np.all(lh == 0) and np.all(lhs == 0), f"{name}: undetected hand block not zero-filled"


def test_local_hand_toggle_off_emits_only_global():
    for name, sk in SKELETONS.items():
        out = ShoulderNormalizer(local_hand=False).normalize(_synthetic_pose(sk), sk)
        assert set(out.blocks) == {"global"}, f"{name}: local_hand=False must emit only the global block"


def test_scales_computed_correctly():
    """Shoulder width 2, wrist->knuckle 1 -> hand ratio 0.5; elbow->body_wrist 1.5
    -> arm ratio 0.75. Both sides, both topologies."""
    for name, sk in SKELETONS.items():
        out = ShoulderNormalizer(local_hand=True).normalize(_synthetic_pose(sk), sk)
        for side in ("left", "right"):
            assert abs(out.scales[f"{side}_hand"] - 0.5) < ATOL, (
                f"{name}: {side}_hand scale wrong: {out.scales[f'{side}_hand']}")
            assert abs(out.scales[f"{side}_arm"] - 0.75) < ATOL, (
                f"{name}: {side}_arm scale wrong: {out.scales[f'{side}_arm']}")


def test_scales_computed_even_when_local_hand_off():
    """Hand/arm scale ratios are body-level measurements, not tied to the local-hand
    SHAPE blocks -- must still be populated when local_hand=False."""
    for name, sk in SKELETONS.items():
        out = ShoulderNormalizer(local_hand=False).normalize(_synthetic_pose(sk), sk)
        assert set(out.blocks) == {"global"}
        assert abs(out.scales["left_hand"] - 0.5) < ATOL, name
        assert abs(out.scales["left_arm"] - 0.75) < ATOL, name


def test_scales_invariant_to_translation_and_scale():
    """Same property as the global block: a ratio of two distances is invariant to
    uniform scale (both scale together) and translation (distance ignores origin)."""
    for name, sk in SKELETONS.items():
        norm = ShoulderNormalizer(local_hand=True)
        pose = _synthetic_pose(sk)
        s1 = norm.normalize(pose, sk).scales

        s, t = 1.7, np.array([2.5, -1.5], dtype=np.float32)
        moved = Pose(keypoints=(pose.keypoints * s + t).astype(np.float32), scores=pose.scores)
        s2 = norm.normalize(moved, sk).scales

        for key in s1:
            assert abs(s1[key] - s2[key]) < ATOL, f"{name}: scales[{key}] not scale/translation invariant"


def test_scales_zero_filled_on_degenerate_hand():
    """An undetected hand (wrist+knuckle score 0) must yield scale 0.0 for that
    hand specifically, matching the point-block zero-fill convention -- and must
    NOT contaminate the other hand or the arm scales."""
    for name, sk in SKELETONS.items():
        norm = ShoulderNormalizer(local_hand=True)
        pose = _synthetic_pose(sk)
        sc = pose.scores.copy()
        sc[sk.anchor("left_hand_wrist")] = 0.0
        sc[sk.anchor("left_hand_middle_mcp")] = 0.0
        out = norm.normalize(Pose(pose.keypoints.copy(), sc), sk)
        assert out.scales["left_hand"] == 0.0, name
        assert abs(out.scales["right_hand"] - 0.5) < ATOL, name
        assert abs(out.scales["left_arm"] - 0.75) < ATOL, (
            f"{name}: left hand degeneracy must not affect left arm scale")


def test_scales_zero_filled_when_shoulder_width_degenerate():
    """A ratio against a degenerate denominator (shoulder width) is meaningless --
    must be 0.0, not a divide-by-near-zero blowup, even though the hand distance
    itself is perfectly valid."""
    for name, sk in SKELETONS.items():
        norm = ShoulderNormalizer(local_hand=True)
        pose = _synthetic_pose(sk)
        kp = pose.keypoints.copy()
        rs = sk.anchor("right_shoulder")
        kp[rs] = kp[sk.anchor("left_shoulder")]  # coincident shoulders -> width 0
        out = norm.normalize(Pose(kp, pose.scores.copy()), sk)
        assert out.scales["left_hand"] == 0.0, name
        assert out.scales["left_arm"] == 0.0, name


def test_scores_and_blendshapes_carry_through():
    sk = MEDIAPIPE_HOLISTIC
    pose = _synthetic_pose(sk)
    bs = np.arange(52, dtype=np.float32)
    pose.blendshapes = bs
    out = ShoulderNormalizer(local_hand=True).normalize(pose, sk)
    assert out.blendshapes is bs, "blendshapes must pass through unchanged"
    # a present hand point carries its score into BOTH the global and local blocks
    gk, gsc = out.block("global")
    lk, lsc = out.block("left_hand")
    assert np.all(gsc[_hand_point_indices(sk, "left")] > 0) and np.all(lsc > 0)


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
