"""features.py: one cached clip -> (T, F); block map covers F; toggles change F
predictably; synthetic translate+scale invariance of the global-derived features.

Runs under pytest OR as a plain script (`python tests/test_features.py`).
"""
import glob
from pathlib import Path

import numpy as np

from aslcv.extractor.base import Pose
from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.features import FeaturePipeline, Standardizer, hand_motion_energy, motion_active_span, to_fixed_length
from aslcv.normalizer.shoulder import ShoulderNormalizer, _hand_point_indices

REPO = Path(__file__).resolve().parents[1]
# (cache dir under data/cache, skeleton)
BACKENDS = [("dwpose", COCO_WHOLEBODY), ("mediapipe", MEDIAPIPE_HOLISTIC)]
ATOL = 1e-4


def _first_npz(extractor):
    hits = sorted(glob.glob(str(REPO / f"data/cache/{extractor}/*.npz")))
    return hits[0] if hits else None


def _pipe(skeleton, **kw):
    return FeaturePipeline(ShoulderNormalizer(local_hand=kw.pop("local_hand", True)), skeleton, **kw)


def _synthetic_clip(skeleton, n_frames=6, seed=0):
    rng = np.random.default_rng(seed)
    k = len(skeleton.names)
    kp = rng.uniform(0.0, 4.0, size=(n_frames, k, 2)).astype(np.float32)
    for t in range(n_frames):  # pin anchors non-degenerate (width 2, wrist->knuckle 1)
        kp[t, skeleton.anchor("left_shoulder")] = (1.0, 5.0)
        kp[t, skeleton.anchor("right_shoulder")] = (3.0, 5.0)
        for side in ("left", "right"):
            kp[t, skeleton.anchor(f"{side}_hand_wrist")] = (2.0, 1.0)
            kp[t, skeleton.anchor(f"{side}_hand_middle_mcp")] = (2.0, 2.0)
            # elbow -> body_wrist distance 2 (shoulder width 2 -> arm ratio 1.0)
            kp[t, skeleton.anchor(f"{side}_elbow")] = (2.0, 3.0)
            kp[t, skeleton.anchor(f"{side}_body_wrist")] = (2.0, 1.0)
    sc = np.ones((n_frames, k), dtype=np.float32)
    return kp, sc


def _rest_active_rest_clip(skeleton, n_rest_head=5, n_active=8, n_rest_tail=5, seed=0):
    """A clip with clean REST -> ACTIVE -> REST structure: every hand-region point
    frozen during the two rest spans, moving randomly during the active span --
    ground truth for hand_motion_energy / motion_active_span / trim_to_motion.
    Shoulders + wrist/mcp anchors stay pinned every frame (non-degenerate frame),
    but the REST of each hand's points obey the rest/active/rest pattern, so the
    hand block's average velocity is exactly 0 at rest and (with high probability,
    being random) nonzero while active."""
    rng = np.random.default_rng(seed)
    n_frames = n_rest_head + n_active + n_rest_tail
    k = len(skeleton.names)
    kp = np.zeros((n_frames, k, 2), dtype=np.float32)
    hand_idx = list(_hand_point_indices(skeleton, "left")) + list(_hand_point_indices(skeleton, "right"))
    rest_head_pos = rng.uniform(1.5, 2.5, size=(len(hand_idx), 2)).astype(np.float32)
    rest_tail_pos = rng.uniform(1.5, 2.5, size=(len(hand_idx), 2)).astype(np.float32)
    for t in range(n_frames):
        if t < n_rest_head:
            kp[t, hand_idx] = rest_head_pos
        elif t >= n_rest_head + n_active:
            kp[t, hand_idx] = rest_tail_pos
        else:
            kp[t, hand_idx] = rng.uniform(0.0, 4.0, size=(len(hand_idx), 2)).astype(np.float32)
    kp[:, skeleton.anchor("left_shoulder")] = (1.0, 5.0)
    kp[:, skeleton.anchor("right_shoulder")] = (3.0, 5.0)
    for side in ("left", "right"):
        kp[:, skeleton.anchor(f"{side}_hand_wrist")] = (2.0, 1.0)
        kp[:, skeleton.anchor(f"{side}_hand_middle_mcp")] = (2.0, 2.0)
    sc = np.ones((n_frames, k), dtype=np.float32)
    return kp, sc, n_rest_head, n_active, n_rest_tail


def test_hand_motion_energy_zero_at_rest_nonzero_while_active():
    for _extractor, sk in BACKENDS:
        kp, sc, n_head, n_active, n_tail = _rest_active_rest_clip(sk)
        poses = [Pose(kp[t], sc[t]) for t in range(kp.shape[0])]
        energy = hand_motion_energy(ShoulderNormalizer(local_hand=True), sk, poses)
        # rest-head: frame 0 is always 0 by convention; frames 1..n_head-1 are a
        # frozen span, so their delta from the PREVIOUS (also frozen) frame is 0
        assert np.allclose(energy[1:n_head], 0.0), f"rest-head not zero: {energy[:n_head]}"
        # rest-tail: same logic, except the FIRST tail frame is a real transition
        # (last active frame -> first rest frame) and is expected to be nonzero
        assert np.allclose(energy[n_head + n_active + 1:], 0.0), f"rest-tail not zero: {energy[n_head+n_active:]}"
        # active span (excluding its own first frame, which transitions FROM rest)
        # should be mostly nonzero -- random content, vanishingly unlikely to tie
        active_mid = energy[n_head + 1:n_head + n_active]
        assert (active_mid > 0).mean() > 0.5, f"active span mostly zero: {active_mid}"


def test_motion_active_span_finds_the_active_window():
    for _extractor, sk in BACKENDS:
        kp, sc, n_head, n_active, n_tail = _rest_active_rest_clip(sk)
        poses = [Pose(kp[t], sc[t]) for t in range(kp.shape[0])]
        energy = hand_motion_energy(ShoulderNormalizer(local_hand=True), sk, poses)
        start, stop = motion_active_span(energy, threshold=1e-6, pad_frames=0)
        # the span must START at or before the transition-in frame and STOP at or
        # after the transition-out frame -- i.e. it must not cut into real motion
        assert start <= n_head, f"trimmed into the rest-head padding incorrectly: start={start}"
        assert stop >= n_head + n_active, f"trimmed into active content: stop={stop}"


def test_motion_active_span_never_trims_to_empty():
    """A clip with NO frame clearing threshold (e.g. threshold set absurdly high)
    must return the full range unchanged, never an empty span."""
    energy = np.array([0.01, 0.02, 0.015, 0.01], dtype=np.float32)
    start, stop = motion_active_span(energy, threshold=1.0, pad_frames=0)
    assert (start, stop) == (0, len(energy))


def test_trim_to_motion_off_by_default():
    for _extractor, sk in BACKENDS:
        pipe = FeaturePipeline(ShoulderNormalizer(local_hand=True), sk)
        assert pipe.trim_to_motion is False


def test_trim_to_motion_shortens_T_on_rest_active_rest_clip():
    for _extractor, sk in BACKENDS:
        kp, sc, n_head, n_active, n_tail = _rest_active_rest_clip(sk)
        untrimmed = _pipe(sk, trim_to_motion=False).assemble_clip(kp, sc)
        trimmed = _pipe(sk, trim_to_motion=True, motion_threshold=1e-6, motion_pad_frames=0).assemble_clip(kp, sc)
        assert untrimmed.features.shape[0] == n_head + n_active + n_tail
        assert trimmed.features.shape[0] < untrimmed.features.shape[0], (
            "trim_to_motion should have dropped rest padding")
        # F (feature width per frame) must be UNCHANGED by trimming -- only T shrinks
        assert trimmed.features.shape[1] == untrimmed.features.shape[1]


def test_trim_to_motion_never_empties_a_clip():
    """A clip that's ENTIRELY rest (motion never clears threshold) must still
    assemble to at least 1 frame -- assemble() itself rejects 0 frames."""
    for _extractor, sk in BACKENDS:
        kp, sc = _synthetic_clip(sk, n_frames=4)  # random per frame, but threshold set unreachable
        clip = _pipe(sk, trim_to_motion=True, motion_threshold=1e9).assemble_clip(kp, sc)
        assert clip.features.shape[0] == 4  # nothing cleared threshold -> full clip kept


def test_cached_clip_to_TF():
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        assert npz, f"no cached clips under data/cache/{extractor}"
        with np.load(npz) as d:
            n_frames = d["keypoints"].shape[0]
        clip = _pipe(sk).assemble_npz(npz)
        assert clip.features.ndim == 2, f"{extractor}: features must be 2-D (T, F)"
        assert clip.features.shape[0] == n_frames, f"{extractor}: T mismatch"
        assert clip.features.shape[1] > 0 and clip.features.dtype == np.float32
        assert np.isfinite(clip.features).all(), f"{extractor}: non-finite features"


def test_block_map_covers_F():
    for extractor, sk in BACKENDS:
        clip = _pipe(sk).assemble_npz(_first_npz(extractor))
        F = clip.features.shape[1]
        spans = sorted((s.start, s.stop) for s in clip.blocks.values())
        # contiguous from 0, non-overlapping, exactly covering F
        assert spans[0][0] == 0 and spans[-1][1] == F, f"{extractor}: blocks don't span [0, F)"
        covered = sum(b - a for a, b in spans)
        assert covered == F, f"{extractor}: block slices cover {covered}, F={F}"
        for (a0, b0), (a1, b1) in zip(spans, spans[1:]):
            assert b0 == a1, f"{extractor}: gap/overlap between block slices"
        assert set(clip.blocks) == {"global", "left_hand", "right_hand"}


def test_velocity_toggle_changes_F():
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        on = _pipe(sk, velocity=True).assemble_npz(npz)
        off = _pipe(sk, velocity=False).assemble_npz(npz)
        assert on.channels_per_point == 5 and off.channels_per_point == 3
        assert on.features.shape[1] * 3 == off.features.shape[1] * 5, f"{extractor}: F ratio != 5/3"


def test_local_hand_toggle_changes_F():
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        with_hands = _pipe(sk, local_hand=True).assemble_npz(npz)
        no_hands = _pipe(sk, local_hand=False).assemble_npz(npz)
        assert set(with_hands.blocks) == {"global", "left_hand", "right_hand"}
        assert set(no_hands.blocks) == {"global"}
        # dropping both local hand blocks removes 42 points * channels_per_point
        diff = with_hands.features.shape[1] - no_hands.features.shape[1]
        assert diff == 42 * with_hands.channels_per_point, f"{extractor}: hand-block F delta wrong"


def test_depth_proxies_off_by_default():
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        clip = _pipe(sk).assemble_npz(npz)
        assert "depth_proxies" not in clip.blocks, f"{extractor}: depth_proxies must be off by default"


def test_depth_proxies_toggle_adds_exactly_4_columns():
    from aslcv.features import DEPTH_PROXY_KEYS
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        off = _pipe(sk, depth_proxies=False).assemble_npz(npz)
        on = _pipe(sk, depth_proxies=True).assemble_npz(npz)
        assert "depth_proxies" not in off.blocks
        assert "depth_proxies" in on.blocks
        diff = on.features.shape[1] - off.features.shape[1]
        assert diff == len(DEPTH_PROXY_KEYS), f"{extractor}: depth_proxies added {diff} cols, expected {len(DEPTH_PROXY_KEYS)}"
        sl = on.blocks["depth_proxies"]
        assert sl.stop == on.features.shape[1], f"{extractor}: depth_proxies must be the trailing block"
        assert sl.stop - sl.start == len(DEPTH_PROXY_KEYS)


def test_depth_proxies_synthetic_values_match_expected_ratios():
    """On the synthetic clip (shoulder width 2, wrist->mcp 1, elbow->body_wrist 2),
    hand ratio should be exactly 0.5 and arm ratio exactly 1.0, every frame -- this
    is the actual computation, not just a shape check."""
    from aslcv.features import DEPTH_PROXY_KEYS
    for _extractor, sk in BACKENDS:
        kp, sc = _synthetic_clip(sk)
        clip = _pipe(sk, depth_proxies=True).assemble_clip(kp, sc)
        block = clip.features[:, clip.blocks["depth_proxies"]]
        for i, key in enumerate(DEPTH_PROXY_KEYS):
            expected = 0.5 if key.endswith("_hand") else 1.0
            np.testing.assert_allclose(
                block[:, i], expected, atol=ATOL, err_msg=f"{key}: expected {expected}")


def test_depth_proxies_zero_when_degenerate():
    """A hand with score 0 (absent) must yield ratio 0.0, not garbage or NaN --
    same zero-fill convention as every other degenerate case in this pipeline."""
    for _extractor, sk in BACKENDS:
        kp, sc = _synthetic_clip(sk, n_frames=1)
        sc = sc.copy()
        sc[0, sk.anchor("left_hand_wrist")] = 0.0
        clip = _pipe(sk, depth_proxies=True).assemble_clip(kp, sc)
        from aslcv.features import DEPTH_PROXY_KEYS
        block = clip.features[:, clip.blocks["depth_proxies"]]
        i = DEPTH_PROXY_KEYS.index("left_hand")
        assert block[0, i] == 0.0
        assert np.isfinite(block).all()


def test_depth_proxies_on_real_cached_clips_finite():
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        clip = _pipe(sk, depth_proxies=True).assemble_npz(npz)
        block = clip.features[:, clip.blocks["depth_proxies"]]
        assert np.isfinite(block).all(), f"{extractor}: non-finite depth-proxy values"
        assert (block >= 0).all(), f"{extractor}: depth-proxy ratios must be non-negative"


def test_binary_mode_same_F_and_binarizes_conf():
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        graded = _pipe(sk, confidence="graded").assemble_npz(npz)
        binary = _pipe(sk, confidence="binary").assemble_npz(npz)
        assert graded.features.shape == binary.features.shape, f"{extractor}: binary changed F"
        C = binary.channels_per_point
        conf = binary.features[:, 2::C]  # confidence channel of every point
        assert np.isin(conf, (0.0, 1.0)).all(), f"{extractor}: binary conf not in {{0,1}}"


def test_synthetic_global_invariance_translate_scale():
    for _extractor, sk in BACKENDS:
        pipe = _pipe(sk)  # face off (default), velocity on
        kp, sc = _synthetic_clip(sk)
        f1 = pipe.assemble_clip(kp, sc)
        s, t = 1.7, np.array([2.5, -1.5], dtype=np.float32)
        f2 = pipe.assemble_clip((kp * s + t).astype(np.float32), sc)
        g1 = f1.features[:, f1.blocks["global"]]
        g2 = f2.features[:, f2.blocks["global"]]
        np.testing.assert_allclose(
            g1, g2, atol=ATOL, err_msg=f"{sk.names[:0] or _extractor}: global features not invariant")


def test_face_toggle_grows_global_block():
    sk = MEDIAPIPE_HOLISTIC  # has a large face mesh
    npz = _first_npz("mediapipe")
    no_face = _pipe(sk, face=False).assemble_npz(npz)
    with_face = _pipe(sk, face=True).assemble_npz(npz)
    assert with_face.features.shape[1] > no_face.features.shape[1], "face=True should add face points"


def test_legs_feet_dropped_by_default():
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        default = _pipe(sk).assemble_npz(npz)
        explicit_off = _pipe(sk, legs_feet=False).assemble_npz(npz)
        assert default.features.shape[1] == explicit_off.features.shape[1], (
            f"{extractor}: default should already drop legs/feet")


def test_legs_feet_toggle_shrinks_global_block():
    for extractor, sk in BACKENDS:
        npz = _first_npz(extractor)
        with_legs = _pipe(sk, legs_feet=True).assemble_npz(npz)
        no_legs = _pipe(sk, legs_feet=False).assemble_npz(npz)
        assert with_legs.features.shape[1] > no_legs.features.shape[1], (
            f"{extractor}: legs_feet=True should add leg/foot points back")
        expected_extra_points = len(sk.region("legs_feet"))
        diff_points = (with_legs.features.shape[1] - no_legs.features.shape[1]) // with_legs.channels_per_point
        assert diff_points == expected_extra_points, (
            f"{extractor}: legs_feet F delta ({diff_points} pts) != region size ({expected_extra_points})")


def test_velocity_unchanged_when_always_present():
    """No presence gap -> the new masked velocity must equal the old raw diff exactly."""
    for _extractor, sk in BACKENDS:
        kp, sc = _synthetic_clip(sk, n_frames=6, seed=2)
        clip = _pipe(sk, legs_feet=True, face=True).assemble_clip(kp, sc)
        C = clip.channels_per_point
        g = clip.features[:, clip.blocks["global"]].reshape(6, -1, C)
        pos, vel = g[:, :, :2], g[:, :, 3:5]
        expected = np.zeros_like(pos)
        expected[1:] = pos[1:] - pos[:-1]
        np.testing.assert_allclose(vel, expected, atol=ATOL)


def test_velocity_zeroed_across_presence_gap():
    """A hand that disappears for 2 frames and returns must show zero velocity at
    the disappearance frame, every absent frame, and the reappearance frame -- not
    a fake jump from differencing against the zero-fill -- while a genuinely
    moving, continuously-present point elsewhere keeps its real velocity."""
    for _extractor, sk in BACKENDS:
        n_frames = 8
        kp, sc = _synthetic_clip(sk, n_frames=n_frames, seed=3)
        hand_idx = list(sk.region("left_hand"))
        probe = hand_idx[0]
        for t in range(n_frames):  # genuine per-frame motion, so a real delta exists
            kp[t, probe] = kp[t, probe] + np.array([t * 3.0, t * 1.5], dtype=np.float32)
        gap = (3, 4)  # whole hand occluded on these frames
        for t in gap:
            for i in hand_idx:
                sc[t, i] = 0.0

        # keep everything in "global" (no face/legs drop) so its row order == the
        # skeleton's own index order, and `probe` can be read directly
        clip = _pipe(sk, legs_feet=True, face=True).assemble_clip(kp, sc)
        C = clip.channels_per_point
        g = clip.features[:, clip.blocks["global"]].reshape(n_frames, -1, C)
        vel = g[:, probe, 3:5]

        assert np.allclose(vel[0], 0.0), "first-frame delta must stay zero"
        assert np.allclose(vel[3], 0.0), "velocity at the disappearance frame must be zero"
        assert np.allclose(vel[4], 0.0), "velocity while absent must be zero"
        assert np.allclose(vel[5], 0.0), "velocity at the reappearance frame must be zero"
        assert not np.allclose(vel[2], 0.0), "continuously-present real velocity must not be zeroed"
        assert not np.allclose(vel[6], 0.0), "continuously-present real velocity after the gap must not be zeroed"


def test_standardizer_fit_transform_and_roundtrip(tmp_path=None):
    sk = COCO_WHOLEBODY
    pipe = _pipe(sk)
    train = [pipe.assemble_clip(*_synthetic_clip(sk, seed=i)) for i in range(3)]
    std = Standardizer.fit(train)
    # standardizing the exact data it was fit on -> ~zero-mean per non-constant feature
    all_frames = np.concatenate([c.features for c in train], axis=0)
    z = std.transform(all_frames)
    nonconst = std.std > 1e-6
    assert np.abs(z[:, nonconst].mean(axis=0)).max() < 1e-3
    # save/load round-trip
    out = REPO / "data" / "cache" / "_test_standardizer.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    std.save(out)
    loaded = Standardizer.load(out)
    assert np.allclose(loaded.mean, std.mean) and np.allclose(loaded.std, std.std)
    out.unlink()
    # to_fixed_length pads/crops the frame axis, keeps F
    padded = to_fixed_length(train[0].features, train[0].features.shape[0] + 5)
    cropped = to_fixed_length(train[0].features, 2)
    assert padded.shape == (train[0].features.shape[0] + 5, train[0].features.shape[1])
    assert cropped.shape == (2, train[0].features.shape[1])


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
