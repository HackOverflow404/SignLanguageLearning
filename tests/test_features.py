"""features.py: one cached clip -> (T, F); block map covers F; toggles change F
predictably; synthetic translate+scale invariance of the global-derived features.

Runs under pytest OR as a plain script (`python tests/test_features.py`).
"""
import glob
from pathlib import Path

import numpy as np

from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.features import FeaturePipeline, Standardizer, to_fixed_length
from aslcv.normalizer.shoulder import ShoulderNormalizer

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
    sc = np.ones((n_frames, k), dtype=np.float32)
    return kp, sc


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
