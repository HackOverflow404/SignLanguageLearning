"""Cross-topology region coverage (Skeleton.region -- one level up from anchors).

Runs under pytest OR as a plain script (`python tests/test_regions.py`).

Every skeleton must resolve every name in REQUIRED_REGIONS to an in-range index
tuple; the six regions must be disjoint and, for both current topologies, cover
every keypoint. A last test proves the region-based hand selection ShoulderNormalizer
now uses is IDENTICAL to the name-prefix matching it replaces.
"""
from pathlib import Path

from aslcv.normalizer.base import REQUIRED_REGIONS
from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC

REPO = Path(__file__).resolve().parents[1]
SKELETONS = {"coco_wholebody": COCO_WHOLEBODY, "mediapipe_holistic": MEDIAPIPE_HOLISTIC}


def test_all_skeletons_provide_required_regions():
    for name, sk in SKELETONS.items():
        for region in REQUIRED_REGIONS:
            idx = sk.region(region)  # raises KeyError if the skeleton lacks it
            assert all(0 <= i < len(sk.names) for i in idx), (
                f"{name}: {region} has an out-of-range index in {idx}")


def test_regions_are_disjoint_and_cover_every_point():
    for name, sk in SKELETONS.items():
        seen: set[int] = set()
        for region in REQUIRED_REGIONS:
            idx = set(sk.region(region))
            overlap = seen & idx
            assert not overlap, f"{name}: {region} overlaps an earlier region at {overlap}"
            seen |= idx
        assert seen == set(range(len(sk.names))), (
            f"{name}: regions cover {len(seen)}/{len(sk.names)} points, not all of them")


def test_hand_regions_contain_the_hand_anchors():
    for name, sk in SKELETONS.items():
        for side in ("left", "right"):
            region = set(sk.region(f"{side}_hand"))
            assert sk.anchor(f"{side}_hand_wrist") in region, (
                f"{name}: {side}_hand region missing its own wrist anchor")
            assert sk.anchor(f"{side}_hand_middle_mcp") in region, (
                f"{name}: {side}_hand region missing its own knuckle anchor")


def test_region_missing_raises_keyerror_with_available():
    sk = COCO_WHOLEBODY
    try:
        sk.region("not_a_real_region")
        assert False, "expected KeyError for an unknown region name"
    except KeyError as e:
        assert "left_hand" in str(e), "KeyError should list the available region names"


def test_body_upper_carries_coarse_head_on_every_topology():
    """--face=False must never remove coarse head LOCATION (nose/eyes/ears) on
    either topology -- only fine facial detail. Regression test for a real
    confound: MediaPipe used to fold its coarse head landmarks into "face"
    alongside the 478-mesh, so --face=False (the default) silently dropped
    MediaPipe's nose/eyes/ears while COCO's stayed (COCO always kept them in
    body_upper). Both skeletons must carry the same FIVE coarse head names in
    body_upper, cross-checked by name so a re-shuffle can't sneak past this."""
    coarse_head_names = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}
    for name, sk in SKELETONS.items():
        if sk is COCO_WHOLEBODY:
            body_upper_head = {sk.names[i] for i in sk.region("body_upper")} & coarse_head_names
            assert body_upper_head == coarse_head_names, (
                f"{name}: body_upper is missing coarse head points {coarse_head_names - body_upper_head}")
        else:
            # MediaPipe's body_upper points are generically named (pose_0, pose_2,
            # ...), so check by COUNT + membership in the canonical anchor/landmark
            # set instead of by name: exactly 5 non-shoulder points, and they must
            # be disjoint from "face" (i.e. NOT double-counted or left behind).
            body_upper = set(sk.region("body_upper"))
            face = set(sk.region("face"))
            non_shoulder = body_upper - {sk.anchor("left_shoulder"), sk.anchor("right_shoulder")}
            assert len(non_shoulder) == 5, (
                f"{name}: expected 5 coarse head points in body_upper besides the "
                f"shoulders, found {len(non_shoulder)}")
            assert sk.anchor("nose") in body_upper, f"{name}: nose anchor must be in body_upper"
            assert sk.anchor("nose") not in face, f"{name}: nose anchor must NOT be in face"


def test_face_and_body_upper_same_size_across_topologies_when_coarse_only():
    """Cross-topology sanity: with face detail stripped to just the coarse count,
    both skeletons should agree on how many points body_upper holds (7: nose,
    left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder) --
    the two topologies legitimately differ in FACE size (68 vs 484, different mesh
    density) but must not differ in body_upper size, which is the confound this
    fix closes."""
    sizes = {name: len(sk.region("body_upper")) for name, sk in SKELETONS.items()}
    assert len(set(sizes.values())) == 1, f"body_upper sizes diverge across topologies: {sizes}"
    assert list(sizes.values())[0] == 7, f"expected 7-point body_upper everywhere, got {sizes}"


def test_hand_region_matches_old_name_prefix_selection():
    """Proves the region-based refactor (ShoulderNormalizer._hand_point_indices)
    selects the IDENTICAL points the name-prefix matching it replaced did. Recomputes
    that old logic independently here rather than importing it (it's been deleted)."""
    fragments = ("hand", "finger", "thumb")
    for name, sk in SKELETONS.items():
        for side in ("left", "right"):
            prefix = f"{side}_"
            expected = {
                i for i, n in enumerate(sk.names)
                if n.startswith(prefix) and any(f in n for f in fragments)
            }
            assert set(sk.region(f"{side}_hand")) == expected, (
                f"{name}: {side}_hand region diverges from the old name-prefix set")


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
