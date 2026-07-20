"""Phase 1 "done when" tests for the dataset loader.

Runs under pytest OR as a plain script (`python tests/test_dataset.py`).

Phase 1 is done when this passes: every curriculum sign has a loadable reference
sequence in every split, the splits are signer-independent, and every sign has
non-null phonological features.
"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import yaml

from aslcv.dataset import SPLITS, load_index, load_pose, load_phonology, iter_records, PoseSequence

REPO = Path(__file__).resolve().parents[1]
EXTRACTOR = "mediapipe"   # default backend; all four caches are complete + verified

# Core first-morpheme parameters ASL-LEX codes for every sign. "NA" is a valid,
# non-null value (e.g. a one-handed sign's non-dominant handshape), so we require
# these always-applicable params to be present rather than every column.
CORE_PARAMS = ["handshape", "selected_fingers", "flexion", "major_location",
               "minor_location", "movement", "sign_type"]


def curriculum_signs():
    doc = yaml.safe_load(open(REPO / "curriculum.yaml", encoding="utf-8"))
    return {s["asllex_code"]: s["gloss"] for u in doc["units"] for s in u["signs"]}


def _count(split):
    c = defaultdict(int)
    for r in load_index(EXTRACTOR, split):
        c[r["asllex_code"]] += 1
    return c


def test_every_sign_loads_in_each_split():
    signs = curriculum_signs()
    for split in SPLITS:
        by_code = defaultdict(list)
        for r in load_index(EXTRACTOR, split):
            by_code[r["asllex_code"]].append(r)
        for code, gloss in signs.items():
            recs = by_code.get(code, [])
            assert recs, f"sign {gloss} ({code}) has no clip in split {split!r}"
            pose = load_pose(EXTRACTOR, recs[0]["video_id"])   # actually load one
            assert pose.n_frames >= 1, f"{gloss}/{split}: empty sequence"
            assert pose.n_keypoints > 0, f"{gloss}/{split}: no keypoints"


def test_no_signer_crosses_splits():
    signers = {sp: {r["signer_id"] for r in load_index(EXTRACTOR, sp)} for sp in SPLITS}
    for a, b in combinations(SPLITS, 2):
        overlap = signers[a] & signers[b]
        assert not overlap, f"signer(s) in both {a} and {b}: {sorted(overlap)}"


def test_phonology_nonnull_for_all_signs():
    phon = load_phonology()
    for code, gloss in curriculum_signs().items():
        row = phon.get(code)
        assert row is not None, f"no phonology row for {gloss} ({code})"
        for p in CORE_PARAMS:
            assert row.get(p, "").strip(), f"{gloss} ({code}) has null phonology param {p!r}"


def test_loader_yields_expected_tuple():
    # public contract: (pose_sequence, id_gloss, phonological_features, signer_id)
    pose, id_gloss, phon, signer = next(iter_records(EXTRACTOR, "val"))
    assert isinstance(pose, PoseSequence) and pose.n_frames >= 1
    assert isinstance(id_gloss, str) and id_gloss
    assert phon is not None and phon["asllex_code"]
    assert isinstance(signer, str) and signer


def test_print_per_sign_count():
    signs = curriculum_signs()
    counts = {sp: _count(sp) for sp in SPLITS}
    print(f"\n{'sign':<16}{'train':>7}{'val':>6}{'test':>6}")
    for code, gloss in sorted(signs.items(), key=lambda kv: kv[1]):
        print(f"{gloss:<16}{counts['train'].get(code, 0):>7}"
              f"{counts['val'].get(code, 0):>6}{counts['test'].get(code, 0):>6}")
    print(f"{'TOTAL':<16}{sum(counts['train'].values()):>7}"
          f"{sum(counts['val'].values()):>6}{sum(counts['test'].values()):>6}")


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
