"""Phase 4 grader tests.

Two tiers, same split used throughout this project's test suite:

  * phonology_labels tests -- deterministic, need only data/phonology.csv (no trained
    checkpoint). Pin the minimum-support gate's logic against REAL curriculum data.
  * EmbeddingGrader tests -- need a trained checkpoint
    (models/embedding_grader/model_best.pt, written by
    scripts/train_embedding_grader.py). SKIPPED if absent, same pattern
    tests/test_gloss_rules_corpus.py uses for PENDING_CASES: report what's
    verifiable now, promote once the artifact exists, never fail the whole suite
    for a training run nobody has been asked to do yet.

Runs under pytest OR as a plain script.
"""
import csv
from pathlib import Path

import pytest

from aslcv.grading.phonology_labels import MIN_SUPPORT, PhonologyLabels

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO / "models" / "embedding_grader"
CHECKPOINT_EXISTS = (CHECKPOINT_DIR / "model_best.pt").exists()


# -- phonology_labels: deterministic, no checkpoint needed -------------------


def test_mother_father_are_a_real_minimal_pair():
    """mother/father is curriculum.yaml's built-in minimal pair, differing ONLY in
    minor_location (Forehead vs Chin) -- same handshape and movement. Pin the
    exact values so the head-independence demo (grader-side) rests on a fact,
    not an assumption."""
    phon = PhonologyLabels()
    assert phon.label_for("mother", "handshape") == phon.label_for("father", "handshape")
    assert phon.label_for("mother", "movement") == phon.label_for("father", "movement")
    assert phon.label_for("mother", "minor_location") != phon.label_for("father", "minor_location")
    assert phon.label_for("mother", "minor_location") == "Chin"
    assert phon.label_for("father", "minor_location") == "Forehead"
    # both well-supported (5 signs each) -- a genuine testable minimal pair, not a thin one
    assert phon.well_supported("minor_location", "Chin")
    assert phon.well_supported("minor_location", "Forehead")


def test_singleton_class_is_not_well_supported():
    """time's major_location is 'Arm', carried by exactly 1 curriculum sign -- the
    canonical thin-class example the minimum-support gate exists for."""
    phon = PhonologyLabels()
    assert phon.label_for("time", "major_location") == "Arm"
    assert phon.support("major_location", "Arm") == 1
    assert not phon.well_supported("major_location", "Arm")
    assert MIN_SUPPORT == 3


def test_repeated_movement_is_balanced_at_60_signs():
    """The whole reason Phase 4 trains on 60 signs, not the 20-sign slice: repeated
    is 30/30 there, vs 16/4 skewed in-slice (see project_workflow.md's Phase 4
    section) -- pin the number that decision rests on."""
    phon = PhonologyLabels()
    counts = {v: phon.support("repeated_movement", v) for v in ("0", "1")}
    assert counts == {"0": 30, "1": 30}


# -- EmbeddingGrader: needs a trained checkpoint -----------------------------


pytestmark_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_EXISTS,
    reason=f"no checkpoint at {CHECKPOINT_DIR} -- run scripts/train_embedding_grader.py first",
)


def _rows_for(sign, split):
    rows = list(csv.DictReader(open(REPO / "data" / "manifest.csv")))
    return [r for r in rows if r["id_gloss"] == sign and r["split"] == split]


@pytest.fixture(scope="module")
def grader():
    """Explicitly CUDA when available -- NOT a speed choice.

    Root-caused (see full_suite crash trace + minimal repro): tests/test_dwpose_
    running_mode.py imports rtmlib/onnxruntime-gpu, which installs an NVBLAS hook
    that intercepts CPU BLAS calls process-wide; a torch CPU GRU forward pass
    (StreamEncoder's nn.GRU) running LATER in the same pytest process then segfaults
    inside torch's native RNN kernel when that hook mishandles the call ("NVBLAS
    ... cublasXtSgemm failed"). Confirmed the crash follows CPU execution specifically
    -- the identical model/data on CUDA does not crash even after rtmlib is imported
    first. So CUDA isn't an optimization here, it's what avoids a broken CPU BLAS
    path in this dependency stack. See test_embedding_model.py's `_device()` for the
    same fix applied to its synthetic-tensor tests."""
    from aslcv.grading.embedding_grader import EmbeddingGrader
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EmbeddingGrader.build(CHECKPOINT_DIR, which="best", device=device)


@pytestmark_checkpoint
def test_grade_ranks_all_signs_ascending(grader):
    row = _rows_for("mother", "val")[0]
    npz = REPO / "data" / "cache" / grader.extractor / f"{row['video_id']}.npz"
    ranked = grader.grade(npz)
    assert len(ranked) == len(grader.signs)
    dists = [d for _, d in ranked]
    assert dists == sorted(dists)


@pytestmark_checkpoint
def test_grade_against_unknown_sign_raises(grader):
    row = _rows_for("mother", "val")[0]
    npz = REPO / "data" / "cache" / grader.extractor / f"{row['video_id']}.npz"
    with pytest.raises(KeyError):
        grader.grade_against(npz, "not_a_real_sign")


@pytestmark_checkpoint
def test_grade_against_shape(grader):
    row = _rows_for("mother", "val")[0]
    npz = REPO / "data" / "cache" / grader.extractor / f"{row['video_id']}.npz"
    result = grader.grade_against(npz, "mother")
    assert result.target_sign == "mother"
    assert isinstance(result.fidelity, float) and result.fidelity >= 0.0
    expected_params = {"handshape", "major_location", "minor_location", "movement", "repeated_movement"}
    assert set(result.parameters) == expected_params
    for v in result.parameters.values():
        assert v.correct in (True, False, None)


@pytestmark_checkpoint
def test_thin_class_verdict_is_gated_regardless_of_model_output(grader):
    """time's major_location ('Arm') has support=1 -- ANY attempt graded against
    'time' must report major_location as "insufficient data" (correct=None), never
    a confident True/False, no matter what the model predicts. This is the
    minimum-support gate itself, checked against the real trained grader -- a
    deterministic property of phonology.csv, not of model quality, so it can't be
    flaky the way a specific prediction outcome could be."""
    row = _rows_for("time", "val")[0]
    npz = REPO / "data" / "cache" / grader.extractor / f"{row['video_id']}.npz"
    result = grader.grade_against(npz, "time")
    assert result.parameters["major_location"].correct is None
    assert result.parameters["major_location"].support == 1


@pytestmark_checkpoint
def test_heads_disagree_independently_on_a_real_minimal_pair(grader):
    """THE property this whole phase exists to demonstrate, on real data: grading a
    real 'father' attempt against 'mother' as target. mother/father differ ONLY in
    minor_location (Forehead vs Chin) -- same handshape, same movement, same
    repeated_movement. Empirically verified via scripts/eval_embedding_grader.py
    (see PHASE4_REPORT.md) before being pinned here as a regression: handshape,
    major_location, movement, and repeated_movement all correctly MATCH, while
    minor_location -- and ONLY minor_location -- disagrees. If the heads secretly
    shared a bottleneck (co-firing instead of being genuinely separable), a location
    mismatch would be far more likely to drag other parameters down with it; it
    doesn't, because PoseGraderNet's heads have no gradient path into each other's
    input streams (see test_embedding_model.py's structural test for why)."""
    row = _rows_for("father", "val")[0]
    npz = REPO / "data" / "cache" / grader.extractor / f"{row['video_id']}.npz"
    result = grader.grade_against(npz, "mother")

    assert result.parameters["handshape"].correct is True
    assert result.parameters["major_location"].correct is True
    assert result.parameters["movement"].correct is True
    assert result.parameters["repeated_movement"].correct is True
    assert result.parameters["minor_location"].correct is False
    assert result.parameters["minor_location"].predicted == "Forehead"  # father's real value
    assert result.parameters["minor_location"].target == "Chin"          # mother's real value


if __name__ == "__main__":
    passed = failed = skipped = 0
    for _name, fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(fn):
            try:
                import inspect
                sig = inspect.signature(fn)
                if "grader" in sig.parameters:
                    if not CHECKPOINT_EXISTS:
                        print(f"  SKIP {_name} (no checkpoint)")
                        skipped += 1
                        continue
                    from aslcv.grading.embedding_grader import EmbeddingGrader
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    fn(EmbeddingGrader.build(CHECKPOINT_DIR, which="best", device=device))
                else:
                    fn()
                print(f"  PASS {_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {_name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    raise SystemExit(1 if failed else 0)
