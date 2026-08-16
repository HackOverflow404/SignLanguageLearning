"""Phase 7 step 3: align_and_grade -- forced alignment against a known target
gloss sequence, then per-segment grading. Needs a trained checkpoint (same
skip-if-absent convention as test_embedding_grader.py) plus real cached
reference clips concatenated into a synthetic "continuous attempt" -- the
same trick step 4's validation benchmark will use at scale, exercised here
at unit-test size.

Runs under pytest OR as a plain script.
"""
from pathlib import Path

import pytest

from aslcv.grading.alignment import align_and_grade
from aslcv.grading.embedding_dataset import _load_poses_npz
from aslcv.production import GlossRuleEngine
from aslcv.production.gloss_rules import Gloss, GlossSequence
from aslcv.production.retrieval import fetch_reference

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO / "models" / "embedding_grader"
CHECKPOINT_EXISTS = (CHECKPOINT_DIR / "model_best.pt").exists()
pytestmark_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_EXISTS,
    reason=f"no checkpoint at {CHECKPOINT_DIR} -- run scripts/train_embedding_grader.py first",
)

E = GlossRuleEngine()


@pytest.fixture(scope="module")
def grader():
    """Explicitly CUDA when available -- same NVBLAS/CPU-BLAS reasoning as
    test_embedding_grader.py's fixture; see its docstring for the full trace."""
    from aslcv.grading.embedding_grader import EmbeddingGrader
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EmbeddingGrader.build(CHECKPOINT_DIR, which="best", device=device)


def _synthetic_continuous_attempt(gloss_sequence, extractor):
    """Concatenate the SAME real reference clips compose_reference_features
    would resolve -- a synthetic "continuous attempt" with a known
    per-gloss frame count, the same trick production.retrieval.
    ComposedReference uses for video, at unit-test scale."""
    poses = []
    for id_gloss in gloss_sequence.gloss_ids:
        clip = fetch_reference(id_gloss, extractor=extractor)
        poses.extend(_load_poses_npz(clip.npz_path))
    return poses


def _single_gloss_sequence(id_gloss):
    return GlossSequence(
        english=id_gloss, in_scope=True, confidence=1.0, reason=None,
        sentence_type="statement", negated=False,
        glosses=[Gloss(text=id_gloss, asllex_id=id_gloss, source=id_gloss, pos="NOUN")],
    )


@pytestmark_checkpoint
def test_align_and_grade_returns_one_result_per_gloss_in_order(grader):
    seq = E.gloss("I want water.")
    attempt_poses = _synthetic_continuous_attempt(seq, grader.extractor)
    distance, graded = align_and_grade(grader, attempt_poses, seq)

    assert [g.target_sign for g in graded] == seq.gloss_ids
    assert distance >= 0.0


@pytestmark_checkpoint
def test_align_and_grade_segments_are_monotonic_and_cover_the_attempt(grader):
    seq = E.gloss("I want water.")
    attempt_poses = _synthetic_continuous_attempt(seq, grader.extractor)
    _, graded = align_and_grade(grader, attempt_poses, seq)

    assert graded[0].frame_range[0] == 0
    assert graded[-1].frame_range[1] == len(attempt_poses)
    prev_stop = 0
    for g in graded:
        start, stop = g.frame_range
        assert 0 <= start < stop <= len(attempt_poses)
        assert start >= prev_stop - 1  # segments may touch but must not jump backwards
        prev_stop = stop


@pytestmark_checkpoint
def test_align_and_grade_grades_true_target_better_than_a_mismatched_one(grader):
    """The attempt IS the exact reference clip for each gloss, concatenated --
    each segment graded against its OWN true target should fidelity-rank
    better (lower distance) than the same segment graded against an
    unrelated sign, the same relative check test_dtw_grader.py's
    test_dtw_orders_by_similarity uses rather than pinning an absolute
    distance whose scale isn't documented anywhere."""
    seq = E.gloss("I want water.")
    attempt_poses = _synthetic_continuous_attempt(seq, grader.extractor)
    _, graded = align_and_grade(grader, attempt_poses, seq)

    water_segment = next(g for g in graded if g.target_sign == "water")
    start, stop = water_segment.frame_range
    mismatched = grader.grade_against_poses(attempt_poses[start:stop], "mother")
    assert water_segment.result.fidelity < mismatched.fidelity


@pytestmark_checkpoint
def test_align_and_grade_single_gloss_sequence_needs_no_special_casing(grader):
    single = _single_gloss_sequence("mother")
    attempt_poses = _synthetic_continuous_attempt(single, grader.extractor)
    distance, graded = align_and_grade(grader, attempt_poses, single)

    assert len(graded) == 1
    assert graded[0].target_sign == "mother"
    assert graded[0].frame_range == (0, len(attempt_poses))


@pytestmark_checkpoint
def test_align_and_grade_refuses_empty_attempt(grader):
    seq = E.gloss("I want water.")
    with pytest.raises(ValueError):
        align_and_grade(grader, [], seq)


if __name__ == "__main__":
    import sys
    import torch
    from aslcv.grading.embedding_grader import EmbeddingGrader

    failures = 0
    if not CHECKPOINT_EXISTS:
        print(f"  SKIP all tests (no checkpoint at {CHECKPOINT_DIR})")
        sys.exit(0)
    g = EmbeddingGrader.build(CHECKPOINT_DIR, which="best",
                               device="cuda" if torch.cuda.is_available() else "cpu")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(g)
                print(f"  OK   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
