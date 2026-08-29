"""Phase 7 step 3 -- align_and_grade: the orchestrator tying dtw_align
(step 1) and compose_reference_features (step 2) together into forced
alignment + per-sign grading for a live continuous attempt.

Reframed problem (see project_workflow.md's Phase 7 section for the full
reasoning): the target gloss sequence is always known in advance (this
system generated it), so this is forced alignment -- align a live attempt
against a known reference -- not open-set continuous recognition.

Nothing here re-implements alignment or grading: it DTW-aligns the attempt's
own features against compose_reference_features's concatenated reference,
projects the reference's known per-frame gloss labels onto the attempt
through the warp path to get per-gloss frame ranges, re-trims each range to
its own live_capture_span (a warp-path boundary is a DTW artifact, not a
clean rest->sign->rest clip -- see align_and_grade's inline comment for the
measured train/serve mismatch this closes), then calls
EmbeddingGrader.grade_against_poses per segment exactly as the existing
isolated-sign live loop already does -- no grading logic is duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import hand_motion_energy, live_capture_span
from ..production.gloss_rules import GlossSequence
from ..production.retrieval import compose_reference_features
from .dtw_grader import dtw_align
from .embedding_dataset import LIVE_PREROLL, LIVE_SETTLE_FRAMES
from .embedding_grader import EmbeddingGrader, GradeResult


@dataclass
class AlignedGrade:
    """One target gloss's slice of a forced-aligned continuous attempt."""

    target_sign: str
    frame_range: "tuple[int, int]"  # [start, stop) into the attempt's poses
    result: GradeResult


def _segment_ranges(path: "list[tuple[int, int]]", frame_gloss_index: np.ndarray,
                     n_glosses: int) -> "list[tuple[int, int]]":
    """From a dtw_align warp path ((attempt_frame, reference_frame) pairs)
    and the reference's per-frame gloss labels, derive each gloss's
    [start, stop) range IN THE ATTEMPT: the min/max attempt frame aligned to
    ANY reference frame belonging to that gloss. Robust to the many-to-one/
    one-to-many stretches a warp path can have on either side -- a COMPLETE
    path is a monotonic staircase from (0,0) to (n-1,m-1), so every
    reference frame index appears at least once, guaranteeing every gloss at
    least one attempt frame."""
    lo: list = [None] * n_glosses
    hi: list = [None] * n_glosses
    for i, j in path:
        g = int(frame_gloss_index[j])
        if lo[g] is None or i < lo[g]:
            lo[g] = i
        if hi[g] is None or i > hi[g]:
            hi[g] = i
    missing = [g for g in range(n_glosses) if lo[g] is None]
    if missing:
        raise RuntimeError(
            f"forced alignment produced no attempt frames for gloss index(es) "
            f"{missing} -- the warp path should be a complete staircase that "
            f"touches every reference frame; this should be unreachable.")
    return [(lo[g], hi[g] + 1) for g in range(n_glosses)]


def align_and_grade(grader: EmbeddingGrader, attempt_poses, gloss_sequence: GlossSequence,
                     *, band: "int | None" = None) -> "tuple[float, list[AlignedGrade]]":
    """Forced-align a live continuous attempt against `gloss_sequence`'s
    composed reference, then grade each detected segment against its own
    target sign.

    `grader` supplies the pipeline/standardizer BOTH sides are featurized
    with (so the alignment and the eventual grading agree on feature space)
    and the actual per-sign grading (`grade_against_poses`, unchanged --
    each segment is graded exactly as an isolated attempt already is).
    `attempt_poses` must be sliceable (a list or array of Pose objects, the
    same live-capture output `diagnose_demo.py`'s single-sign path already
    grades). Returns `(alignment_distance, [AlignedGrade, ...])` in
    `gloss_sequence`'s gloss order, mirroring `dtw_align`'s own
    `(distance, path)` return shape.
    """
    attempt_poses = list(attempt_poses)
    composed = compose_reference_features(
        gloss_sequence, grader.extractor, grader.pipeline, grader.standardizer)

    attempt_clip = grader.pipeline.assemble(attempt_poses)
    attempt_features = grader.standardizer.transform(attempt_clip.features)

    distance, path = dtw_align(attempt_features, composed.features, band=band)
    if not path:
        raise ValueError("forced alignment failed -- empty attempt or reference sequence")

    n_glosses = len(gloss_sequence.gloss_ids)
    ranges = _segment_ranges(path, composed.frame_gloss_index, n_glosses)

    graded = []
    for gloss_id, (start, stop) in zip(gloss_sequence.gloss_ids, ranges):
        raw_segment = attempt_poses[start:stop]
        # A DTW-derived segment boundary is a warp-path artifact, not a clean
        # rest->sign->rest clip -- a live continuous attempt naturally has brief
        # pauses between words (sentence mode's own SENTENCE_SETTLE_FRAMES is
        # built to ride through them, per diagnose_demo.py), and DTW has no
        # "no match" option, so that rest gets glued onto whichever segment is
        # nearest in feature space. The model was trained on tightly-trimmed
        # clips (embedding_dataset.LIVE_PREROLL/LIVE_SETTLE_FRAMES), so grading
        # the raw slice as-is re-creates the exact train/serve rest-padding
        # mismatch already fixed once for single-sign capture -- verified
        # empirically here too: 25 synthetic continuous attempts built from
        # raw (untrimmed, real, natural-rest) val clips flipped handshape
        # 18.2%, repeated_movement 6.7%, others 5-8%, vs. grading the same
        # clips in isolation; re-trimming each segment to its own
        # live_capture_span before grading (below) cut every one of those
        # roughly 2-3x (handshape 18.2%->5.5%, repeated_movement 6.7%->4.0%).
        energy = hand_motion_energy(grader.pipeline.normalizer, grader.pipeline.skeleton, raw_segment)
        t_start, t_stop = live_capture_span(
            energy, grader.pipeline.motion_threshold, LIVE_PREROLL, LIVE_SETTLE_FRAMES)
        segment = raw_segment[t_start:t_stop]
        result = grader.grade_against_poses(segment, gloss_id)
        graded.append(AlignedGrade(
            target_sign=gloss_id, frame_range=(start + t_start, start + t_stop), result=result))
    return distance, graded
