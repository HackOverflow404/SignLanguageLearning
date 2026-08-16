#!/usr/bin/env python3
"""Phase 7 step 4 -- the go/no-go validation checkpoint for forced alignment,
BEFORE any live-UI work (see project_workflow.md's Phase 7 section for the
full 6-step plan this gates).

Builds synthetic "continuous" sentences by concatenating REAL, TRIMMED,
HELD-OUT (val-split) clips end-to-end for a random sample of curriculum
signs -- the same trick production.retrieval.ComposedReference uses for
video, at benchmark scale, keeping the ground-truth per-gloss frame counts
this time since we did the concatenating ourselves. The reference side
(compose_reference_features, via align_and_grade) is built from TRAIN-split
clips exactly as a real session would use it, so this measures alignment
against genuinely different footage of the same signs -- not the trivial
identical-clip case tests/test_alignment.py's unit tests use for plumbing
checks only.

Reports two numbers per trial, aggregated:
  (a) BOUNDARY ERROR: |predicted segment length - true segment length| /
      true segment length, per gloss -- how well the warp path recovers the
      real cut points.
  (b) GRADING AGREEMENT: does grade_against_poses's per-parameter verdict on
      the ALIGNED segment agree with grading that exact same clip in
      isolation (the already-validated PHASE4_REPORT.md path)? This is
      deliberately NOT measured against the true phonology label -- the
      isolated grader itself is only ~81% accurate (PHASE4_REPORT.md), so
      comparing against ground truth would conflate alignment error with
      already-known, already-documented model error. Agreement with the
      isolated grade isolates what THIS step adds: did segmentation corrupt
      an otherwise-gradable segment.

HONEST CAVEAT, unchanged regardless of the result (see CLAUDE.md/
project_workflow.md's Phase 7 section): concatenated real clips have a hard
cut and no coarticulation -- this measures an EASIER problem than genuine
fluent continuous signing. A good number here is necessary, not sufficient,
before building live capture (step 5) and the sentence-mode UI (step 6).

    .venv/bin/python scripts/eval_forced_alignment.py
    .venv/bin/python scripts/eval_forced_alignment.py --n-trials 50 --min-signs 2 --max-signs 4
"""
import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from aslcv.grading.alignment import align_and_grade
from aslcv.grading.embedding_grader import EmbeddingGrader
from aslcv.production.retrieval import skeleton_for
from aslcv.research.synthetic_sentences import make_trial, rows_by_sign_and_split

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "models" / "embedding_grader"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


def run(n_trials, min_signs, max_signs, checkpoint_dir, extractor, seed, band):
    rng = random.Random(seed)
    grader = EmbeddingGrader.build(checkpoint_dir)
    skeleton = skeleton_for(extractor)
    by_sign_split = rows_by_sign_and_split(extractor)

    boundary_errors = []  # relative, one per (trial, gloss)
    boundary_errors_abs = []  # frames
    param_agree = defaultdict(lambda: [0, 0])   # parameter -> [agree, total] (correct-flag)
    label_agree = defaultdict(lambda: [0, 0])   # parameter -> [agree, total] (predicted-label)
    n_run = n_skipped = 0

    for _ in tqdm(range(n_trials), desc="forced-alignment trials", unit="trial"):
        k = rng.randint(min_signs, max_signs)
        trial = make_trial(by_sign_split, extractor, skeleton, k, rng)
        if trial is None:
            n_skipped += 1
            continue
        seq, attempt_poses, true_lengths, val_segments = trial

        try:
            _, graded = align_and_grade(grader, attempt_poses, seq, band=band)
        except Exception as exc:  # noqa: BLE001 -- a benchmark script degrades, doesn't crash on one bad trial
            print(f"  trial failed ({seq.english}): {exc}")
            n_skipped += 1
            continue
        n_run += 1

        for (start, stop), true_len, sign, segment in zip(
                (g.frame_range for g in graded), true_lengths, seq.gloss_ids, val_segments):
            pred_len = stop - start
            boundary_errors.append(abs(pred_len - true_len) / true_len)
            boundary_errors_abs.append(abs(pred_len - true_len))

        isolated_by_sign = {}
        for sign, segment in zip(seq.gloss_ids, val_segments):
            isolated_by_sign[sign] = grader.grade_against_poses(segment, sign)

        for g in graded:
            isolated = isolated_by_sign[g.target_sign]
            for p, aligned_v in g.result.parameters.items():
                isolated_v = isolated.parameters[p]
                if isolated_v.correct is None:
                    continue  # thin/insufficient-support target -- excluded from every other report too
                param_agree[p][1] += 1
                param_agree[p][0] += int(aligned_v.correct == isolated_v.correct)
                label_agree[p][1] += 1
                label_agree[p][0] += int(aligned_v.predicted == isolated_v.predicted)

    return {
        "n_run": n_run, "n_skipped": n_skipped,
        "boundary_errors": boundary_errors, "boundary_errors_abs": boundary_errors_abs,
        "param_agree": dict(param_agree), "label_agree": dict(label_agree),
    }


def format_report(results, args) -> str:
    be = results["boundary_errors"]
    bea = results["boundary_errors_abs"]
    lines = [
        f"# Phase 7 step 4 -- forced-alignment validation ({args.extractor}, seed={args.seed})",
        "",
        f"{results['n_run']} trials run, {results['n_skipped']} skipped (insufficient sign pool or a "
        f"failed alignment), {args.min_signs}-{args.max_signs} signs/sentence, "
        f"band={args.band}.",
        "",
        "## Boundary error (predicted vs. true segment length, per gloss)",
        "",
        f"- mean relative error: {statistics.mean(be):.1%}" if be else "- no samples",
        f"- median relative error: {statistics.median(be):.1%}" if be else "",
        f"- mean absolute error: {statistics.mean(bea):.1f} frames" if bea else "",
        (f"- worst relative error: {max(be):.1%} ({sum(1 for e in be if e > 0.5)}/{len(be)} "
         f"segments exceed 50% relative error)") if be else "",
        "",
        "## Grading agreement (aligned segment vs. isolated grade of the same clip)",
        "",
        "Excludes thin/insufficient-support target parameters (MIN_SUPPORT gate) --",
        "same convention every other report in this project uses.",
        "",
        "| parameter | correct-flag agreement | predicted-label agreement | n |",
        "|---|---|---|---|",
    ]
    for p in sorted(results["param_agree"]):
        pa, pn = results["param_agree"][p]
        la, ln = results["label_agree"][p]
        pa_pct = f"{pa/pn:.1%}" if pn else "n/a"
        la_pct = f"{la/ln:.1%}" if ln else "n/a"
        lines.append(f"| {p} | {pa_pct} | {la_pct} | {pn} |")
    lines += [
        "",
        "## Honest caveat",
        "",
        "Concatenated real clips have a hard cut and no coarticulation -- this measures",
        "an EASIER problem than genuine fluent continuous signing (see CLAUDE.md /",
        "project_workflow.md's Phase 7 section). A good number here is necessary, not",
        "sufficient, before building live capture (step 5) and the sentence-mode UI",
        "(step 6).",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--min-signs", type=int, default=2)
    ap.add_argument("--max-signs", type=int, default=4)
    ap.add_argument("--extractor", default="mediapipe")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band", type=int, default=None)
    ap.add_argument("--report", type=Path, default=REPO / "PHASE7_ALIGNMENT_REPORT.md")
    args = ap.parse_args()

    results = run(args.n_trials, args.min_signs, args.max_signs, args.checkpoint,
                   args.extractor, args.seed, args.band)
    report = format_report(results, args)
    print("\n" + report)
    args.report.write_text(report)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
