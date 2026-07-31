#!/usr/bin/env python3
"""Evaluate the Phase 4 learned grader against the DTW baseline on the 60-sign val
split, and demonstrate per-parameter head disagreement on a real minimal pair.

This exercises EmbeddingGrader's PUBLIC interface only (grade / grade_against) --
not the training script's internals -- so the numbers reflect what the deliverable
actually does from the outside, the same way eval_slice.py measures DTWGrader.

The DTW baseline is RE-MEASURED here on the 60-sign val split (not reused from the
20-sign Phase 2 milestone numbers, which aren't comparable to a 60-way bank).

    .venv/bin/python scripts/eval_embedding_grader.py
    .venv/bin/python scripts/eval_embedding_grader.py --which final   # compare vs best
"""
import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.grading.dtw_grader import DTWGrader
from aslcv.grading.embedding_grader import EmbeddingGrader
from aslcv.grading.phonology_labels import CATEGORICAL_PARAMETERS, MIN_SUPPORT, PhonologyLabels
from aslcv.pipeline_config import add_pipeline_args, build_pipeline

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "models" / "embedding_grader"
PARENTS = ("mother", "father")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


def skeleton_for(extractor):
    return MEDIAPIPE_HOLISTIC if extractor == "mediapipe" else COCO_WHOLEBODY


def load_rows(extractor, signs):
    rows = list(csv.DictReader(open(REPO / "data" / "manifest.csv")))
    wanted = set(signs)
    by_split = defaultdict(list)
    for r in rows:
        if r["id_gloss"] in wanted:
            by_split[r["split"]].append(r)
    return by_split


def eval_dtw_baseline(extractor, signs, by_split, agg, band):
    cache_dir = REPO / "data" / "cache" / extractor
    ap = argparse.ArgumentParser()
    add_pipeline_args(ap)
    args = ap.parse_args([])
    skeleton = skeleton_for(extractor)
    pipeline = build_pipeline(args, skeleton, extractor_name=extractor, quiet=True)

    print(f"building DTW reference bank: {len(by_split['train'])} train clips over {len(signs)} signs ...")
    t0 = time.time()
    grader = DTWGrader.build(pipeline, cache_dir, signs, by_split["train"], agg=agg, band=band)
    print(f"  built in {time.time() - t0:.1f}s")

    top1 = top5 = total = 0
    for r in tqdm(by_split["val"], desc="DTW grading val", unit="clip"):
        npz = cache_dir / f"{r['video_id']}.npz"
        if not npz.exists():
            continue
        ranked = grader.grade(grader.featurize_npz(npz))
        names = [s for s, _ in ranked]
        total += 1
        top1 += names[0] == r["id_gloss"]
        top5 += r["id_gloss"] in names[:5]
    return {"top1": top1 / total, "top5": top5 / total, "n": total}


def eval_learned(checkpoint_dir, which, extractor, signs, by_split, phon):
    cache_dir = REPO / "data" / "cache" / extractor
    print(f"loading EmbeddingGrader checkpoint ({which}) from {checkpoint_dir} ...")
    grader = EmbeddingGrader.build(checkpoint_dir, signs=signs, which=which)

    top1 = top5 = total = 0
    # per-parameter tally, split by whether the TARGET's true label is well-supported
    tally = {p: {"well": Counter(), "thin": Counter()} for p in CATEGORICAL_PARAMETERS + ("repeated_movement",)}

    for r in tqdm(by_split["val"], desc="learned grading val", unit="clip"):
        npz = cache_dir / f"{r['video_id']}.npz"
        if not npz.exists():
            continue
        true_sign = r["id_gloss"]
        ranked = grader.grade(npz)
        names = [s for s, _ in ranked]
        total += 1
        top1 += names[0] == true_sign
        top5 += true_sign in names[:5]

        result = grader.grade_against(npz, true_sign)
        for p, verdict in result.parameters.items():
            bucket = "well" if verdict.correct is not None else "thin"
            key = "correct" if verdict.correct else ("incorrect" if verdict.correct is False else "n/a")
            tally[p][bucket][key] += 1

    metrics = {"top1": top1 / total, "top5": top5 / total, "n": total}
    return metrics, tally


def format_tally(tally) -> str:
    lines = []
    for p, buckets in tally.items():
        well = buckets["well"]
        n_well = well["correct"] + well["incorrect"]
        acc = f"{well['correct']}/{n_well} = {well['correct']/n_well:.1%}" if n_well else "no well-supported clips"
        n_thin = sum(buckets["thin"].values())
        lines.append(f"  {p:<20} well-supported: {acc}   thin/insufficient-data (excluded): {n_thin} clips")
    return "\n".join(lines)


def minimal_pair_demo(grader) -> str:
    """Grade a real father clip against mother (and vice versa) -- father/mother is
    curriculum.yaml's built-in minimal pair, differing ONLY in minor_location
    (Forehead vs Chin), same handshape and movement. Reports whatever the model
    ACTUALLY does -- not asserted as correct, just shown."""
    rows = list(csv.DictReader(open(REPO / "data" / "manifest.csv")))
    lines = []
    for true_sign, other in (("father", "mother"), ("mother", "father")):
        val_rows = [r for r in rows if r["id_gloss"] == true_sign and r["split"] == "val"]
        if not val_rows:
            lines.append(f"  no val clip for {true_sign!r}, skipped")
            continue
        cache_dir = REPO / "data" / "cache" / grader.extractor
        npz = cache_dir / f"{val_rows[0]['video_id']}.npz"
        result = grader.grade_against(npz, other)
        lines.append(f"  attempt={true_sign} target={other}  fidelity(embedding dist)={result.fidelity:.3f}")
        for p, v in result.parameters.items():
            tag = "MATCH" if v.correct else ("DISAGREE" if v.correct is False else "insufficient data")
            lines.append(f"    {p:<20} predicted={v.predicted!r:<12} target={v.target!r:<12} [{tag}]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="mediapipe")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--agg", default="min", choices=["min", "mean"])
    ap.add_argument("--band", type=int, default=None)
    ap.add_argument("--report", type=Path, default=REPO / "PHASE4_REPORT.md")
    args = ap.parse_args()

    phon = PhonologyLabels()
    signs = sorted(phon.by_gloss)
    by_split = load_rows(args.extractor, signs)
    print(f"60-sign val split: {len(by_split['val'])} clips\n")

    dtw_metrics = eval_dtw_baseline(args.extractor, signs, by_split, args.agg, args.band)
    print(f"\nDTW baseline (60-sign val): top1={dtw_metrics['top1']:.1%} top5={dtw_metrics['top5']:.1%} (n={dtw_metrics['n']})\n")

    learned_metrics, tally = eval_learned(args.checkpoint, args.which, args.extractor, signs, by_split, phon)
    print(f"\nlearned grader ({args.which}, 60-sign val): top1={learned_metrics['top1']:.1%} top5={learned_metrics['top5']:.1%} (n={learned_metrics['n']})\n")

    print("per-parameter diagnosis accuracy on val (well-supported target classes only; N>=%d signs):" % MIN_SUPPORT)
    print(format_tally(tally))

    print("\nmother/father minimal-pair disagreement demo (differ ONLY in minor_location):")
    grader = EmbeddingGrader.build(args.checkpoint, signs=signs, which=args.which)
    demo_text = minimal_pair_demo(grader)
    print(demo_text)

    beat_dtw_top1 = learned_metrics["top1"] > dtw_metrics["top1"]
    beat_dtw_top5 = learned_metrics["top5"] > dtw_metrics["top5"]

    report = f"""# Phase 4 learned grader -- evaluation report

Checkpoint: `{args.checkpoint}` (`{args.which}`), extractor `{args.extractor}`, 60-sign val split ({dtw_metrics['n']} clips).

## grade() / grade_against(): learned vs. DTW baseline

| | top-1 | top-5 |
|---|---|---|
| DTW baseline (re-measured, 60-sign val) | {dtw_metrics['top1']:.1%} | {dtw_metrics['top5']:.1%} |
| Learned embedding grader | {learned_metrics['top1']:.1%} | {learned_metrics['top5']:.1%} |

**{'Beats' if beat_dtw_top1 and beat_dtw_top5 else 'Does NOT cleanly beat'} the DTW baseline** on this split.

## Per-parameter diagnosis accuracy (val, well-supported target classes only, N>={MIN_SUPPORT} signs)

```
{format_tally(tally)}
```

Thin/singleton-class clips are excluded from the accuracy figures above by design
(see `phonology_labels.py`'s `MIN_SUPPORT` gate) -- their verdicts are reported to a
user as "insufficient data," not folded into an average that would overstate
confidence.

## Head-independence demonstration (mother/father, a real minimal pair)

mother/father differ ONLY in `minor_location` (Forehead vs Chin) -- same handshape,
same movement. Grading a real attempt of one against the other as target:

```
{demo_text}
```

## Overfitting check

See `models/embedding_grader/history.json` for the full per-epoch train/val curve;
the training script tracks and saves the best-val-top1 checkpoint separately from
the final epoch specifically because train top-1 saturates well before val does on
this dataset size (~15 clips/sign) -- the gap itself is the signal, not hidden.
"""
    args.report.write_text(report)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
