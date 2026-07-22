#!/usr/bin/env python3
"""Evaluate the DTW nearest-sign grader on the 20-sign Phase-2 val split.

The milestone as a NUMBER, before any webcam: build the reference bank from the
slice's TRAIN clips (per-sign), grade every VAL clip (held-out signers), and report
top-1 / top-5 accuracy, a small confusion summary, and -- specifically -- whether
`mother` and `father` (a single-parameter minimal pair) are told apart.

If val accuracy is poor, the bug is upstream of any camera.

    .venv/bin/python scripts/eval_slice.py --extractor dwpose
    .venv/bin/python scripts/eval_slice.py --extractor dwpose --limit 10   # smoke
"""
import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.features import FeaturePipeline
from aslcv.grading.dtw_grader import DTWGrader
from aslcv.normalizer.shoulder import ShoulderNormalizer

REPO = Path(__file__).resolve().parents[1]
PARENTS = ("mother", "father")

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(it, **kw):
        return it


def skeleton_for(extractor):
    return MEDIAPIPE_HOLISTIC if extractor == "mediapipe" else COCO_WHOLEBODY


def load_slice_rows(signs):
    """Manifest rows for the slice signs, grouped by split."""
    rows = list(csv.DictReader(open(REPO / "data" / "manifest.csv")))
    wanted = set(signs)
    by_split = defaultdict(list)
    for r in rows:
        if r["id_gloss"] in wanted:
            by_split[r["split"]].append(r)
    return by_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extractor", default="dwpose",
                    choices=["dwpose", "rtmw", "vitpose", "mediapipe"])
    ap.add_argument("--agg", default="min", choices=["min", "mean"],
                    help="per-sign aggregate of DTW distances over its reference clips")
    ap.add_argument("--band", type=int, default=None,
                    help="Sakoe-Chiba radius (frames); default full DP")
    ap.add_argument("--limit", type=int, default=None, help="cap val clips (smoke test)")
    ap.add_argument("--face", action="store_true", help="keep face keypoints in features")
    args = ap.parse_args()

    cache_dir = REPO / "data" / "cache" / args.extractor
    if not cache_dir.is_dir():
        raise SystemExit(f"no cache at {cache_dir} -- run scripts/extract_landmarks.py --extractor {args.extractor}")

    signs = yaml.safe_load(open(REPO / "curriculum.yaml"))["phase2_slice"]
    by_split = load_slice_rows(signs)
    train_rows, val_rows = by_split["train"], by_split["val"]
    if args.limit:
        val_rows = val_rows[:args.limit]

    skeleton = skeleton_for(args.extractor)
    pipeline = FeaturePipeline(ShoulderNormalizer(local_hand=True), skeleton, face=args.face)

    print(f"extractor={args.extractor}  agg={args.agg}  band={args.band}  face={args.face}")
    print(f"building reference bank: {len(train_rows)} train clips over {len(signs)} signs ...")
    t0 = time.time()
    grader = DTWGrader.build(pipeline, cache_dir, signs, train_rows, agg=args.agg, band=args.band)
    print(f"  built in {time.time() - t0:.1f}s"
          + (f"  ({grader.missing} train clips skipped: cache missing)" if grader.missing else ""))

    # -- grade the val split ------------------------------------------------
    top1 = top5 = total = skipped = 0
    predicted = defaultdict(Counter)   # true sign -> Counter(top-1 predictions)
    per_sign_total = Counter()
    parent = {p: {"n": 0, "correct": 0, "closer_to_other": 0} for p in PARENTS}

    t0 = time.time()
    for r in tqdm(val_rows, desc="grading val", unit="clip"):
        npz = cache_dir / f"{r['video_id']}.npz"
        if not npz.exists():
            skipped += 1
            continue
        true = r["id_gloss"]
        ranked = grader.grade(grader.featurize_npz(npz))
        names = [s for s, _ in ranked]
        dist = dict(ranked)

        total += 1
        per_sign_total[true] += 1
        predicted[true][names[0]] += 1
        top1 += names[0] == true
        top5 += true in names[:5]

        if true in PARENTS:
            other = "father" if true == "mother" else "mother"
            parent[true]["n"] += 1
            parent[true]["correct"] += names[0] == true
            parent[true]["closer_to_other"] += dist[other] < dist[true]

    elapsed = time.time() - t0
    if total == 0:
        raise SystemExit("no val clips graded (cache missing?)")

    # -- report -------------------------------------------------------------
    print(f"\n=== {args.extractor}: DTW nearest-sign on the 20-sign val split ===")
    print(f"graded {total} clips in {elapsed:.1f}s"
          + (f"  ({skipped} skipped: cache missing)" if skipped else ""))
    print(f"  top-1 accuracy : {top1}/{total} = {top1 / total:.1%}")
    print(f"  top-5 accuracy : {top5}/{total} = {top5 / total:.1%}")
    print(f"  chance (1/{len(signs)}) : {1 / len(signs):.1%}")

    print("\nper-sign top-1 (worst first):")
    per_sign_acc = []
    for s in signs:
        n = per_sign_total[s]
        if not n:
            continue
        correct = predicted[s][s]
        worst = predicted[s].most_common(1)[0]
        confused = "" if worst[0] == s else f"  most confused -> {worst[0]} ({worst[1]})"
        per_sign_acc.append((correct / n, s, correct, n, confused))
    for acc, s, correct, n, confused in sorted(per_sign_acc):
        print(f"  {s:<10} {correct}/{n} = {acc:4.0%}{confused}")

    print("\nmother / father minimal-pair test (single-parameter difference):")
    for p in PARENTS:
        d = parent[p]
        other = "father" if p == "mother" else "mother"
        if d["n"] == 0:
            print(f"  {p}: no val clips")
            continue
        print(f"  {p:<7} {d['n']} clips: top-1 correct {d['correct']}/{d['n']}; "
              f"graded closer to {other} than to {p}: {d['closer_to_other']}/{d['n']}")
    mn, fn = parent["mother"]["n"], parent["father"]["n"]
    separated = (
        mn and fn
        and parent["mother"]["closer_to_other"] <= mn / 2
        and parent["father"]["closer_to_other"] <= fn / 2
    )
    print(f"  VERDICT: {'SEPARATED' if separated else 'NOT clearly separated'} "
          f"(each parent grades closer to itself than to the other in the majority of clips)")


if __name__ == "__main__":
    main()
