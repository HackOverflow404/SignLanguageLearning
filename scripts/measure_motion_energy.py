#!/usr/bin/env python3
"""Measure per-frame hand motion energy across the cached corpus, per extractor.

WHY: ASL Citizen clips are recorded rest -> sign -> rest. DTW normalizes total
path cost by path length, and a run of near-identical rest frames aligns cheaply
against EVERY candidate sign (rest looks like rest, regardless of which sign
follows it) -- so those frames contribute near-zero cost to every candidate and
compress the gap between the right answer and the wrong ones. This script
measures how big that effect actually is, on real cached data, BEFORE any
trim-to-motion-active-span default changes -- see aslcv.features.FeaturePipeline's
`trim_to_motion` toggle (off by default until this measurement says otherwise).

Uses `aslcv.features.hand_motion_energy()` directly -- the SAME function
FeaturePipeline's trim step calls internally, and the one Phase 6's live boundary
detector is meant to reuse. This script is a second consumer proving that "one
implementation" claim, not a parallel reimplementation.

    .venv/bin/python scripts/measure_motion_energy.py                    # all 4 extractors, full corpus
    .venv/bin/python scripts/measure_motion_energy.py --extractor dwpose
    .venv/bin/python scripts/measure_motion_energy.py --limit 200        # faster smoke run
"""
import argparse
import glob
from pathlib import Path

import numpy as np

from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.base import Pose
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.features import hand_motion_energy, motion_active_span
from aslcv.normalizer.shoulder import ShoulderNormalizer

REPO = Path(__file__).resolve().parents[1]
ALL_EXTRACTORS = ["mediapipe", "dwpose", "rtmw", "vitpose"]

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


def skeleton_for(name):
    return MEDIAPIPE_HOLISTIC if name == "mediapipe" else COCO_WHOLEBODY


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="all", choices=ALL_EXTRACTORS + ["all"])
    ap.add_argument("--limit", type=int, default=None, help="cap clips per extractor (smoke test)")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="motion-energy threshold defining 'low motion' (default matches "
                         "FeaturePipeline's trim_to_motion default)")
    args = ap.parse_args()
    extractors = ALL_EXTRACTORS if args.extractor == "all" else [args.extractor]

    print(f"threshold={args.threshold} (normalized local-hand-frame units/frame)\n")
    print(f"{'extractor':<12}{'n clips':>9}{'mean head%':>12}{'mean tail%':>12}"
          f"{'median head%':>14}{'median tail%':>14}{'frac clips 20-40% either end':>30}")

    for extractor in extractors:
        norm = ShoulderNormalizer(local_hand=True)
        skeleton = skeleton_for(extractor)
        cache_dir = REPO / "data" / "cache" / extractor
        files = sorted(glob.glob(str(cache_dir / "*.npz")))
        if args.limit:
            files = files[:args.limit]
        if not files:
            print(f"{extractor:<12} SKIPPED: no cache at {cache_dir}")
            continue

        head_fracs, tail_fracs = [], []
        for f in tqdm(files, desc=extractor, unit="clip"):
            with np.load(f) as d:
                kp, sc = d["keypoints"], d["scores"]
            n = kp.shape[0]
            if n < 3:
                continue
            poses = [Pose(kp[t], sc[t]) for t in range(n)]
            energy = hand_motion_energy(norm, skeleton, poses)
            # unpadded span: the raw rest/active boundary, not the padded trim
            # window a real FeaturePipeline(trim_to_motion=True) would use
            start, stop = motion_active_span(energy, args.threshold, pad_frames=0)
            head_fracs.append(start / n)
            tail_fracs.append((n - stop) / n)

        head_fracs = np.array(head_fracs)
        tail_fracs = np.array(tail_fracs)
        either_20_40 = ((head_fracs >= 0.20) & (head_fracs <= 0.40)) | \
                       ((tail_fracs >= 0.20) & (tail_fracs <= 0.40))
        print(f"{extractor:<12}{len(files):>9}{head_fracs.mean() * 100:>11.1f}%"
              f"{tail_fracs.mean() * 100:>11.1f}%{np.median(head_fracs) * 100:>13.1f}%"
              f"{np.median(tail_fracs) * 100:>13.1f}%{either_20_40.mean() * 100:>29.1f}%")

    print("\nhead% / tail% = fraction of the clip's OWN frame count trimmed as low-motion")
    print("from that end (unpadded). 'frac clips 20-40% either end' = fraction of clips")
    print("where head OR tail rest-frame fraction falls in the 20-40% range flagged as")
    print("'a real effect' -- not a mean, since means can hide a bimodal split (some")
    print("clips mostly rest, others mostly active).")


if __name__ == "__main__":
    main()
