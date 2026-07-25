#!/usr/bin/env python3
"""Verify ViTPose actually emits COCO-WholeBody index order -- programmatically,
not by eyeballing a drawn frame.

vitpose.py's own docstring flags this as unverified: "confirm the 133-keypoint
index order matches COCO_WHOLEBODY the first time you draw a real frame -- ...
this is a benchmarking candidate to validate in Phase 3." vitpose already has a
full column in scripts/eval_minimal_pairs.py's results; if its topology were
permuted, those numbers would be meaningless, not merely noisy.

METHOD: dwpose and vitpose both claim COCO-WholeBody and both already have a full
cache over the same source videos (no re-extraction needed). For every clip, load
both cached (keypoints, scores), and for every keypoint INDEX present (score > 0)
in both, compute the pixel distance between the two backends' estimate of that
same index, normalized by that frame's shoulder width (from dwpose, since it's
already the validated backend -- see tests/test_anchors.py). Aggregate the mean
normalized distance overall and per region (Skeleton.region()).

INTERPRETATION: if the topology matches, index i means "the same point" in both
caches, so error should be small and roughly uniform across regions (some
elevation expected in whichever backend is less accurate at a given region -- see
the resolution confound below). If the topology is PERMUTED, index i in one cache
is really some OTHER point in the other -- e.g. a permuted hand would compare
"vitpose's real thumb tip" against "dwpose's real pinky tip" under the same index,
producing large, region-concentrated error (a full hand-width or more, not a
few-pixel jitter), while unaffected regions stay small. That structural signature
-- large error walled off to specific regions vs. small error everywhere -- is
what distinguishes "permuted" from "just less accurate."

RESOLUTION CONFOUND (read before trusting any vitpose-vs-others number, not just
this script's output): vitpose runs its pose model at POSE_INPUT_SIZE (192, 256);
dwpose and rtmw both run at (288, 384) -- 2.25x the pixel area. ANY accuracy gap
seen between vitpose and the other two rtmlib backends (here or in
eval_minimal_pairs.py / eval_slice.py) is confounded by input resolution, not a
clean architecture comparison. easy_ViTPose (the checkpoint vitpose.py uses) does
not ship a matching 384x288 wholebody export as of this writing -- only
192x256 (this "-l-" variant) and smaller ("-b-"/"-s-", same 192x256 family). If a
matching-resolution checkpoint appears later, swap POSE_INPUT_SIZE and POSE_URL
in vitpose.py and re-run the sweep; until then, state this caveat wherever a
vitpose number is reported.

    .venv/bin/python scripts/verify_vitpose_topology.py               # sample of clips
    .venv/bin/python scripts/verify_vitpose_topology.py --limit 200   # more clips, slower
"""
import argparse
import glob
from pathlib import Path

import numpy as np

from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY

REPO = Path(__file__).resolve().parents[1]
REGIONS = ("body_upper", "arms", "left_hand", "right_hand", "face", "legs_feet")

# A permutation error should look like a swapped LIMB/finger, i.e. on the order of
# the region's own physical size (a hand-width, a face-width). Flag anything whose
# mean normalized error clears this as suspicious; it is not a hard pass/fail line.
# NOTE: legs_feet is EXPECTED to clear this even with a correct topology -- ASL
# Citizen's seated, upper-body framing means legs are almost never actually in
# frame, so both backends confidently hallucinate independent, uncorrelated leg
# positions (see CLAUDE.md known issue #2). The SWAP_PAIRS check below is what
# actually distinguishes "permuted" from "just hallucinated" for this region.
SUSPICION_THRESHOLD = 0.5  # in shoulder-widths

# Named left/right pairs to swap-test: if the topology is permuted (e.g. a
# left/right handedness bug), DWPOSE's left point should land closer to VITPOSE's
# RIGHT point than to vitpose's own left point. If the topology is correct, same-
# index is the better match regardless of how noisy either backend is -- this is
# what separates "scrambled indices" from "just imprecise," which the raw
# per-region distance table above cannot do on its own.
SWAP_PAIRS = (
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_hand_root", "right_hand_root"),
)


def shoulder_width(kp, sc):
    """(T,) shoulder width per frame from dwpose keypoints; nan where undetected."""
    ls, rs = COCO_WHOLEBODY.anchor("left_shoulder"), COCO_WHOLEBODY.anchor("right_shoulder")
    w = np.linalg.norm(kp[:, ls] - kp[:, rs], axis=-1)
    present = (sc[:, ls] > 0) & (sc[:, rs] > 0)
    w[~present] = np.nan
    return w


def compare_clip(dw_path, vit_path):
    """Per-keypoint-index normalized distance between dwpose and vitpose on the
    SAME clip. Returns (region -> list of per-frame-per-point normalized dists)."""
    with np.load(dw_path) as d:
        dw_kp, dw_sc = d["keypoints"], d["scores"]
    with np.load(vit_path) as d:
        vp_kp, vp_sc = d["keypoints"], d["scores"]
    if dw_kp.shape != vp_kp.shape:
        return None  # frame-count/topology-size mismatch; skip rather than misalign

    scale = shoulder_width(dw_kp, dw_sc)  # (T,)
    both_present = (dw_sc > 0) & (vp_sc > 0)  # (T, K)
    dist = np.linalg.norm(dw_kp - vp_kp, axis=-1)  # (T, K), raw pixels

    out = {}
    for region in REGIONS:
        idx = list(COCO_WHOLEBODY.region(region))
        mask = both_present[:, idx] & np.isfinite(scale)[:, None]
        d_norm = dist[:, idx][mask] / scale[:, None].repeat(len(idx), axis=1)[mask]
        if d_norm.size:
            out[region] = d_norm
    return out


def swap_check(dw_dir, vp_dir, common):
    """For each named left/right pair, (same-index dist, cross L-vs-R dist) mean
    in raw pixels, over frames where both backends detect BOTH sides. same < cross
    means the topology's left/right assignment is correct for that pair."""
    rows = []
    for left_name, right_name in SWAP_PAIRS:
        li, ri = COCO_WHOLEBODY.names.index(left_name), COCO_WHOLEBODY.names.index(right_name)
        same, cross = [], []
        for vid in common:
            with np.load(dw_dir / f"{vid}.npz") as d:
                dw_kp, dw_sc = d["keypoints"], d["scores"]
            with np.load(vp_dir / f"{vid}.npz") as d:
                vp_kp, vp_sc = d["keypoints"], d["scores"]
            if dw_kp.shape != vp_kp.shape:
                continue
            mask = (dw_sc[:, li] > 0) & (vp_sc[:, li] > 0) & (dw_sc[:, ri] > 0) & (vp_sc[:, ri] > 0)
            if not mask.any():
                continue
            same.append(np.linalg.norm(dw_kp[mask, li] - vp_kp[mask, li], axis=-1))
            cross.append(np.linalg.norm(dw_kp[mask, li] - vp_kp[mask, ri], axis=-1))
        if not same:
            continue
        same_mean = np.concatenate(same).mean()
        cross_mean = np.concatenate(cross).mean()
        rows.append((left_name, right_name, same_mean, cross_mean))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=60, help="max clips to compare (both must be cached)")
    args = ap.parse_args()

    dw_dir = REPO / "data" / "cache" / "dwpose"
    vp_dir = REPO / "data" / "cache" / "vitpose"
    if not dw_dir.is_dir() or not vp_dir.is_dir():
        raise SystemExit("need both data/cache/dwpose and data/cache/vitpose populated")

    dw_ids = {Path(p).stem for p in glob.glob(str(dw_dir / "*.npz"))}
    vp_ids = {Path(p).stem for p in glob.glob(str(vp_dir / "*.npz"))}
    common = sorted(dw_ids & vp_ids)[:args.limit]
    if not common:
        raise SystemExit("no clips cached under both dwpose and vitpose")

    print(f"comparing {len(common)} clips cached under BOTH dwpose and vitpose "
          f"(same source video, index-for-index)\n")

    per_region: dict[str, list] = {r: [] for r in REGIONS}
    skipped = 0
    for vid in common:
        result = compare_clip(dw_dir / f"{vid}.npz", vp_dir / f"{vid}.npz")
        if result is None:
            skipped += 1
            continue
        for region, vals in result.items():
            per_region[region].append(vals)

    print(f"{'region':<14}{'n points':>12}{'mean':>10}{'median':>10}{'p90':>10}   verdict")
    any_suspicious = False
    for region in REGIONS:
        vals = per_region[region]
        if not vals:
            print(f"{region:<14}{'--':>12}   no comparable (score>0 both sides) points found")
            continue
        arr = np.concatenate(vals)
        mean, median, p90 = arr.mean(), np.median(arr), np.percentile(arr, 90)
        suspicious = mean > SUSPICION_THRESHOLD
        any_suspicious |= suspicious
        verdict = "SUSPICIOUS -- check for a permutation" if suspicious else "small, consistent with matching topology"
        print(f"{region:<14}{arr.size:>12}{mean:>10.3f}{median:>10.3f}{p90:>10.3f}   {verdict}")

    overall = np.concatenate([np.concatenate(v) for v in per_region.values() if v])
    print(f"\n{'overall':<14}{overall.size:>12}{overall.mean():>10.3f}{np.median(overall):>10.3f}"
          f"{np.percentile(overall, 90):>10.3f}")
    if skipped:
        print(f"\n({skipped} clips skipped: frame-count mismatch between the two caches)")

    # -- swap check: is any region's error actually a left/right permutation? -----
    print("\n" + "-" * 78)
    print("Left/right swap check (raw pixels; same-index vs. cross L<->R distance)")
    print("same < cross means that pair's left/right assignment is correct")
    print("-" * 78)
    print(f"{'pair':<32}{'same-index':>12}{'cross(L-R)':>12}   verdict")
    swap_rows = swap_check(dw_dir, vp_dir, common)
    any_swapped = False
    for left_name, right_name, same_mean, cross_mean in swap_rows:
        swapped = cross_mean < same_mean
        any_swapped |= swapped
        verdict = "SWAPPED -- permutation" if swapped else "correct L/R assignment"
        print(f"{left_name + '/' + right_name:<32}{same_mean:>12.1f}{cross_mean:>12.1f}   {verdict}")

    print("\n" + "=" * 78)
    if any_swapped:
        print("VERDICT: at least one left/right pair matches BETTER when swapped than in")
        print("its own index -- a genuine topology permutation. Do NOT trust vitpose numbers.")
    elif any_suspicious:
        print("VERDICT: a region's raw error is large (see table above), but the swap check")
        print("shows same-index is still the closer match everywhere -- NOT a permutation.")
        print("legs_feet in particular is expected to show large, uncorrelated error even")
        print("with a correct topology: ASL Citizen's seated framing means legs are almost")
        print("never actually visible, so both backends confidently hallucinate independent")
        print("(and therefore mutually inconsistent) leg positions -- see CLAUDE.md known")
        print("issue #2. This is a data/framing limitation, not evidence vitpose's index")
        print("order is wrong.")
    else:
        print("VERDICT: error is small and roughly uniform across every region, and no")
        print("left/right pair swaps -- consistent with vitpose using the SAME")
        print("COCO-WholeBody index order as dwpose. This does NOT clear vitpose of the")
        print("resolution confound below -- only of a scrambled topology.")
    print("=" * 78)
    print("\nRESOLUTION CONFOUND: vitpose runs its pose model at 192x256; dwpose/rtmw run")
    print("at 288x384 (2.25x the pixel area). Any vitpose vs. dwpose/rtmw accuracy gap --")
    print("here or in eval_minimal_pairs.py / eval_slice.py -- is confounded by input")
    print("resolution, not a clean architecture comparison. No matching-resolution")
    print("easy_ViTPose wholebody checkpoint is known to exist as of this writing.")


if __name__ == "__main__":
    main()
