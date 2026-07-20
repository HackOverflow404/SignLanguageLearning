#!/usr/bin/env python3
"""Integrity-check the extracted pose cache (run after any crash/interrupted run).

    .venv/bin/python scripts/verify_cache.py                 # report only
    .venv/bin/python scripts/verify_cache.py --delete-bad    # delete corrupt/invalid files

For every backend under data/cache/{extractor}/, checks each {video_id}.npz for:
  - loads at all (a truncated/half-flushed file fails CRC on read -> caught here),
  - required arrays present with consistent shapes (keypoints (T,K,2), scores (T,K),
    matching T; MediaPipe also blendshapes (T,52)),
  - correct K for the backend's topology (553 mediapipe / 133 the rtmlib models),
  - no NaN/Inf, T >= 1,
  - metadata extractor tag matches the folder.
Also cross-checks against data/manifest.csv: every clip present, no orphans.

Exits 0 if everything is clean, 1 if any issues are found. With --delete-bad, the
bad files are removed so a subsequent `extract_landmarks.py --extractor all` run
re-extracts exactly those (resume skips the good ones).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "cache"
MANIFEST = REPO / "data" / "manifest.csv"

BACKENDS = ["mediapipe", "dwpose", "rtmw", "vitpose"]
K_EXPECT = {"mediapipe": 553, "dwpose": 133, "rtmw": 133, "vitpose": 133}
HAS_BLENDSHAPES = {"mediapipe"}


def manifest_ids():
    with open(MANIFEST, newline="") as f:
        return [r["video_id"] for r in csv.DictReader(f)]


def check_npz(path, backend):
    """Return a list of problem strings (empty = healthy)."""
    problems = []
    try:
        with np.load(path) as z:
            names = set(z.files)
            if "keypoints" not in names or "scores" not in names:
                return [f"missing arrays (has {sorted(names)})"]
            kp, sc = z["keypoints"], z["scores"]          # forces read -> CRC check
            K = K_EXPECT[backend]
            if kp.ndim != 3 or kp.shape[2] != 2:
                problems.append(f"keypoints shape {kp.shape}")
            if sc.ndim != 2:
                problems.append(f"scores shape {sc.shape}")
            if kp.shape[0] == 0:
                problems.append("0 frames")
            if kp.shape[0] != sc.shape[0]:
                problems.append(f"frame mismatch kp={kp.shape[0]} sc={sc.shape[0]}")
            if kp.shape[1] != K or sc.shape[-1] != K:
                problems.append(f"K kp={kp.shape[1]} sc={sc.shape[-1]} != {K}")
            if not np.isfinite(kp).all():
                problems.append("NaN/Inf in keypoints")
            if not np.isfinite(sc).all():
                problems.append("NaN/Inf in scores")
            if backend in HAS_BLENDSHAPES:
                if "blendshapes" not in names:
                    problems.append("missing blendshapes")
                else:
                    bs = z["blendshapes"]
                    if bs.shape != (kp.shape[0], 52):
                        problems.append(f"blendshapes shape {bs.shape}")
                    elif not np.isfinite(bs).all():
                        problems.append("NaN/Inf in blendshapes")
            if "extractor" in names and str(z["extractor"]) != backend:
                problems.append(f"extractor tag '{z['extractor']}' != {backend}")
    except Exception as e:
        return [f"LOAD FAILED: {type(e).__name__}: {e}"]
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete-bad", action="store_true",
                    help="delete corrupt/invalid files so a resume re-extracts them")
    args = ap.parse_args()

    ids = manifest_ids()
    idset = set(ids)
    print(f"manifest: {len(ids)} clips\n")

    all_clean = True
    total_bad = 0
    for b in BACKENDS:
        d = CACHE / b
        files = sorted(d.glob("*.npz")) if d.is_dir() else []
        present = {p.stem for p in files}
        missing = idset - present
        orphan = present - idset

        bad = []
        for p in files:
            probs = check_npz(p, b)
            if probs:
                bad.append((p, "; ".join(probs)))

        healthy = len(files) - len(bad)
        status = "OK" if (len(files) == len(ids) and not missing and not orphan and not bad) else "ISSUES"
        if status != "OK":
            all_clean = False
        print(f"[{b}] {status}: {healthy}/{len(ids)} healthy | "
              f"corrupt={len(bad)} missing={len(missing)} orphan={len(orphan)}")

        for p, reason in bad[:25]:
            print(f"    CORRUPT {p.name} -> {reason}")
        if len(bad) > 25:
            print(f"    ... and {len(bad) - 25} more corrupt")
        for m in sorted(missing)[:15]:
            print(f"    MISSING {m}")
        if len(missing) > 15:
            print(f"    ... and {len(missing) - 15} more missing")
        for o in sorted(orphan)[:15]:
            print(f"    ORPHAN  {o} (not in manifest)")

        if args.delete_bad and bad:
            for p, _ in bad:
                p.unlink()
            print(f"    deleted {len(bad)} bad file(s)")
        total_bad += len(bad)

    print()
    if all_clean:
        print("ALL CLEAN — every backend has all clips, no corruption.")
    else:
        print("ISSUES FOUND." + (
            f" Re-run with --delete-bad, then `extract_landmarks.py --extractor all` to repair."
            if total_bad and not args.delete_bad else
            " Missing clips: run `extract_landmarks.py --extractor all` to fill them."))
    sys.exit(0 if all_clean else 1)


if __name__ == "__main__":
    main()
