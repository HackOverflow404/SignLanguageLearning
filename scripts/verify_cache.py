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

PROVENANCE CHECK: also reads each clip's recorded extraction config (running_mode,
process_every_n_frames, checkpoint -- see extract_landmarks.py) and flags any clip
whose recorded config does NOT match what the current pipeline would produce for
that backend -- e.g. a checkpoint that has since changed, or (the bug this whole
mechanism exists to catch) a VIDEO-mode clip recorded with process_every_n_frames
!= 1. This is reported as STALE, separate from CORRUPT: stale data isn't damaged,
it just needs re-extracting under current settings to be comparable with the rest
of the cache. Clips cached before provenance was recorded have none of these
fields at all -- reported as "provenance unknown", never as an error or a reason
to fail the run; commit_hash is recorded but never compared (most commits don't
touch extraction, so a different hash alone doesn't mean a cache is wrong).

Exits 0 if everything is clean, 1 if any issues are found (corrupt/missing/orphan/
stale -- "provenance unknown" alone does not fail the run). With --delete-bad, the
bad (corrupt) files are removed so a subsequent `extract_landmarks.py --extractor
all` run re-extracts exactly those (resume skips the good ones); stale files are
NOT auto-deleted (their data is valid, just built under different settings -- that's
a judgment call for a human, not something to silently discard).
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

# What the CURRENT extraction pipeline would produce for each backend (mirrors
# extract_landmarks.py's BUILDERS) -- used to compare against a cached clip's
# recorded provenance. running_mode/process_every_n_frames are cheap constants;
# expected_checkpoint() below reads a class attribute without constructing the
# (heavy, model-downloading) extractor itself.
EXPECTED_RUNNING_MODE = {"mediapipe": "video", "dwpose": "image", "rtmw": "image", "vitpose": "image"}
EXPECTED_PROCESS_EVERY_N_FRAMES = 1
PROVENANCE_FIELDS = ("running_mode", "process_every_n_frames", "checkpoint", "commit_hash")


def expected_checkpoint(name):
    """Model checkpoint identifier the CURRENT pipeline would use for `name`,
    without constructing (and downloading/loading) the actual extractor -- just
    reads the class-level POSE_URL / module-level POSE_MODEL_PATH constant, the
    same identifiers extract_landmarks.py's checkpoint_id() records."""
    if name == "mediapipe":
        from aslcv.extractor.mediapipe import POSE_MODEL_PATH
        return POSE_MODEL_PATH
    from aslcv.extractor import dwpose, rtmw, vitpose
    cls = {"dwpose": dwpose.DWPoseExtractor, "rtmw": rtmw.RTMWExtractor,
           "vitpose": vitpose.ViTPoseExtractor}[name]
    return cls.POSE_URL


def manifest_ids():
    with open(MANIFEST, newline="") as f:
        return [r["video_id"] for r in csv.DictReader(f)]


def check_provenance(z, backend):
    """(status, mismatches) for one already-open npz's recorded extraction config.

    status is "unknown" if the clip predates provenance recording (none of the
    fields are present -- see module docstring: never treated as an error) or
    "checked" otherwise. `mismatches` is only meaningful when status=="checked";
    an empty list means the recorded config matches what the current pipeline
    would produce for `backend`.
    """
    names = set(z.files)
    if not any(f in names for f in PROVENANCE_FIELDS):
        return "unknown", []
    mismatches = []
    if "running_mode" in names:
        got, want = str(z["running_mode"]), EXPECTED_RUNNING_MODE[backend]
        if got != want:
            mismatches.append(f"running_mode={got!r} (current pipeline uses {want!r})")
    else:
        mismatches.append("running_mode field missing")
    if "process_every_n_frames" in names:
        got = int(z["process_every_n_frames"])
        if got != EXPECTED_PROCESS_EVERY_N_FRAMES:
            mismatches.append(
                f"process_every_n_frames={got} (current pipeline uses "
                f"{EXPECTED_PROCESS_EVERY_N_FRAMES}) -- n>1 in VIDEO mode duplicates "
                f"frames and zeroes velocity deltas between them")
    else:
        mismatches.append("process_every_n_frames field missing")
    if "checkpoint" in names:
        got, want = str(z["checkpoint"]), expected_checkpoint(backend)
        if got != want:
            mismatches.append(f"checkpoint changed (cached={got!r}, current={want!r})")
    else:
        mismatches.append("checkpoint field missing")
    return "checked", mismatches


def check_npz(path, backend):
    """Return (problems, provenance_status, provenance_mismatches).

    problems: structural/data-integrity issues (empty = healthy data).
    provenance_status/provenance_mismatches: see check_provenance -- kept
    separate from `problems` because a provenance mismatch means "built under
    different settings," not "the data is damaged."
    """
    problems = []
    try:
        with np.load(path) as z:
            names = set(z.files)
            if "keypoints" not in names or "scores" not in names:
                return [f"missing arrays (has {sorted(names)})"], "unknown", []
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
            prov_status, prov_mismatches = check_provenance(z, backend)
    except Exception as e:
        return [f"LOAD FAILED: {type(e).__name__}: {e}"], "unknown", []
    return problems, prov_status, prov_mismatches


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
    total_stale = 0
    total_missing = 0
    for b in BACKENDS:
        d = CACHE / b
        files = sorted(d.glob("*.npz")) if d.is_dir() else []
        present = {p.stem for p in files}
        missing = idset - present
        orphan = present - idset

        bad = []
        stale = []
        unknown_provenance = 0
        for p in files:
            probs, prov_status, prov_mismatches = check_npz(p, b)
            if probs:
                bad.append((p, "; ".join(probs)))
            if prov_status == "unknown":
                unknown_provenance += 1
            elif prov_mismatches:
                stale.append((p, "; ".join(prov_mismatches)))

        healthy = len(files) - len(bad)
        status = ("OK" if (len(files) == len(ids) and not missing and not orphan
                           and not bad and not stale) else "ISSUES")
        if status != "OK":
            all_clean = False
        print(f"[{b}] {status}: {healthy}/{len(ids)} healthy | "
              f"corrupt={len(bad)} missing={len(missing)} orphan={len(orphan)} "
              f"stale={len(stale)} | provenance unknown={unknown_provenance} (pre-dates recording)")

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
        for p, reason in stale[:25]:
            print(f"    STALE   {p.name} -> {reason}")
        if len(stale) > 25:
            print(f"    ... and {len(stale) - 25} more stale (provenance mismatch)")

        if args.delete_bad and bad:
            for p, _ in bad:
                p.unlink()
            print(f"    deleted {len(bad)} bad file(s)")
        total_bad += len(bad)
        total_stale += len(stale)
        total_missing += len(missing)

    print()
    if all_clean:
        print("ALL CLEAN — every backend has all clips, no corruption, no stale provenance.")
    else:
        actions = []
        if total_bad and not args.delete_bad:
            actions.append("re-run with --delete-bad, then `extract_landmarks.py "
                           "--extractor all` to repair corrupt clips")
        if total_missing:
            actions.append("run `extract_landmarks.py --extractor all` to fill missing clips")
        if total_stale:
            actions.append("stale clips have valid data but were built under different "
                           "settings than the current pipeline -- re-extract with --force "
                           "if they need to match (not auto-deleted; that's a judgment call)")
        print("ISSUES FOUND. " + " | ".join(actions))
    sys.exit(0 if all_clean else 1)


if __name__ == "__main__":
    main()
