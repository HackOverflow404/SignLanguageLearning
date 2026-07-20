#!/usr/bin/env python3
"""Batch pose extraction over data/manifest.csv -> data/cache/{extractor}/{video_id}.npz.

    .venv/bin/python scripts/extract_landmarks.py --extractor dwpose [--limit N] [--force]
    .venv/bin/python scripts/extract_landmarks.py --extractor all      # every backend, in turn

Caches RAW keypoints in each extractor's NATIVE topology -- NO normalization at
cache time (normalization is a read-time transform in features.py, so the cache
never needs re-extracting when it changes). Resumable: an existing .npz is skipped
unless --force (writes are atomic, so an interrupted run never leaves a half file
that looks complete). Frames with no detection are written as all-zero
keypoints/scores -- never dropped, because timing matters for DTW.

Per clip -> data/cache/{extractor}/{video_id}.npz with arrays:
    keypoints    (T, K, 2) float32   pixel coords in the extractor's native topology
    scores       (T, K)    float32   per-keypoint confidence
    blendshapes  (T, 52)   float32   MediaPipe only -- ARKit facial coeffs (NMM)
and metadata: extractor, skeleton, K, fps, source_video.

Also writes data/cache/{extractor}/_manifest.csv = the input rows + n_frames +
hand_dropout_rate per cached clip.

Handedness: videos are NOT mirrored, so MediaPipe is built mirrored=False and
cv2.flip is never applied (asserted at startup). MediaPipe runs in VIDEO mode
(monotonic timestamps); the rtmlib backends run in IMAGE mode (detect every frame).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifest.csv"
CACHE_ROOT = REPO / "data" / "cache"

HAND_CONF_THRESHOLD = 0.3   # a frame has a hand if max score over hand keypoints >= this
ALL_ORDER = ["mediapipe", "dwpose", "rtmw", "vitpose"]  # fastest / most reliable first


# ---- backend registry (lazy imports so the main process only loads what it uses) ----

def _build_mediapipe():
    from aslcv.extractor.base import RunningMode
    from aslcv.extractor.mediapipe import MediaPipePoseExtractor
    return MediaPipePoseExtractor(mirrored=False, running_mode=RunningMode.VIDEO)


def _build_dwpose():
    from aslcv.extractor.base import RunningMode
    from aslcv.extractor.dwpose import DWPoseExtractor
    return DWPoseExtractor(running_mode=RunningMode.IMAGE)


def _build_rtmw():
    from aslcv.extractor.base import RunningMode
    from aslcv.extractor.rtmw import RTMWExtractor
    return RTMWExtractor(running_mode=RunningMode.IMAGE)


def _build_vitpose():
    from aslcv.extractor.base import RunningMode
    from aslcv.extractor.vitpose import ViTPoseExtractor
    return ViTPoseExtractor(running_mode=RunningMode.IMAGE)


BUILDERS = {
    "mediapipe": _build_mediapipe,
    "dwpose": _build_dwpose,
    "rtmw": _build_rtmw,
    "vitpose": _build_vitpose,
}


def skeleton_for(name):
    """(skeleton, skeleton_name) via a lightweight import -- no model load."""
    if name == "mediapipe":
        from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
        return MEDIAPIPE_HOLISTIC, "mediapipe_holistic"
    from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
    return COCO_WHOLEBODY, "coco_wholebody"


def hand_indices(name, skeleton):
    if name == "mediapipe":
        return np.array([i for i, n in enumerate(skeleton.names) if "hand" in n])
    from aslcv.extractor.coco_wholebody import LEFT_HAND, RIGHT_HAND
    return np.array(list(LEFT_HAND) + list(RIGHT_HAND))


# ---- per-video extraction ---------------------------------------------------

def extract_video(extractor, video_path, K, want_blendshapes):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    kp_list, sc_list, bs_list = [], [], []
    while True:
        ok, frame = cap.read()         # BGR, never flipped
        if not ok:
            break
        pose = extractor.extract(frame)
        if pose is None:
            kp_list.append(np.zeros((K, 2), np.float32))
            sc_list.append(np.zeros(K, np.float32))
            if want_blendshapes:
                bs_list.append(np.zeros(52, np.float32))
        else:
            kp_list.append(pose.keypoints.astype(np.float32))
            sc_list.append(pose.scores.astype(np.float32))
            if want_blendshapes:
                bs = pose.blendshapes
                bs_list.append(np.zeros(52, np.float32) if bs is None else bs.astype(np.float32))
    cap.release()
    if not kp_list:
        raise RuntimeError(f"no frames decoded from {video_path}")
    keypoints = np.stack(kp_list)      # (T, K, 2)
    scores = np.stack(sc_list)         # (T, K)
    blendshapes = np.stack(bs_list) if want_blendshapes else None
    return keypoints, scores, blendshapes, float(fps)


def hand_dropout_rate(scores, hand_idx):
    """Fraction of frames with no hand detected (max hand score < threshold)."""
    if len(hand_idx) == 0 or scores.shape[0] == 0:
        return 0.0
    present = scores[:, hand_idx].max(axis=1) >= HAND_CONF_THRESHOLD
    return float(1.0 - present.mean())


def write_npz(path, keypoints, scores, blendshapes, meta):
    """Atomic write: savez to a temp file (via a handle, so numpy doesn't re-append
    .npz), then os.replace -- so an interrupted write never yields a 'complete' file."""
    tmp = Path(str(path) + ".tmp")
    arrays = dict(keypoints=keypoints, scores=scores, **meta)
    if blendshapes is not None:
        arrays["blendshapes"] = blendshapes
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


# ---- per-backend run --------------------------------------------------------

def run_backend(name, rows, force):
    print(f"\n{'=' * 70}\n[{name}] START  ({len(rows)} clips)\n{'=' * 70}", flush=True)
    skeleton, skeleton_name = skeleton_for(name)
    K = len(skeleton.names)
    hand_idx = hand_indices(name, skeleton)
    want_blendshapes = (name == "mediapipe")
    cache_dir = CACHE_ROOT / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] videos NOT mirrored -> no cv2.flip; K={K}, skeleton={skeleton_name}",
          flush=True)

    todo = [r for r in rows if force or not (cache_dir / f"{r['video_id']}.npz").exists()]
    skipped = len(rows) - len(todo)
    print(f"[{name}] {skipped} already cached, {len(todo)} to process", flush=True)

    processed = failed = total_frames = 0
    dropout_sum = 0.0
    t0 = time.time()

    if todo:
        extractor = BUILDERS[name]()
        if name == "mediapipe":
            assert extractor._mirrored is False, "mediapipe must be built mirrored=False"
        is_tty = sys.stderr.isatty()
        iterator = tqdm(todo, desc=name, file=sys.stderr) if is_tty else todo
        try:
            for i, r in enumerate(iterator, 1):
                vid = r["video_id"]
                try:
                    kp, sc, bs, fps = extract_video(
                        extractor, REPO / r["video_path"], K, want_blendshapes)
                    meta = dict(extractor=name, skeleton=skeleton_name, K=K, fps=fps,
                                source_video=r["video_path"])
                    write_npz(cache_dir / f"{vid}.npz", kp, sc, bs, meta)
                    processed += 1
                    total_frames += kp.shape[0]
                    dropout_sum += hand_dropout_rate(sc, hand_idx)
                except Exception as e:
                    failed += 1
                    print(f"[{name}] FAILED {vid}: {e}", file=sys.stderr, flush=True)
                if not is_tty and (i % 100 == 0 or i == len(todo)):
                    print(f"[{name}] {i}/{len(todo)}  (ok={processed} fail={failed}, "
                          f"{(time.time() - t0) / 60:.1f} min)", flush=True)
        finally:
            extractor.close()

    dt = time.time() - t0
    mean_frames = total_frames / processed if processed else 0.0
    mean_dropout = dropout_sum / processed if processed else 0.0
    print(f"[{name}] DONE: processed={processed} skipped={skipped} failed={failed} | "
          f"mean_frames={mean_frames:.1f} | mean hand-dropout={mean_dropout * 100:.1f}% | "
          f"{dt / 60:.1f} min", flush=True)
    write_output_manifest(name, cache_dir, rows, hand_idx)


def write_output_manifest(name, cache_dir, rows, hand_idx):
    """Input rows + n_frames + hand_dropout_rate, for every row with a cached .npz."""
    if not rows:
        return
    out = cache_dir / "_manifest.csv"
    cols = list(rows[0].keys()) + ["n_frames", "hand_dropout_rate"]
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            npz = cache_dir / f"{r['video_id']}.npz"
            if not npz.exists():
                continue
            try:
                with np.load(npz) as d:
                    sc = d["scores"]
                row = dict(r)
                row["n_frames"] = sc.shape[0]
                row["hand_dropout_rate"] = round(hand_dropout_rate(sc, hand_idx), 4)
                w.writerow(row)
                n += 1
            except Exception as e:
                print(f"[{name}] _manifest skip {r['video_id']}: {e}", file=sys.stderr)
    print(f"[{name}] wrote {out.relative_to(REPO)} ({n} rows)", flush=True)


# ---- main -------------------------------------------------------------------

def read_manifest(limit):
    if not MANIFEST.exists():
        sys.exit(f"ERROR: {MANIFEST} not found (run tools/build_manifest.py first)")
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", required=True, choices=list(BUILDERS) + ["all"])
    ap.add_argument("--limit", type=int, default=None, help="process only the first N clips")
    ap.add_argument("--force", action="store_true", help="re-extract even if a .npz exists")
    ap.add_argument("--workers", type=int, default=1,
                    help="reserved; extraction runs serially (safest for a shared GPU)")
    args = ap.parse_args()

    names = ALL_ORDER if args.extractor == "all" else [args.extractor]
    rows = read_manifest(args.limit)
    print(f"manifest: {len(rows)} clips | backends: {', '.join(names)} | "
          f"python: {sys.executable}", flush=True)

    t0 = time.time()
    ok = True
    for name in names:
        try:
            run_backend(name, rows, args.force)
        except Exception as e:
            ok = False
            print(f"[{name}] BACKEND FAILED (continuing to next): {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
    print(f"\nALL BACKENDS COMPLETE in {(time.time() - t0) / 60:.1f} min "
          f"({'ok' if ok else 'WITH BACKEND FAILURES -- see log'})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
