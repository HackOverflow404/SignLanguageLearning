#!/usr/bin/env python3
"""Phase 3's deferred cheap screen, actually run: FPS/latency on this machine,
hand-tracking jitter, and hand-dropout under real signing -- for all four
extractors, so "MediaPipe wins" can be qualified by more than the minimal-pair
accuracy result alone.

    .venv/bin/python scripts/benchmark_extractors.py [--n-clips 8] [--seed 0] [--gpu]

`--gpu` requests MediaPipe's GPU delegate for the FPS measurement (see
extractor/mediapipe.py's `delegate` docstring for the fallback behavior if
GPU isn't actually usable on this machine); it has no effect on the other
three backends, which already default to `device="cuda"`.

This was originally scoped as part of Phase 3's full-sweep design and explicitly
left un-run ("kept for reference -- run it only if the extractor decision is
reopened") once the minimal-pair accuracy result alone looked decisive. Running it
now closes that gap rather than leaving "MediaPipe is better across the board" an
unverified assumption -- the accuracy result was real, but real-time viability was
never actually checked against the rtmlib backends.

Three measurements:
  1. FPS/latency -- times the SAME extract_video() this repo's actual batch
     extraction uses (imported from extract_landmarks.py, not reimplemented), on a
     deterministic sample of already-downloaded clips. This is genuinely fresh
     inference, not reused from the cache (rtmlib backends default to
     device="cuda"; MediaPipe's Python Tasks API here takes no GPU delegate and
     runs CPU-only -- confirmed by the run log, not assumed).
  2. Hand jitter -- reuses ALREADY-CACHED keypoints (all four backends have full
     cache coverage from Phase 1), no new extraction. Measures the LOW end (25th
     percentile) of hand_motion_energy() -- the exact signal Phase 2's
     trim_to_motion and Phase 5a's retrieval trimming already use, same units, same
     presence-gated zero-fill convention -- restricted to frames where the hand was
     genuinely tracked in both this frame and the last. (An earlier version of this
     script tried to locate the true rest window outside the active span instead,
     the way "hand jitter on a held-still clip" literally reads; that failed almost
     everywhere -- ASL Citizen's rest frames mostly have hands lowered out of
     detection range for every backend, not held up and still, so there was
     usually no rest window with visible hands to measure at all. The low
     percentile of tracked in-clip energy is the closest available proxy: how much
     apparent motion a backend invents even when the hand is closest to
     stationary.) Lower is a steadier detector.
  3. Hand dropout -- two views, both from cached data, no new inference. The
     full-dataset one reads data/cache/{name}/_manifest.csv directly (cheap, but
     dominated by REST-frame dropout that gets trimmed away before grading ever
     sees it). The region-split one (on the FPS/jitter sample) separates dropout
     inside the trimmed active-signing span -- what grading actually uses -- from
     the discarded rest span, via hand_motion_energy()/motion_active_span(), the
     same signal Phase 2's trim_to_motion uses. The active-span number is the one
     that actually matters.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifest.csv"
CACHE = REPO / "data" / "cache"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import, like diagnose_demo.py

from extract_landmarks import ALL_ORDER, BUILDERS, extract_video, hand_indices, skeleton_for  # noqa: E402

from aslcv.extractor.base import RunningMode  # noqa: E402
from aslcv.extractor.mediapipe import MediaPipePoseExtractor  # noqa: E402
from aslcv.features import hand_motion_energy, motion_active_span  # noqa: E402
from aslcv.grading.embedding_dataset import _load_poses_npz  # noqa: E402
from aslcv.normalizer.shoulder import ShoulderNormalizer  # noqa: E402

_NORMALIZER = ShoulderNormalizer(local_hand=True)
_HAND_SCORE_THR = 0.3  # same convention as extract_landmarks.HAND_CONF_THRESHOLD


def sample_clips(n, seed):
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


# ---- 1. FPS / latency --------------------------------------------------------

def benchmark_fps(name, rows, gpu=False):
    skeleton, _ = skeleton_for(name)
    K = len(skeleton.names)
    want_blendshapes = (name == "mediapipe")
    if name == "mediapipe" and gpu:
        extractor = MediaPipePoseExtractor(mirrored=False, running_mode=RunningMode.VIDEO, delegate="gpu")
    else:
        extractor = BUILDERS[name]()
    try:
        # warm up (model/provider init, first-inference JIT) on a throwaway frame,
        # discarded, so the timed loop reflects steady-state per-frame cost only.
        first_path = REPO / rows[0]["video_path"]
        import cv2
        cap = cv2.VideoCapture(str(first_path))
        ok, frame = cap.read()
        cap.release()
        if ok:
            extractor.extract(frame)

        total_frames, total_time = 0, 0.0
        per_clip = []
        for r in rows:
            t0 = time.perf_counter()
            kp, sc, bs, fps = extract_video(extractor, REPO / r["video_path"], K, want_blendshapes)
            dt = time.perf_counter() - t0
            total_frames += kp.shape[0]
            total_time += dt
            per_clip.append(dt / kp.shape[0] * 1000)
        return dict(
            frames=total_frames,
            seconds=total_time,
            ms_per_frame=total_time / total_frames * 1000,
            fps=total_frames / total_time,
            worst_clip_ms_per_frame=max(per_clip),
        )
    finally:
        extractor.close()


# ---- 2. hand jitter: noise floor of hand_motion_energy on tracked frames -----

def benchmark_jitter(name, rows):
    skeleton, _ = skeleton_for(name)
    hand_idx = hand_indices(name, skeleton)
    per_clip_floor = []
    for r in rows:
        npz = CACHE / name / f"{r['video_id']}.npz"
        if not npz.exists():
            continue
        poses = _load_poses_npz(npz)
        energy = hand_motion_energy(_NORMALIZER, skeleton, poses)
        with np.load(npz) as d:
            sc = d["scores"]
        # "tracked" = hand actually detected (same threshold/convention as
        # extract_landmarks.hand_dropout_rate) in BOTH this frame and the last --
        # energy[t] is a real measured delta only when both_present, exactly
        # hand_motion_energy's own zero-fill gate (see its docstring).
        hand_present = sc[:, hand_idx].max(axis=1) >= _HAND_SCORE_THR
        both_present = hand_present[1:] & hand_present[:-1]
        tracked_energy = energy[1:][both_present]
        if len(tracked_energy) < 5:
            continue
        per_clip_floor.append(float(np.percentile(tracked_energy, 25)))
    if not per_clip_floor:
        return dict(n_clips=0, mean=None, p95=None)
    arr = np.array(per_clip_floor)
    return dict(n_clips=len(arr), mean=float(arr.mean()), p95=float(np.percentile(arr, 95)))


# ---- 3. hand dropout, read from the existing full-dataset cache manifest -----

def dropout_from_cache_manifest(name):
    """Full-dataset dropout as extract_landmarks.py already computed it --
    cheap, but includes REST frames (hands lowered, out of detection range),
    which get trimmed away before grading ever sees them. See
    benchmark_dropout_by_region below for the metric that actually matters."""
    path = CACHE / name / "_manifest.csv"
    if not path.exists():
        return dict(n_clips=0, mean=None, p95=None)
    with open(path, newline="") as f:
        rates = [float(r["hand_dropout_rate"]) for r in csv.DictReader(f) if r["hand_dropout_rate"]]
    if not rates:
        return dict(n_clips=0, mean=None, p95=None)
    arr = np.array(rates)
    return dict(n_clips=len(arr), mean=float(arr.mean()), p95=float(np.percentile(arr, 95)))


def benchmark_dropout_by_region(name, rows):
    """Dropout split into the trimmed active-signing span (what grading
    actually uses) vs. the discarded rest span, on the same sample as FPS/
    jitter. This is the corrected view from Phase 3 loose end C's follow-up:
    the full-dataset number above is dominated by rest-frame dropout that
    never reaches grading and overstates the real problem."""
    skeleton, _ = skeleton_for(name)
    hand_idx = hand_indices(name, skeleton)
    active, rest = [], []
    for r in rows:
        npz = CACHE / name / f"{r['video_id']}.npz"
        if not npz.exists():
            continue
        poses = _load_poses_npz(npz)
        energy = hand_motion_energy(_NORMALIZER, skeleton, poses)
        start, stop = motion_active_span(energy, 0.02, pad_frames=3)
        with np.load(npz) as d:
            sc = d["scores"]
        present = sc[:, hand_idx].max(axis=1) >= _HAND_SCORE_THR
        rest_mask = np.ones(len(present), bool)
        rest_mask[start:stop] = False
        if rest_mask.sum() > 0:
            rest.append(1 - present[rest_mask].mean())
        if stop > start:
            active.append(1 - present[start:stop].mean())
    return dict(
        n_clips=len(active),
        active_mean=float(np.mean(active)) if active else None,
        rest_mean=float(np.mean(rest)) if rest else None,
    )


# ---- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-clips", type=int, default=8, help="clips to sample for FPS timing + jitter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extractor", default="all", choices=ALL_ORDER + ["all"])
    ap.add_argument("--gpu", action="store_true",
                    help="mediapipe only: request the GPU delegate for the FPS timing")
    args = ap.parse_args()

    names = ALL_ORDER if args.extractor == "all" else [args.extractor]
    rows = sample_clips(args.n_clips, args.seed)
    print(f"sample: {len(rows)} clips (seed={args.seed}): "
          f"{', '.join(r['id_gloss'] for r in rows)}\n")

    results = {}
    for name in names:
        use_gpu = args.gpu and name == "mediapipe"
        print(f"[{name}] timing FPS/latency ({len(rows)} clips, fresh inference"
              f"{', gpu' if use_gpu else ''})...", flush=True)
        fps_res = benchmark_fps(name, rows, gpu=use_gpu)
        print(f"[{name}] jitter noise floor (cached keypoints, no new inference)...", flush=True)
        jitter_res = benchmark_jitter(name, rows)
        print(f"[{name}] dropout by region (cached keypoints, no new inference)...", flush=True)
        dropout_full = dropout_from_cache_manifest(name)
        dropout_region = benchmark_dropout_by_region(name, rows)
        results[name] = dict(fps=fps_res, jitter=jitter_res, dropout_full=dropout_full,
                              dropout_region=dropout_region)
        print(f"[{name}] done\n", flush=True)

    print("=" * 100)
    print(f"{'backend':<12} {'fps':>8} {'ms/frame':>10} | {'jitter floor':>13} {'jitter p95':>11} | "
          f"{'dropout active%':>15} {'dropout rest%':>14} {'dropout full%':>14}")
    print("-" * 100)
    for name in names:
        r = results[name]
        f = r["fps"]
        j = r["jitter"]
        df = r["dropout_full"]
        dr = r["dropout_region"]
        label = name + ("+gpu" if (args.gpu and name == "mediapipe") else "")
        j_mean = f"{j['mean']:.4f}" if j["mean"] is not None else "n/a"
        j_p95 = f"{j['p95']:.4f}" if j["p95"] is not None else "n/a"
        d_active = f"{dr['active_mean'] * 100:.1f}" if dr["active_mean"] is not None else "n/a"
        d_rest = f"{dr['rest_mean'] * 100:.1f}" if dr["rest_mean"] is not None else "n/a"
        d_full = f"{df['mean'] * 100:.1f}" if df["mean"] is not None else "n/a"
        print(f"{label:<12} {f['fps']:>8.1f} {f['ms_per_frame']:>10.2f} | {j_mean:>13} {j_p95:>11} | "
              f"{d_active:>15} {d_rest:>14} {d_full:>14}")
    print("=" * 100)
    print("jitter = 25th-pct hand_motion_energy on tracked frames (normalizer's local-hand units,")
    print("same as the 0.02 trim_to_motion threshold) -- lower is steadier. dropout active% is the")
    print("metric that matters (inside the trimmed signing span, over the sample above); dropout")
    print("rest% is the discarded rest span (same sample); dropout full% is the naive full-dataset")
    print("number from data/cache/{name}/_manifest.csv, kept for comparison -- it overstates the")
    print("real problem by mixing in the rest span.")


if __name__ == "__main__":
    main()
