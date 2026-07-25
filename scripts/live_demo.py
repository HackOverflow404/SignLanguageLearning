#!/usr/bin/env python3
"""live_demo.py -- sign into the webcam, get the closest of the 20 Phase-2 signs live.

The Phase-2 "done when": the extract -> normalize -> feature -> DTW pipeline, run on a
live camera instead of cached clips. Each frame the extractor's LIVE running mode
produces a Pose; the last N poses are a sliding window that gets featurized
(features.py) and graded against the 20-sign reference bank (dtw_grader). The current
best guess + distance are drawn over the skeleton, updated continuously.

Sign boundaries are handled the simple way for now -- a fixed sliding window, always
showing the current best guess (continuous segmentation is Phase 7). Press `c` to
clear the window between attempts (handy for the mother-vs-father location flip).

Handedness/position must match the references (ASL Citizen, NOT flipped, mirrored=
False), so grading always runs on the RAW frame. `--mirror` only flips the DISPLAY
(selfie view); it never touches what is graded.

    .venv/bin/python scripts/live_demo.py                 # MediaPipe, mirrored display
    .venv/bin/python scripts/live_demo.py --selftest      # no camera: verify wiring on cached clips
    .venv/bin/python scripts/live_demo.py --extractor dwpose

Default config: MediaPipe extractor, ShoulderNormalizer (global + local_hand),
velocity on, graded confidence, DTW nearest-reference -- all overridable via the
shared flags in aslcv.pipeline_config (--face/--legs-feet/--confidence/
--no-velocity/--no-local-hand; see --help), the SAME module eval_slice.py and
eval_minimal_pairs.py build their pipeline through, so a number from any of the
three scripts is built identically unless a flag says otherwise.
"""
import argparse
import csv
import threading
import time
from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from aslcv.extractor.base import Pose, RunningMode
from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC, MediaPipePoseExtractor
from aslcv.grading.dtw_grader import DTWGrader
from aslcv.pipeline_config import add_pipeline_args, build_pipeline

REPO = Path(__file__).resolve().parents[1]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_extractor(name, running_mode):
    """(extractor, skeleton) for a backend. Live grading always uses mirrored=False
    so positions/handedness match the non-flipped ASL Citizen references."""
    if name == "mediapipe":
        return MediaPipePoseExtractor(mirrored=False, running_mode=running_mode), MEDIAPIPE_HOLISTIC
    if name in ("dwpose", "rtmw", "vitpose"):
        from importlib import import_module
        cls = {"dwpose": "DWPoseExtractor", "rtmw": "RTMWExtractor", "vitpose": "ViTPoseExtractor"}[name]
        mod = import_module(f"aslcv.extractor.{name}")
        return getattr(mod, cls)(running_mode=running_mode), COCO_WHOLEBODY
    raise ValueError(f"unknown extractor {name!r}")


def slice_rows(signs, split):
    wanted = set(signs)
    return [r for r in csv.DictReader(open(REPO / "data" / "manifest.csv"))
            if r["id_gloss"] in wanted and r["split"] == split]


def cap_per_sign(rows, max_refs):
    """Keep at most `max_refs` clips per sign (speeds the demo's reference bank)."""
    if not max_refs:
        return rows
    seen = defaultdict(int)
    out = []
    for r in rows:
        if seen[r["id_gloss"]] < max_refs:
            out.append(r)
            seen[r["id_gloss"]] += 1
    return out


def make_grader(args):
    signs = yaml.safe_load(open(REPO / "curriculum.yaml"))["phase2_slice"]
    cache_dir = REPO / "data" / "cache" / args.extractor
    if not cache_dir.is_dir():
        raise SystemExit(f"no cache at {cache_dir} -- run scripts/extract_landmarks.py --extractor {args.extractor}")
    skeleton = MEDIAPIPE_HOLISTIC if args.extractor == "mediapipe" else COCO_WHOLEBODY
    pipeline = build_pipeline(args, skeleton, extractor_name=args.extractor)
    train = cap_per_sign(slice_rows(signs, "train"), args.max_refs)
    print(f"building reference bank: {len(train)} clips over {len(signs)} signs "
          f"({args.extractor}) ...")
    t0 = time.time()
    grader = DTWGrader.build(pipeline, cache_dir, signs, train, agg=args.agg, band=args.band)
    print(f"  built in {time.time() - t0:.1f}s"
          + (f"  ({grader.missing} skipped: cache missing)" if grader.missing else ""))
    return grader, pipeline, skeleton, signs, cache_dir


# ---------------------------------------------------------------- overlay ----

def draw_overlay(canvas, ranked, n_win, win_cap, fps, grade_ms):
    h, w = canvas.shape[:2]
    band = canvas.copy()
    cv2.rectangle(band, (0, 0), (w, 96), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.55, canvas, 0.45, 0, canvas)
    if ranked:
        top_sign, top_dist = ranked[0]
        cv2.putText(canvas, top_sign, (16, 48), FONT, 1.4, (80, 255, 120), 3)
        cv2.putText(canvas, f"dist {top_dist:.3f}", (18, 82), FONT, 0.7, (190, 255, 200), 2)
        for i, (s, d) in enumerate(ranked[1:4]):
            cv2.putText(canvas, f"{i + 2}. {s} ({d:.3f})", (w - 300, 30 + 26 * i),
                        FONT, 0.6, (200, 200, 200), 1)
    else:
        cv2.putText(canvas, "collecting frames...", (16, 52), FONT, 1.0, (60, 200, 255), 2)
    status = (f"window {n_win}/{win_cap}   {fps:.0f} fps   "
              f"grade {grade_ms:.0f} ms   [q]uit  [c]lear")
    cv2.putText(canvas, status, (16, h - 14), FONT, 0.55, (220, 220, 220), 1)


# ------------------------------------------------------------------- live ----

def run_live(args):
    grader, pipeline, skeleton, signs, _ = make_grader(args)
    k = len(skeleton.names)
    zero_pose = lambda: Pose(np.zeros((k, 2), np.float32), np.zeros(k, np.float32))

    window = deque(maxlen=args.window)
    win_lock = threading.Lock()
    shared = {"ranked": None, "ms": 0.0}
    res_lock = threading.Lock()
    stop = threading.Event()

    def grade_loop():
        while not stop.is_set():
            with win_lock:
                snap = list(window)
            if len(snap) < args.min_frames:
                time.sleep(0.03)
                continue
            t0 = time.time()
            try:
                ranked = grader.grade(pipeline.assemble(snap).features)
            except Exception as exc:  # keep the demo alive on a bad window
                print("grade error:", exc)
                ranked = None
            with res_lock:
                shared["ranked"] = ranked
                shared["ms"] = (time.time() - t0) * 1000.0

    print(f"opening camera {args.camera} in LIVE mode (extractor {args.extractor}) ...")
    extractor, _ = build_extractor(args.extractor, RunningMode.LIVE)
    worker = threading.Thread(target=grade_loop, name="grader", daemon=True)
    worker.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        stop.set()
        extractor.close()
        raise SystemExit(f"cannot open camera {args.camera}")

    print("controls: [q]/ESC quit   [c] clear window")
    fps_t, fps_n, fps = time.time(), 0, 0.0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            pose = extractor.extract(frame)  # RAW frame, mirrored=False -> matches refs
            with win_lock:
                window.append(pose if pose is not None else zero_pose())

            canvas = extractor.draw(frame, pose) if pose is not None else frame.copy()
            if args.mirror:
                canvas = cv2.flip(canvas, 1)  # display-only selfie flip
            if canvas.shape[1] != 960:
                canvas = cv2.resize(canvas, (960, int(960 * canvas.shape[0] / canvas.shape[1])))

            with res_lock:
                ranked, ms = shared["ranked"], shared["ms"]
            with win_lock:
                n_win = len(window)
            draw_overlay(canvas, ranked, n_win, args.window, fps, ms)
            cv2.imshow("ASL live demo", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                with win_lock:
                    window.clear()
                with res_lock:
                    shared["ranked"] = None

            fps_n += 1
            if time.time() - fps_t >= 0.5:
                fps = fps_n / (time.time() - fps_t)
                fps_t, fps_n = time.time(), 0
    finally:
        stop.set()
        worker.join(timeout=1.0)
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------- selftest ----

def _poses_from_npz(path):
    with np.load(path) as d:
        kp, sc = d["keypoints"], d["scores"]
    return [Pose(kp[t], sc[t]) for t in range(len(kp))]


def run_selftest(args):
    """No camera: push cached clips' frames through the SAME window -> assemble ->
    grade path the live loop uses, so the wiring is verifiable offline. If this names
    signs correctly (and flips mother<->father), a live failure is camera/lighting."""
    grader, pipeline, skeleton, signs, cache_dir = make_grader(args)
    val = slice_rows(signs, "val")

    # a mother clip, a father clip, then a few others -- demonstrates the flip
    picks, taken = [], set()
    for target in ("mother", "father"):
        row = next((r for r in val if r["id_gloss"] == target), None)
        if row:
            picks.append(row)
            taken.add(row["video_id"])
    for r in val:
        if len(picks) >= 6:
            break
        if r["video_id"] not in taken:
            picks.append(r)

    print(f"\nselftest: {len(picks)} val clips through the live grade path "
          f"(sliding window = last {args.window} frames)\n")
    correct = 0
    for r in picks:
        npz = cache_dir / f"{r['video_id']}.npz"
        if not npz.exists():
            print(f"  {r['id_gloss']:<8} (cache missing, skipped)")
            continue
        window = deque(_poses_from_npz(npz), maxlen=args.window)  # emulate the sliding window
        ranked = grader.grade(pipeline.assemble(list(window)).features)
        hit = ranked[0][0] == r["id_gloss"]
        correct += hit
        top3 = ", ".join(f"{s}({d:.3f})" for s, d in ranked[:3])
        print(f"  true={r['id_gloss']:<8} guess={ranked[0][0]:<8} {'OK ' if hit else '   '} top3: {top3}")
    print(f"\n  {correct}/{len(picks)} named correctly -- pipeline wiring OK (this is the offline path).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="mediapipe",
                    choices=["mediapipe", "dwpose", "rtmw", "vitpose"])
    ap.add_argument("--camera", type=int, default=0, help="cv2 VideoCapture index")
    ap.add_argument("--window", type=int, default=60, help="sliding window length (frames)")
    ap.add_argument("--min-frames", type=int, default=20, help="frames needed before grading")
    ap.add_argument("--agg", default="min", choices=["min", "mean"])
    ap.add_argument("--band", type=int, default=None, help="Sakoe-Chiba radius (frames); faster DTW")
    ap.add_argument("--max-refs", type=int, default=None, help="cap reference clips per sign (faster)")
    ap.add_argument("--mirror", dest="mirror", action="store_true", default=True,
                    help="selfie-mirror the DISPLAY (default on; grading uses raw frame)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--selftest", action="store_true",
                    help="no camera: run the grade path on cached val clips")
    add_pipeline_args(ap)
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
