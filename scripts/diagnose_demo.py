#!/usr/bin/env python3
"""diagnose_demo.py -- live webcam demo of Phase 4's grade_against(attempt, target).

Prompts a target sign, plays that sign's real reference clip on loop next to the live
webcam view, captures your attempt over a sliding window, and shows the PER-PARAMETER
diagnosis from EmbeddingGrader.grade_against_poses -- handshape / major_location /
minor_location / movement / repeated_movement, each MATCH/OFF/insufficient-data with
the model's confidence, plus overall fidelity. Correct one parameter, clear (`c`), and
re-try -- the last verdict stays on screen so you can see what changed.

This is scripts/live_demo.py's own scaffolding (webcam loop, sliding window, mirrored-
DISPLAY-only handling, background-threaded grading, pipeline_config.py wiring) with
exactly two substantive changes: the grader is Phase 4's learned EmbeddingGrader, not
Phase 2's DTWGrader, and the interface is grade_against(attempt, KNOWN target), not
open-set "which sign is this."

HONEST LIMITS (also shown on screen, every frame): this confirms the plumbing and
lets you practice imitating a real reference clip. It does NOT independently verify
ASL correctness -- "all correct" means "matched the reference," not "fluent." A
target with no cached reference clip is refused outright (fail-closed), never graded
without something on screen to imitate.

    .venv/bin/python scripts/diagnose_demo.py                       # mediapipe, cycle the default targets
    .venv/bin/python scripts/diagnose_demo.py --target father        # start on father, [n] still cycles onward
    .venv/bin/python scripts/diagnose_demo.py --targets you,me,water
    .venv/bin/python scripts/diagnose_demo.py --selftest             # no camera: verify the whole path offline
"""
import argparse
import csv
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from aslcv.extractor.base import Pose, RunningMode
from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.grading.embedding_grader import EmbeddingGrader
from aslcv.grading.phonology_labels import ALL_PARAMETERS
from aslcv.pipeline_config import add_pipeline_args, build_pipeline
from live_demo import _poses_from_npz, build_extractor  # reuse, don't rebuild

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifest.csv"
DEFAULT_CHECKPOINT = REPO / "models" / "embedding_grader"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# A small, phonologically clear starting set -- includes the curriculum's built-in
# mother/father minimal pair (differ ONLY in minor_location). [n] cycles this list;
# --target/--targets override it (see resolve_targets).
DEFAULT_TARGETS = ["mother", "father", "you", "me", "water", "thank_you", "yes", "good"]

DISCLAIMER = ("PRACTICE AID ONLY -- confirms plumbing + match to a real reference clip. "
              "Does NOT independently verify ASL correctness.")

PARAM_LABEL = {
    "handshape": "HANDSHAPE", "major_location": "MAJOR LOC", "minor_location": "MINOR LOC",
    "movement": "MOVEMENT", "repeated_movement": "REPEATED",
}


# ------------------------------------------------------------- manifest / clips ----

def manifest_rows():
    return list(csv.DictReader(open(MANIFEST)))


def reference_row(rows, sign):
    """Manifest row to show as the reference clip for `sign` -- prefer a train-split
    clip (same convention as render_clip.py / compose_sentence.py), else the first."""
    matches = [r for r in rows if r["id_gloss"] == sign]
    if not matches:
        return None
    return next((r for r in matches if r["split"] == "train"), matches[0])


def load_reference_frames(video_path, max_width=440):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(max_width * h / w)))
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"reference video decoded 0 frames: {video_path}")
    return frames


class ReferenceLoop:
    """The current target's reference clip, decoded ONCE and looped continuously.

    Mandatory alongside the live view, never optional: a learner can only correct
    toward a reference they can see, and this demo must never imply it can judge
    correctness without one on screen (see resolve_targets' fail-closed check)."""

    def __init__(self):
        self.sign = None
        self.video_id = None
        self.frames: list = []
        self.idx = 0

    def set_target(self, sign, video_id, frames):
        self.sign, self.video_id, self.frames, self.idx = sign, video_id, frames, 0

    def next_frame(self):
        if not self.frames:
            return None
        f = self.frames[self.idx % len(self.frames)]
        self.idx += 1
        return f


def resolve_targets(args, rows, grader):
    """The ordered [n]-cycle list, each entry pre-validated to have a real cached
    reference clip. Refuses to start rather than let a target reach the live loop
    with nothing to show on screen (fail-closed, same spirit as the rest of the
    project's out-of-scope refusals)."""
    targets = args.targets.split(",") if args.targets else list(DEFAULT_TARGETS)
    if args.target:
        targets = [args.target] + [t for t in targets if t != args.target]

    resolved = []
    for sign in targets:
        if sign not in grader.signs:
            raise SystemExit(f"target {sign!r} is not a sign this checkpoint knows; have {grader.signs}")
        row = reference_row(rows, sign)
        video_path = REPO / row["video_path"] if row else None
        if row is None or not video_path.exists():
            raise SystemExit(
                f"no cached reference VIDEO for target {sign!r} -- refusing to grade against it "
                f"without a reference to imitate on screen. Drop it from --targets/--target, or "
                f"fix data/manifest.csv.")
        resolved.append((sign, row))
    return resolved


# ------------------------------------------------------------- config verification --

_PIPELINE_FIELDS = ("face", "legs_feet", "confidence", "binary_threshold", "velocity",
                    "depth_proxies", "trim_to_motion", "motion_threshold", "motion_pad_frames")


def verify_pipeline_matches_checkpoint(args, grader, skeleton):
    """Build the live-side FeaturePipeline from CLI flags (prints the resolved config,
    same as every other script routes through pipeline_config.py) and refuse to start
    if it differs from the checkpoint's own TRAINING-TIME config in any field. A silent
    mismatch here would corrupt every verdict without any visible symptom -- fail
    closed instead of guessing."""
    live_pipeline = build_pipeline(args, skeleton, extractor_name=args.extractor)
    saved = grader.pipeline

    mismatches = []
    if args.extractor != grader.extractor:
        mismatches.append(f"extractor: live={args.extractor!r} checkpoint={grader.extractor!r}")
    for field in _PIPELINE_FIELDS:
        lv, sv = getattr(live_pipeline, field), getattr(saved, field)
        if lv != sv:
            mismatches.append(f"{field}: live={lv} checkpoint={sv}")
    if live_pipeline.normalizer.local_hand != saved.normalizer.local_hand:
        mismatches.append(f"local_hand: live={live_pipeline.normalizer.local_hand} "
                           f"checkpoint={saved.normalizer.local_hand}")

    if mismatches:
        raise SystemExit(
            "REFUSING TO START: live feature pipeline does not match this checkpoint's "
            "training-time config -- every verdict would silently be wrong.\n  "
            + "\n  ".join(mismatches)
            + "\nDrop the overriding flag(s), or point --checkpoint at a model trained with this config.")
    print("verified: live pipeline config matches the checkpoint's training-time config.")


# ---------------------------------------------------------------------- overlay ----

def _tag_and_color(verdict):
    if verdict.correct is True:
        return "MATCH", (80, 255, 120)
    if verdict.correct is False:
        return "OFF", (60, 60, 255)
    return "insufficient data", (170, 170, 170)


def draw_verdict(canvas, result, target_sign, stale, n_win, win_cap, fps, grade_ms):
    h, w = canvas.shape[:2]
    band = canvas.copy()
    cv2.rectangle(band, (0, 0), (w, 230), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.6, canvas, 0.4, 0, canvas)

    cv2.putText(canvas, f"TARGET: {target_sign}", (14, 26), FONT, 0.8, (255, 220, 120), 2)

    if result is None:
        msg = "no verdict yet -- previous attempt cleared" if stale else "collecting frames..."
        cv2.putText(canvas, msg, (14, 54), FONT, 0.65, (60, 200, 255), 2)
    else:
        fid_color = (140, 140, 140) if stale else (210, 210, 210)
        suffix = "  (PREVIOUS ATTEMPT -- re-signing)" if stale else ""
        cv2.putText(canvas, f"fidelity {result.fidelity:.3f}{suffix}", (14, 54), FONT, 0.6, fid_color, 1)
        y = 80
        for p in ALL_PARAMETERS:
            v = result.parameters[p]
            tag, color = _tag_and_color(v)
            if stale:
                color = tuple(c // 2 for c in color)
            line = f"{PARAM_LABEL[p]:<10} {v.predicted:<12} [{tag}] {v.confidence:.0%}"
            cv2.putText(canvas, line, (14, y), FONT, 0.55, color, 1)
            y += 26

    status = f"window {n_win}/{win_cap}  {fps:.0f} fps  grade {grade_ms:.0f} ms  [q]uit [c]lear [n]ext target"
    cv2.putText(canvas, status, (14, h - 36), FONT, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, DISCLAIMER, (14, h - 12), FONT, 0.45, (0, 165, 255), 1)


def compose_canvas(ref_frame, live_frame, target_sign, result, stale, n_win, win_cap, fps, ms, height=480):
    def fit(frame, fallback_w=360):
        if frame is None:
            return np.zeros((height, fallback_w, 3), np.uint8)
        h, w = frame.shape[:2]
        scale = height / h
        return cv2.resize(frame, (max(1, int(w * scale)), height))

    left = fit(ref_frame)
    right = fit(live_frame)
    cv2.rectangle(left, (0, 0), (left.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(left, f"REFERENCE: {target_sign}", (10, 24), FONT, 0.65, (255, 255, 255), 2)
    draw_verdict(right, result, target_sign, stale, n_win, win_cap, fps, ms)
    return np.hstack([left, right])


# ------------------------------------------------------------------------ live ----

def run_live(args):
    grader = EmbeddingGrader.build(args.checkpoint, which=args.which)
    args.extractor = args.extractor or grader.extractor
    skeleton = MEDIAPIPE_HOLISTIC if args.extractor == "mediapipe" else COCO_WHOLEBODY
    verify_pipeline_matches_checkpoint(args, grader, skeleton)

    rows = manifest_rows()
    cycle = resolve_targets(args, rows, grader)
    print(f"targets ({len(cycle)}, [n] cycles): {', '.join(s for s, _ in cycle)}")
    print(DISCLAIMER)

    ref = ReferenceLoop()
    state = {"idx": 0, "target": None}

    def switch_target(i):
        state["idx"] = i % len(cycle)
        sign, row = cycle[state["idx"]]
        frames = load_reference_frames(REPO / row["video_path"])
        ref.set_target(sign, row["video_id"], frames)
        state["target"] = sign
        print(f"target -> {sign}  (reference clip: {row['video_id']}, {len(frames)} frames)")

    switch_target(0)

    window = deque(maxlen=args.window)
    win_lock = threading.Lock()
    shared = {"result": None, "ms": 0.0, "stale": False}
    res_lock = threading.Lock()
    stop = threading.Event()

    def grade_loop():
        while not stop.is_set():
            with win_lock:
                snap = list(window)
            if len(snap) < args.min_frames:
                time.sleep(0.03)
                continue
            with res_lock:
                target = state["target"]
            t0 = time.time()
            try:
                result = grader.grade_against_poses(snap, target)
            except Exception as exc:  # keep the demo alive on a bad window
                print("grade error:", exc)
                result = None
            with res_lock:
                if result is not None:
                    shared["result"] = result
                    shared["stale"] = False
                shared["ms"] = (time.time() - t0) * 1000.0

    print(f"opening camera {args.camera} in LIVE mode (extractor {args.extractor}"
          f"{', gpu' if args.gpu else ''}) ...")
    extractor, _ = build_extractor(args.extractor, RunningMode.LIVE, gpu=args.gpu)
    k = len(skeleton.names)
    zero_pose = lambda: Pose(np.zeros((k, 2), np.float32), np.zeros(k, np.float32))

    worker = threading.Thread(target=grade_loop, name="grader", daemon=True)
    worker.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        stop.set()
        extractor.close()
        raise SystemExit(f"cannot open camera {args.camera}")

    print("controls: [q]/ESC quit   [c] clear window (keeps last verdict)   [n] next target")
    fps_t, fps_n, fps = time.time(), 0, 0.0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            pose = extractor.extract(frame)  # RAW frame, mirrored=False -> matches refs (issue #6)
            with win_lock:
                window.append(pose if pose is not None else zero_pose())

            live_canvas = extractor.draw(frame, pose) if pose is not None else frame.copy()
            if args.mirror:
                live_canvas = cv2.flip(live_canvas, 1)  # display-only selfie flip

            ref_frame = ref.next_frame()

            with res_lock:
                result, ms, stale = shared["result"], shared["ms"], shared["stale"]
            with win_lock:
                n_win = len(window)

            canvas = compose_canvas(ref_frame, live_canvas, state["target"], result, stale,
                                     n_win, args.window, fps, ms)
            cv2.imshow("Phase 4 diagnose demo", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                with win_lock:
                    window.clear()
                with res_lock:
                    shared["stale"] = True  # last verdict stays visible, just marked stale
            if key == ord("n"):
                switch_target(state["idx"] + 1)
                with win_lock:
                    window.clear()
                with res_lock:
                    shared["result"] = None
                    shared["stale"] = False

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


# --------------------------------------------------------------------- selftest ----

def run_selftest(args):
    """No camera: push cached val clips' frames through the SAME window ->
    grade_against_poses -> verdict path the live loop uses, so the wiring (and the
    headline mother/father head-independence behavior) is verifiable offline."""
    grader = EmbeddingGrader.build(args.checkpoint, which=args.which)
    args.extractor = args.extractor or grader.extractor
    skeleton = MEDIAPIPE_HOLISTIC if args.extractor == "mediapipe" else COCO_WHOLEBODY
    verify_pipeline_matches_checkpoint(args, grader, skeleton)

    rows = manifest_rows()
    val_by_sign = defaultdict(list)
    for r in rows:
        if r["split"] == "val":
            val_by_sign[r["id_gloss"]].append(r)
    cache_dir = REPO / "data" / "cache" / grader.extractor

    def emulate(row):
        npz = cache_dir / f"{row['video_id']}.npz"
        return list(deque(_poses_from_npz(npz), maxlen=args.window))

    def print_result(true_sign, target_sign, result):
        parts = []
        for p in ALL_PARAMETERS:
            v = result.parameters[p]
            tag, _ = _tag_and_color(v)
            parts.append(f"{p}={tag}({v.confidence:.0%})")
        print(f"  attempt={true_sign:<10} target={target_sign:<10} "
              f"fidelity={result.fidelity:.3f}  {' '.join(parts)}")

    print(f"\nselftest: prompt -> grade_against_poses -> verdict path, no camera "
          f"(sliding window = last {args.window} frames)\n")

    print("self-check (attempt graded against its OWN true sign):")
    for sign in ("mother", "father", "you", "me", "water", "thank_you"):
        clips = val_by_sign.get(sign)
        if not clips:
            print(f"  {sign:<10} (no val clip, skipped)")
            continue
        result = grader.grade_against_poses(emulate(clips[0]), sign)
        print_result(sign, sign, result)

    print("\nmother/father minimal-pair cross-check (differ ONLY in minor_location):")
    for true_sign, target_sign in (("father", "mother"), ("mother", "father")):
        clips = val_by_sign.get(true_sign)
        if not clips:
            print(f"  no val clip for {true_sign}, skipped")
            continue
        result = grader.grade_against_poses(emulate(clips[0]), target_sign)
        print_result(true_sign, target_sign, result)

    print("\nselftest complete -- this is the offline path; a live failure is camera/lighting, not grading.")
    print(DISCLAIMER)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--target", default=None, help="starting target sign; [n] still cycles onward from here")
    ap.add_argument("--targets", default=None, help="comma-separated cycle list (default: a curated set incl. mother/father)")
    ap.add_argument("--camera", type=int, default=0, help="cv2 VideoCapture index")
    ap.add_argument("--window", type=int, default=60, help="sliding window length (frames)")
    ap.add_argument("--min-frames", type=int, default=20, help="frames needed before grading")
    ap.add_argument("--mirror", dest="mirror", action="store_true", default=True,
                    help="selfie-mirror the DISPLAY only (default on; grading uses the raw frame)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--extractor", default=None,
                    help="override for verification only -- must match the checkpoint's own "
                         "extractor or the demo refuses to start (default: the checkpoint's own)")
    ap.add_argument("--gpu", action="store_true",
                    help="mediapipe only: request the GPU delegate (~3.3x faster, measured; "
                         "falls back to CPU automatically if unavailable on this machine)")
    ap.add_argument("--selftest", action="store_true", help="no camera: verify the whole path on cached val clips")
    add_pipeline_args(ap)
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
