"""Replay logged live sentence attempts (diagnose_demo.py --sentence
--log-attempts) through the real CaptureBuffer under different settings --
tuning sentence-mode capture against real recordings instead of the
synthetic energy patterns Phase 7 step 5 was verified on.

Per attempt it reports, in frames AND seconds (the settle window is counted
in frames, so its real-time length depends on the session's fps):
  - fidelity check: replay with the live parameters must reproduce the live
    settle point exactly, or nothing else here can be trusted;
  - longest internal pause: the longest below-threshold gap between motion
    bursts inside the captured attempt -- the settle window must exceed it;
  - per candidate settle_frames: where the capture settles, and whether it
    settled PREMATURELY (real motion resumed after the settle point, i.e. the
    learner was still signing). Motion in the last --tail-ignore seconds of
    the stream is ignored: that's the hand reaching for the keyboard.

    .venv/bin/python scripts/replay_attempts.py                       # all logged attempts
    .venv/bin/python scripts/replay_attempts.py --settle 20,30,45,60 --plot
    .venv/bin/python scripts/replay_attempts.py data/live_sessions/<stamp>.npz --regrade
"""
import argparse
import functools
from pathlib import Path

import numpy as np

from aslcv.attempt_log import STATE_CODES, load_attempt, longest_internal_pause, replay_capture
from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.features import hand_motion_energy
from aslcv.grading.embedding_grader import EmbeddingGrader

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "data" / "live_sessions"
DEFAULT_CHECKPOINT = REPO / "models" / "embedding_grader"


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _sec(frames, fps):
    return f"{frames / fps:.2f}s" if fps else "?s"


def resumed_motion(energy, threshold, after, ignore_from, min_run=3):
    """True if `energy` has a run of >= `min_run` above-threshold frames
    starting after index `after` and before `ignore_from`: the learner kept
    signing past the settle point. A single-frame blip is tracking noise."""
    run = 0
    for i in range(after + 1, ignore_from):
        run = run + 1 if energy[i] > threshold else 0
        if run >= min_run:
            return True
    return False


def plot_attempt(att, energy, threshold, live_span, replays, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(energy, color="0.25", lw=1, label="hand motion energy")
    ax.axhline(threshold, color="tab:red", ls="--", lw=1, label=f"threshold {threshold:g}")
    if live_span:
        ax.axvspan(*live_span, color="tab:blue", alpha=0.12, label="live capture")
    for (settle, r), color in zip(replays, ("tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink")):
        if r.settle_index is not None:
            ax.axvline(r.settle_index, color=color, lw=1.4, label=f"settle={settle} -> settles")
    ax.set_xlabel(f"frame ({att.fps:.1f} fps)" if att.fps else "frame")
    ax.set_title(f"{att.path.name}  \"{att.meta['sentence']}\"  [{att.meta['event']}]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("attempts", nargs="*", type=Path,
                    help=f"logged .npz files (default: every attempt in {LOG_DIR.relative_to(REPO)}/)")
    ap.add_argument("--settle", type=_ints, default=None,
                    help="comma-separated settle_frames candidates (default: the live value plus a spread)")
    ap.add_argument("--max-frames", type=int, default=None, help="override the capture cap (default: live value)")
    ap.add_argument("--threshold-scale", type=float, default=1.0,
                    help="multiply the live motion threshold by this (default 1.0)")
    ap.add_argument("--tail-ignore", type=float, default=1.5,
                    help="seconds at the end of each stream to ignore for premature-settle detection")
    ap.add_argument("--plot", action="store_true", help="write an energy-trace PNG next to each attempt")
    ap.add_argument("--regrade", action="store_true",
                    help="re-run align_and_grade on each candidate's replayed capture (needs the checkpoint)")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    args = ap.parse_args()

    paths = args.attempts or sorted(LOG_DIR.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no logged attempts -- record some with "
                         f"diagnose_demo.py --sentence \"...\" --log-attempts")

    grader = EmbeddingGrader.build(args.checkpoint, which=args.which)
    skeleton = MEDIAPIPE_HOLISTIC if grader.extractor == "mediapipe" else COCO_WHOLEBODY
    energy_fn = functools.partial(hand_motion_energy, grader.pipeline.normalizer, skeleton)

    summary = {}  # settle -> [n_settled, n_premature, n_never_settled]
    pauses_s = []
    for path in paths:
        att = load_attempt(path)
        cfg = att.meta["capture"]
        if att.meta["extractor"] != grader.extractor:
            print(f"\n{path.name}: recorded with {att.meta['extractor']}, checkpoint uses "
                  f"{grader.extractor} -- skipped")
            continue
        threshold = cfg["motion_threshold"] * args.threshold_scale
        max_frames = args.max_frames or cfg["max_frames"]
        settles = args.settle or sorted({cfg["settle_frames"], 15, 20, 30, 45, 60})
        fps = att.fps
        energy = np.asarray(energy_fn(att.poses))
        ignore_from = len(energy) - (int(round(args.tail_ignore * fps)) if fps else 0)

        print(f"\n{path.name}  \"{att.meta['sentence']}\"  event={att.meta['event']}  "
              f"{len(att.poses)} frames @ {fps:.1f} fps" if fps else f"\n{path.name}")

        live = replay_capture(att.poses, energy_fn, cfg["motion_threshold"],
                              cfg["settle_frames"], cfg["preroll"], cfg["max_frames"])
        live_settled = np.nonzero(att.states == STATE_CODES["settled"])[0]
        live_idx = int(live_settled[0]) if len(live_settled) else None
        ok = live.settle_index == live_idx
        print(f"  replay fidelity: live settled at {live_idx}, replay at {live.settle_index} "
              f"-> {'OK' if ok else 'MISMATCH (results below are not trustworthy)'}")

        if live.span:
            pause = longest_internal_pause(energy, cfg["motion_threshold"], *live.span)
            pauses_s.append(pause / fps if fps else None)
            print(f"  live capture {live.span} ({_sec(live.span[1] - live.span[0], fps)}); "
                  f"longest internal pause {pause} frames ({_sec(pause, fps)}) "
                  f"vs live settle window {cfg['settle_frames']} ({_sec(cfg['settle_frames'], fps)})")

        replays = []
        for settle in settles:
            r = replay_capture(att.poses, energy_fn, threshold, settle, cfg["preroll"], max_frames)
            replays.append((settle, r))
            summary.setdefault(settle, [0, 0, 0])
            if r.settle_index is None:
                summary[settle][2] += 1
                print(f"  settle={settle:<3} never settled ({r.state})")
                continue
            premature = resumed_motion(energy, threshold, r.settle_index, ignore_from)
            capped = (r.span[1] - r.span[0]) >= max_frames
            summary[settle][0] += 1
            summary[settle][1] += premature
            line = (f"  settle={settle:<3} ({_sec(settle, fps)}) captures {r.span} "
                    f"len {r.span[1] - r.span[0]}{'  HIT CAP' if capped else ''}"
                    f"{'  PREMATURE: motion resumed after settle' if premature else ''}")
            if args.regrade:
                from aslcv.grading.alignment import align_and_grade
                from aslcv.production import GlossRuleEngine
                seq = GlossRuleEngine().gloss(att.meta["sentence"])
                _, graded = align_and_grade(grader, att.poses[r.span[0]:r.span[1]], seq)
                marks = []
                for g in graded:
                    judged = [v for v in g.result.parameters.values() if v.correct is not None]
                    marks.append(f"{g.target_sign}:{sum(v.correct for v in judged)}/{len(judged)}")
                line += "  | " + " ".join(marks)
            print(line)

        if args.plot:
            out = path.with_suffix(".png")
            plot_attempt(att, energy, cfg["motion_threshold"], live.span, replays, out)
            print(f"  plot -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")

    print("\n" + "=" * 70)
    print(f"summary over {len(paths)} attempt(s)  (premature = learner still signing after settle)")
    for settle in sorted(summary):
        n_set, n_pre, n_never = summary[settle]
        print(f"  settle={settle:<3}  settled {n_set}  premature {n_pre}  never settled {n_never}")
    known = [p for p in pauses_s if p is not None]
    if known:
        print(f"  longest internal pause across attempts: max {max(known):.2f}s, "
              f"median {float(np.median(known)):.2f}s")


if __name__ == "__main__":
    main()
