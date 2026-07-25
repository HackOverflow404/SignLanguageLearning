#!/usr/bin/env python3
"""render_clip.py -- look up an ASL-LEX 2.0 entry_id, run ONE extractor's class over
its video, and both SHOW the drawn skeleton live and save the annotated clip to disk.

Looks up by `entry_id` (e.g. "dog", "what_1") -- ASL-LEX's EntryID, "an English word
that uniquely identifies each entry," and exactly what manifest.csv's `id_gloss`
column already carries per row for these 60 curriculum signs. Deliberately NOT
`lemma_id`: ASL-LEX's LemmaID collapses phonological/inflectional variants together
(dog_2/dog_3/dog_4 -> lemma "dog"), which is ambiguous for picking one specific sign
-- entry_id is the finer-grained, unambiguous handle. The underlying join still runs
on `asllex_code` internally (CLAUDE.md's settled cross-dataset join key -- gloss
strings drift between ASL-LEX/Citizen/curriculum.yaml); this only changes what a
human types at the CLI, not how rows are actually resolved across datasets.

One entry_id can have many clips (different signers/splits); pick one with
--video-id or --list them.

Builds the extractor via extract_landmarks.py's own BUILDERS, so this uses the EXACT
same construction (mirrored=False, running mode) that produced the cache -- what you
see here is what the pipeline actually sees, not a live-demo mirrored/webcam variant.

    .venv/bin/python scripts/render_clip.py dog --extractor dwpose
    .venv/bin/python scripts/render_clip.py dog --list                  # show every clip for this entry
    .venv/bin/python scripts/render_clip.py dog --extractor dwpose --video-id 5346957049353451-WE
    .venv/bin/python scripts/render_clip.py mother --extractor mediapipe --no-save   # window only

Output (unless --no-save): data/rendered/{extractor}/{video_id}.mp4
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

from extract_landmarks import BUILDERS, hand_indices, hand_dropout_rate, skeleton_for

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifest.csv"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def rows_for_entry(entry_id):
    """manifest.csv rows for an entry_id (== id_gloss == ASL-LEX EntryID here)."""
    with open(MANIFEST, newline="") as f:
        return [r for r in csv.DictReader(f) if r["id_gloss"] == entry_id]


def pick_row(rows, video_id):
    if video_id:
        row = next((r for r in rows if r["video_id"] == video_id), None)
        if row is None:
            sys.exit(f"video_id {video_id!r} not found under this entry_id; "
                     f"pass --list to see what's available")
        return row
    # deterministic default: prefer a train-split clip (what the reference bank
    # actually uses), else just the first match
    return next((r for r in rows if r["split"] == "train"), rows[0])


def draw_overlay(canvas, row, extractor_name, frame_idx, n_frames):
    h, w = canvas.shape[:2]
    band = canvas.copy()
    cv2.rectangle(band, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.rectangle(band, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, f"{row['id_gloss']}  {row['asllex_code']}  ({extractor_name})",
                (12, 26), FONT, 0.7, (120, 255, 160), 2)
    cv2.putText(canvas, f"{row['video_id']}  signer={row['signer_id']}  split={row['split']}",
                (12, 48), FONT, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, f"frame {frame_idx + 1}/{n_frames}   [q]uit",
                (12, h - 9), FONT, 0.5, (200, 200, 200), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entry_id", help="ASL-LEX 2.0 EntryID, e.g. dog, what_1, mother")
    ap.add_argument("--extractor", default="mediapipe", choices=list(BUILDERS))
    ap.add_argument("--video-id", default=None, help="pick a specific clip (see --list)")
    ap.add_argument("--list", action="store_true", help="list every clip for this entry and exit")
    ap.add_argument("--out", default=None, help="output path (default data/rendered/{extractor}/{video_id}.mp4)")
    ap.add_argument("--no-save", action="store_true", help="show the window only, don't write a file")
    ap.add_argument("--threshold", type=float, default=None, help="draw confidence threshold")
    args = ap.parse_args()

    rows = rows_for_entry(args.entry_id)
    if not rows:
        sys.exit(f"no clips found for entry_id {args.entry_id!r} in {MANIFEST} "
                 f"(this is manifest.csv's id_gloss column -- the 60-sign curriculum only)")

    if args.list:
        print(f"{len(rows)} clip(s) for {args.entry_id} (asllex_code={rows[0]['asllex_code']}):")
        for r in rows:
            print(f"  {r['video_id']:<40} signer={r['signer_id']:<6} split={r['split']}")
        return

    row = pick_row(rows, args.video_id)
    video_path = REPO / row["video_path"]
    if not video_path.exists():
        sys.exit(f"source video missing: {video_path}")

    skeleton, _ = skeleton_for(args.extractor)
    hand_idx = hand_indices(args.extractor, skeleton)
    draw_kwargs = {} if args.threshold is None else {"threshold": args.threshold}

    print(f"clip: {row['video_id']}  ({row['id_gloss']} / {row['asllex_code']})  "
          f"signer={row['signer_id']} split={row['split']}")
    print(f"extractor: {args.extractor}  (built exactly as extract_landmarks.py does)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = None
    writer = None
    if not args.no_save:
        out_path = Path(args.out) if args.out else (
            REPO / "data" / "rendered" / args.extractor / f"{row['video_id']}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = BUILDERS[args.extractor]()
    scores_seen = []
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pose = extractor.extract(frame)  # RAW frame, mirrored=False -- matches the cache
            canvas = extractor.draw(frame, pose, **draw_kwargs) if pose is not None else frame.copy()
            if pose is not None:
                scores_seen.append(pose.scores)
            draw_overlay(canvas, row, args.extractor, frame_idx, n_frames)

            if writer is None and out_path is not None:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if writer is not None:
                writer.write(canvas)

            cv2.imshow(f"{row['id_gloss']} ({args.extractor})", canvas)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key in (ord("q"), 27):
                print("stopped early by user")
                break
            frame_idx += 1
    finally:
        extractor.close()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"processed {frame_idx}/{n_frames} frames")
    if scores_seen:
        import numpy as np
        dropout = hand_dropout_rate(np.stack(scores_seen), hand_idx)
        print(f"hand-dropout over processed frames: {dropout * 100:.1f}%")
    if out_path is not None and out_path.exists():
        print(f"saved: {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
