"""Phase 8: golden file for the phone's MediaPipe -> keypoint-layout mapping.

The phone runs MediaPipe's pose and hand landmarkers natively and gets back
normalized landmarks plus raw handedness labels; the Dart port then has to
place them in the combined MEDIAPIPE_HOLISTIC array exactly the way
MediaPipePoseExtractor does (pixel coordinates, pose visibility as score,
handedness score on every hand point, one hand per anatomical slot). This
writes, for a few real val frames, both sides of that mapping:

  input    -- the raw landmarker output, in the shape the iOS plugin returns
              (pose: 33 x (x, y, visibility); hands: label, score, 21 x (x, y))
  expected -- MediaPipePoseExtractor(mirrored=False).extract() on the same
              frame, face-mesh points zeroed (the phone runs no face model and
              the features never read them)

Frames come from unflipped ASL Citizen video, the orientation the extractor's
mirrored=False path and the phone's un-mirrored camera frames both use.

    .venv/bin/python scripts/export_layout_fixture.py
"""
import argparse
import base64
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from aslcv import dataset as dataset_mod
from aslcv.extractor.base import RunningMode
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC, MediaPipePoseExtractor

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "models" / "mobile" / "fixtures" / "mediapipe_layout.json"


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def raw_output(extractor, frame):
    """The landmarkers' own results for `frame`, packed like the iOS plugin's."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose, _, hands = extractor._detect_all(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    out = {"width": frame.shape[1], "height": frame.shape[0], "hands": []}
    if pose.pose_landmarks:
        out["pose"] = [v for lm in pose.pose_landmarks[0] for v in (lm.x, lm.y, lm.visibility)]
    for lms, handed in zip(hands.hand_landmarks, hands.handedness):
        out["hands"].append({"label": handed[0].category_name, "score": handed[0].score,
                             "points": [v for lm in lms for v in (lm.x, lm.y)]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-frames", type=int, default=6)
    args = ap.parse_args()

    # The face-mesh points only: the face REGION also holds six pose landmarks
    # (eye corners, mouth), which the phone's pose model does produce.
    face = np.array([i for i, n in enumerate(MEDIAPIPE_HOLISTIC.names) if n.startswith("face_")])
    # IMAGE mode: every detect call is independent, so calling the landmarkers
    # once for the raw output and again inside extract() sees the same result.
    extractor = MediaPipePoseExtractor(mirrored=False, running_mode=RunningMode.IMAGE)
    cases = []
    try:
        for row in dataset_mod.load_index("mediapipe", "val")[::7]:
            cap = cv2.VideoCapture(str(REPO / row["video_path"]))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            raw = raw_output(extractor, frame)
            # Prefer frames exercising both hand slots; take a one-hand frame
            # only once so the single-hand path is covered too.
            n_hands = len(raw["hands"])
            if "pose" not in raw or n_hands == 0:
                continue
            if n_hands == 1 and any(len(c["input"]["hands"]) == 1 for c in cases):
                continue
            pose = extractor.extract(frame)
            kp, sc = pose.keypoints.copy(), pose.scores.copy()
            kp[face], sc[face] = 0, 0
            cases.append({"video_id": row["video_id"], "id_gloss": row["id_gloss"], "input": raw,
                          "expected": {"keypoints": _b64(kp.reshape(-1)), "scores": _b64(sc)}})
            if len(cases) >= args.n_frames:
                break
    finally:
        extractor.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}))
    hands = [len(c["input"]["hands"]) for c in cases]
    print(f"wrote {len(cases)} frames (hands per frame: {hands}) -> {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
