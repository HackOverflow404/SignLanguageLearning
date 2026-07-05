# ASL-CV

A from-scratch American Sign Language learning tool built on computer vision.
Camera → MediaPipe **HolisticLandmarker** (pose + hands + face) → landmark
sequences → a temporal model that recognizes signs → a practice/feedback loop.

This scaffold gets Phases 0–2 running (see the roadmap). The modeling and
learning-app phases are yours to build on top of the reusable landmark layer.

## Structure

```
asl-cv/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml                 # paths, feature toggles, which extractor to use
├── configs/                    # per-model hyperparameters (lstm.yaml, ...)   [7]
│
├── models/                     # DWPose .onnx weights — gitignored
│   └── checkpoints/            # your trained sign models — gitignored        [5]
│
├── data/
│   ├── raw/                    # your recorded videos (optional) — gitignored
│   ├── landmarks/              # extracted sequences, one folder per sign — gitignored
│   └── external/               # WLASL / MSASL downloads — gitignored         [8]
│
├── notebooks/
│   ├── 00_explore_landmarks.ipynb
│   └── colab_train.ipynb       # training on Colab's bigger GPU               [5,8]
│
├── scripts/
│   ├── record_signs.py         # collect your own clips, live                 [4]
│   ├── extract_landmarks.py    # batch: videos → landmark sequences           [8]
│   └── prepare_wlasl.py        # download + organize the public datasets      [8]
│
├── src/aslcv/
│   ├── __init__.py
│   ├── config.py
│   ├── pose/                   # the swappable extractor layer
│   │   ├── base.py             #   PoseExtractor interface
│   │   ├── dwpose.py           #   rtmlib/DWPose implementation (default)     [2]
│   │   └── mediapipe.py        #   alternate implementation
│   ├── features.py             # keypoints → normalized feature vector        [3]
│   ├── dataset.py              # load sequences, pad/crop, train/val split    [5]
│   ├── nets/
│   │   ├── lstm.py                                                            [5]
│   │   ├── transformer.py                                                     [7]
│   │   └── stgcn.py                                                           [7]
│   ├── train.py                # training loop                                [5]
│   ├── evaluate.py             # accuracy, confusion matrix, hand-dropout     [5]
│   ├── capture.py              # live skeleton visualizer                     [2]
│   └── recognize.py            # real-time sliding-window inference           [6]
│
├── app/                        # the learning tool                           [9]
│   ├── srs.py                  #   spaced-repetition scheduling
│   ├── session.py              #   prompt a sign, capture, score, record
│   └── progress.json           #   your review history — gitignored
│
└── tests/
    └── test_features.py        # check normalization is signer/scale invariant
```

The important design choice: **`landmarks.py` is the reusable core.** Every
other piece (visualizer, data collector, and later your trainer and real-time
recognizer) depends on it, so the feature representation is defined in exactly
one place.

## Setup

MediaPipe supports **Python 3.9–3.12 only** (not 3.13+). Create the venv on 3.12:

```bash
# with uv (recommended)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# or with stock tools, if `python3.12` is installed
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the holistic model bundle and drop it at `models/holistic_landmarker.task`.
Grab the current link from the official task page (URLs change, so don't guess):
https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker

## Run

```bash
# Phase 0 — watch your own landmarks (confirms the pipeline works)
PYTHONPATH=src python -m aslcv.capture

# Phase 2 — record 20 clips of a sign as landmark sequences
python scripts/record_signs.py --label hello --clips 20 --frames 40
```

Each recorded clip is saved as `data/landmarks/<label>/NNNN.npy` with shape
`(frames, feature_dim)`. With `face_mode: blendshapes`, `feature_dim` is 279
(99 pose + 63 + 63 hands + 2 presence flags + 52 blendshapes).

## Roadmap

- **Phase 0 — instrument.** `capture.py`. See the data, sanity-check detection. ✅ scaffolded
- **Phase 1 — static baseline.** Fingerspelling A–Z from single-frame hand
  landmarks with a simple classifier. Reuses `vectorize()` per frame.
- **Phase 2 — collect dynamic signs.** `record_signs.py`. Build a small
  vocabulary (start with ~10–20 signs). ✅ scaffolded
- **Phase 3 — sequence model.** Add `src/aslcv/dataset.py` (a torch `Dataset`
  that loads the `.npy` sequences, pads/crops to a fixed length) and
  `src/aslcv/nets/` (start with an LSTM, then a small Transformer). Add
  `train.py`. This is the real ML learning curve.
- **Phase 4 — real-time recognition.** A sliding window over live frames from
  `capture.py`, feeding the trained model.
- **Phase 5 — learning app + public data.** Spaced-repetition practice loop;
  bring in WLASL / MSASL for a larger vocabulary.

## Notes / gotchas

- MediaPipe's Tasks result nesting has shifted between versions. If you hit
  shape errors, print `result.pose_landmarks` once and adjust `_points()` /
  `_blendshapes()` in `landmarks.py` to match your install.
- Normalization is shoulder-anchored (origin = shoulder midpoint, scale =
  shoulder width), which is what makes sign *location* comparable across
  people and camera distances. If you sign partly off-frame, pose detection
  degrades — keep your upper body in view.
