# ASL Adaptive Tutor

A from-scratch American Sign Language learning tool built on computer vision — an
adaptive tutor that watches you sign, diagnoses exactly where your form is off, and
generates personalized practice to close your gaps, live.

For the full phase-by-phase build plan, see **[`project_workflow.md`](project_workflow.md)**.
This README is the overview and quickstart.

## The one principle

**Grounded answer key, dynamic everything else.** The system has exactly one fixed,
sourced asset — a library of *correct sign references* from Deaf-created data. Every
other thing the learner experiences (diagnosis, sequencing, practice, feedback,
difficulty) is generated live and personalized. Like a real instructor: fully dynamic
in *how* they teach, fixed only on *what a correct sign is*. Grounding the answer key
is what makes the live feedback trustworthy rather than a confident hallucination.

## How it works

A closed loop over two grounded inputs — the reference library and your live attempt:

    assess attempt → diagnose parameters → update learner model
        → re-plan next task → generate task → present + coach → repeat

- **Perception** — a swappable `PoseExtractor` turns each frame into whole-body
  keypoints (body + hands + face).
- **Grading** — compares your attempt to the target's reference by distance and by
  *phonological parameters* (handshape, location, movement, orientation), so a
  malformed attempt yields a specific diagnosis instead of a wrong label.
- **Learner model + adaptive engine** — track per-sign and per-parameter mastery and
  choose what to drill next.
- **Task generator + feedback** — produce fresh drills and contrastive pairs, and
  coach you against the real Deaf-signer reference clip.

## Status

**Built:** the perception layer — `src/aslcv/pose/` — a swappable `PoseExtractor`
interface with two working backends (DWPose via rtmlib as default, MediaPipe Holistic
as alternate) over a shared `Pose` / `Skeleton` abstraction. Model weights are in
`models/`.

**Next (Phase 2):** the normalized feature layer, a cached-sequence dataset loader,
and a distance-based grader — the thin end-to-end slice. See the plan.

## Structure

Current:

    SignLanguageLearning/
    ├── models/                 # pose model weights (rtmlib .onnx + MediaPipe .task)
    ├── notebooks/
    │   ├── PoseExtraction1.py      # live DWPose (rtmlib) demo
    │   ├── PoseExtraction2.py      # live MediaPipe holistic demo
    │   └── extractorBenchmarking.py
    ├── src/aslcv/
    │   └── pose/                # DONE — the swappable extractor layer
    │       ├── base.py             #   PoseExtractor / Pose / Skeleton interface
    │       ├── coco_wholebody.py   #   COCO-WholeBody topology
    │       ├── dwpose.py           #   DWPose (rtmlib) — default
    │       └── mediapipe.py        #   MediaPipe holistic — alternate
    ├── data/raw/
    ├── project_workflow.md     # the detailed build plan
    └── pyproject.toml / uv.lock / .python-version

Target modules added per the plan: `features.py`, `dataset.py`, `grading/`,
`production/`, `learner/`, `generator/`, and `app/`.

**Design core:** `src/aslcv/pose/` is the reusable perception foundation — everything
downstream consumes `Pose` objects, so the extractor stays swappable and the feature
representation is defined in one place.

## Setup

Requires **Python 3.12** (rtmlib/MediaPipe don't ship wheels for 3.13+). The repo uses
[uv](https://docs.astral.sh/uv/):

```bash
uv sync          # creates the .venv from pyproject.toml + uv.lock and installs deps
```

Model weights live in `models/`:
- **DWPose / RTMW / YOLOX** (`.onnx`) auto-download via rtmlib the first time an
  extractor runs, and are cached in `models/`.
- **MediaPipe** landmarkers (`.task`) are downloaded from the official task pages
  (pose, face, hand landmarker) and placed in `models/`.

## Run

The live extractor demos work today:

```bash
# DWPose (default) — webcam → whole-body skeleton overlay
python notebooks/PoseExtraction1.py

# MediaPipe holistic — webcam → pose + face mesh + hands
python notebooks/PoseExtraction2.py
```

Both open your webcam and draw the detected skeleton; press `q` or `Esc` to quit.
Module entry points for the feature layer, grader, and app arrive with Phase 2 onward.

## Non-negotiable constraints

These follow from the one principle and hold throughout the build:

- **Grade by distance to a reference, not by classifying** — a learner's attempt is
  often not any valid sign, and a classifier would confidently mislabel it.
- **Retrieve reference video, never generate it** — the learner always imitates real
  Deaf-signer clips.
- **Grammar is a rule engine, not a trained model** — ASL's reorder/drop/non-manual
  rules are enumerable; encode them.
- **An LLM only ever touches English** (sentence prompts, feedback wording); it never
  authors or judges ASL.
- **Data comes from consented, Deaf-created sources** — ASL Citizen (isolated signs)
  and ASL-LEX (phonological features), not scraped datasets.
- **Deaf review gates any correction** shown to a user as authoritative.

## Scope

In scope: isolated-sign recognition, parameter-level feedback, adaptive practice
(v1); sentence prompts with rule-composed targets and continuous recognition (v2).
Out of scope by design: classifiers/spatial agreement/productive use of space,
free-form translation, and any synthesized sign video.
