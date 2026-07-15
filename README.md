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
- **Rule engine** — turns an English prompt into an ordered ASL gloss sequence + NMM
  tags, so practice sentences are composed from grounded references rather than
  invented.

## Data sources (three distinct roles)

| Source | Role | Used for |
|---|---|---|
| **ASL Citizen** | the *data* | training/eval videos + the reference clips shown to you |
| **ASL-LEX 2.0** | the *phonology* | per-sign parameter breakdown + ID-gloss keys |
| **ASL SignBank** | the *naming standard* | canonical ID-glosses so labels never drift |

Day to day only the first two are touched; ASL-LEX is already cross-referenced to
SignBank's ID-glosses, so SignBank sits behind the keys as the authority to consult,
not a third dataset to pull from.

## Status

**Built:** the perception layer — `src/aslcv/pose/` — a swappable `PoseExtractor`
interface with two working backends (DWPose via rtmlib as default, MediaPipe Holistic
as alternate) over a shared `Pose` / `Skeleton` abstraction, with `Skeleton.anchor()`
exposing named landmark indices so normalization stays topology-agnostic. Model
weights are in `models/`.

**Done:** Phase 0 — `curriculum.yaml` (60 starter signs, teaching order, contrastive
pairs, v2 constructions). ASL-LEX 2.0 downloaded to `data/ASL_LEX/`.

**In progress:** ASL Citizen download (blocks Phase 1 + reference retrieval);
the gloss rule engine in `src/aslcv/production/` (no data dependency).

**Next:** the Phase 2 vertical slice — normalized feature layer, cached-sequence
dataset loader, and a DTW distance grader, end to end on ~20 signs.

## Structure

Current:

    SignLanguageLearning/
    ├── curriculum.yaml         # Phase 0 — signs, order, contrastive pairs, constructions
    ├── models/                 # pose model weights (rtmlib .onnx + MediaPipe .task)
    ├── notebooks/
    │   ├── PoseExtraction1.py      # live DWPose (rtmlib) demo
    │   ├── PoseExtraction2.py      # live MediaPipe holistic demo
    │   └── extractorBenchmarking.py
    ├── src/aslcv/
    │   ├── pose/                # DONE — the swappable extractor layer
    │   │   ├── base.py             #   PoseExtractor / Pose / Skeleton interface
    │   │   ├── coco_wholebody.py   #   COCO-WholeBody topology
    │   │   ├── dwpose.py           #   DWPose (rtmlib) — default
    │   │   └── mediapipe.py        #   MediaPipe holistic — alternate
    │   └── production/          # in progress — gloss rule engine
    ├── data/
    │   ├── ASL_LEX/            # phonological features + ID-gloss keys
    │   └── raw/
    ├── project_workflow.md     # the detailed build plan
    └── pyproject.toml / uv.lock / .python-version

Target modules added per the plan: `features.py`, `dataset.py`, `grading/`,
`learner/`, `generator/`, and `app/`.

**Design core:** `src/aslcv/pose/` is the reusable perception foundation — everything
downstream consumes `Pose` objects, so the extractor stays swappable and the feature
representation is defined in one place.

## Model weights, decoded

The filenames in `models/` look interchangeable and aren't. `RTMPose` / `RTMW` /
`ViTPose` are **architectures**; `DWPose` is a **training method** (two-stage
distillation) — so `dw` in a filename announces the recipe, not the network.

- `rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288` → **DWPose** (RTMPose-large,
  DW-distilled). The current default.
- `rtmw-dw-x-l_simcc-cocktail14_270e-{256x192, 384x288}` → **RTMW-x** (newer
  architecture, also DW-distilled). The two files differ **only in input
  resolution** — 384×288 for detail, 256×192 for speed.
- `yolox_m_...` → the **person detector, required**. RTMPose/RTMW/DWPose are
  top-down: they only place keypoints inside a box handed to them.

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

Reference data:
- **ASL-LEX 2.0** — https://osf.io/zpha4/ → `data/ASL_LEX/`
- **ASL Citizen** — → `data/asl_citizen/`

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
  rules are enumerable; encode them as reviewable declarative data.
- **Fail closed** — the rule engine refuses out-of-scope input rather than emitting a
  wrong target. Every generated sentence is validated back through the engine.
- **An LLM only ever touches English** (sentence prompts, feedback wording); it never
  authors or judges ASL.
- **Data comes from consented, Deaf-created sources** — ASL Citizen and ASL-LEX, not
  scraped datasets.
- **Deaf review gates any correction** shown to a user as authoritative.

## Scope

In scope: isolated-sign recognition, parameter-level feedback, adaptive practice
(v1); sentence prompts with rule-composed targets and continuous recognition (v2).
Out of scope by design: classifiers/spatial agreement/productive use of space,
free-form translation, and any synthesized sign video.
