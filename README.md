# ASL Adaptive Tutor

A from-scratch American Sign Language learning tool built on computer vision — an
adaptive tutor that watches you sign, diagnoses exactly where your form is off, and
generates personalized practice to close your gaps, live.

For the full phase-by-phase build plan and history, see
**[`project_workflow.md`](project_workflow.md)**. This README is the overview and
quickstart.

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

- **Perception** (`extractor/`) — a swappable extractor turns each frame into
  whole-body keypoints. MediaPipe Holistic is the picked default (fastest, steadiest,
  and most accurate of the four backends benchmarked); DWPose/RTMW/ViTPose (via
  rtmlib) remain for comparison.
- **Normalization + features** (`normalizer/`, `features.py`) — puts keypoints into a
  shoulder-centered frame (plus per-hand local frames for handshape), then assembles,
  standardizes, and optionally trims them to the motion-active span.
- **Grading** (`grading/`) — two graders: a training-free DTW nearest-reference
  baseline, and a metric-learned embedding grader (`PoseGraderNet`) with five
  *disjoint* phonological heads (handshape, major/minor location, movement,
  repetition) — so a malformed attempt gets a specific per-parameter diagnosis, never
  a confident wrong label. Grading is always **distance to the one known target**,
  never open-set classification.
- **Live capture** (`capture.py`) — a motion-aware buffer that grows from the start of
  signing to a natural rest boundary instead of a fixed sliding window, so a slower
  signer's attempt isn't silently truncated.
- **Sentence grading** (`grading/alignment.py`) — a full sentence is forced-aligned
  (DTW) against a reference built from the exact words you were prompted with, then
  each word is graded independently. The target is always known in advance, so this
  is forced alignment, not open continuous recognition.
- **Learner model + adaptive engine** (`learner/`) — SM2-style spaced repetition over
  per-sign, per-parameter mastery, gap-targeting, and automatic contrastive
  minimal-pair drills (stress-testing signs you've nearly mastered against their
  closest confusable neighbor).
- **Task generator + feedback** (`generator/`) — grounded, always-on descriptions of
  target sign phonology; templated or (optional, hosted-API) LLM-phrased coaching;
  LLM-written example sentences, validated back through the rule engine before ever
  being shown.
- **Rule engine + retrieval** (`production/`) — turns an English prompt into an
  ordered ASL gloss sequence + non-manual-marker tags (fail-closed: refuses
  out-of-scope input rather than emit a wrong target), then retrieves and stitches
  the real reference clips for it. Never generates signing video.

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

**Phases 0–7: done.** Curriculum (60 signs) → cached pose extraction across 4
backends → normalized feature pipeline + DTW baseline → extractor benchmarking
(MediaPipe picked) → the learned embedding grader with disjoint phonological heads →
the gloss rule engine + reference retrieval → the full adaptive learning loop
(spaced repetition, contrastive drills, coaching, optional LLM phrasing) → sentence
grading via forced alignment. All of it is wired live in `scripts/diagnose_demo.py`.

**Recently**: a real train/serve distribution mismatch was found and fixed — the
grader was trained on full raw clips (much more rest padding than live capture ever
keeps) but graded live-captured attempts, which measurably biased the
`repeated_movement` head toward false positives. Retrained on live-shaped clip
framing instead; false positives on that parameter roughly halved. See
`PHASE4_REPORT.md` and `project_workflow.md`'s Phase 4 section for the full
investigation.

**Also done, kept separate from the product:** an isolated research comparison
against CTC-based continuous sign recognition (`src/aslcv/research/`, never imported
by the live demo) — confirming this project's distance-based, always-know-the-target
approach is the right fit for tutoring, while being honest about where an
open-vocabulary CTC model genuinely wins instead. See `CTC_VS_ALIGNMENT_REPORT.md`.

**Open:** fluent Deaf review of the gloss rule engine's output (a human bottleneck,
not a coding task — see `scripts/export_review_sheet.py`); sentence-mode capture
timing has only been tuned against synthetic energy patterns, not a real camera yet;
Phase 8 (porting to a phone) and Phase 9 (expanding past 60 signs) are both scoped in
`project_workflow.md` but not started.

## Structure

    SignLanguageLearning/
    ├── curriculum.yaml         # Phase 0 — 60 signs, teaching order, contrastive pairs
    ├── models/                 # pose model weights + trained checkpoints (gitignored)
    ├── data/
    │   ├── ASL_Citizen/            # reference videos
    │   ├── ASL_LEX/                # phonological features + ID-gloss keys
    │   ├── cache/{extractor}/      # cached raw keypoints per clip, per backend
    │   ├── manifest.csv            # clip -> sign -> split -> asllex_code join table
    │   └── phonology.csv           # per-sign parameter labels
    ├── src/aslcv/
    │   ├── extractor/           # DONE — 4 swappable backends (mediapipe default)
    │   ├── normalizer/          # DONE — shoulder-frame + per-hand local normalization
    │   ├── features.py          # feature assembly, standardization, motion trimming
    │   ├── capture.py           # motion-aware live capture boundary detection
    │   ├── pipeline_config.py   # the one place a feature pipeline is built from CLI args
    │   ├── dataset.py           # cached-sequence loader + splits + label encoding
    │   ├── grading/             # DTW baseline + the learned embedding grader
    │   ├── learner/             # per-sign/per-parameter mastery + spaced-rep scheduler
    │   ├── generator/           # coaching text, sign descriptions, LLM-phrased upgrades
    │   ├── production/          # gloss rule engine + reference video/feature retrieval
    │   └── research/            # isolated CTC-CSLR comparison, NOT part of the product
    ├── scripts/
    │   ├── diagnose_demo.py         # THE live demo — single-sign and --sentence modes
    │   ├── live_demo.py             # Phase 2 DTW baseline demo (research/ablation tool)
    │   ├── train_embedding_grader.py / eval_embedding_grader.py
    │   ├── train_ctc_cslr.py / eval_ctc_vs_alignment.py / eval_forced_alignment.py
    │   └── gloss_repl.py / export_review_sheet.py / compose_sentence.py / ...
    ├── tests/                   # `make test` / `uv run pytest tests/` — one command, everything
    ├── project_workflow.md      # the detailed, phase-by-phase build plan + history
    └── pyproject.toml / uv.lock / .python-version

## Model weights, decoded

The filenames in `models/` look interchangeable and aren't. `RTMPose` / `RTMW` /
`ViTPose` are **architectures**; `DWPose` is a **training method** (two-stage
distillation) — so `dw` in a filename announces the recipe, not the network.

- `pose_landmarker_full.task` / `hand_landmarker.task` / `face_landmarker.task` →
  **MediaPipe Holistic**, the picked default: fastest (with the GPU delegate),
  steadiest, and the most accurate on minimal-pair separation of the four backends
  benchmarked (see `project_workflow.md`'s Phase 3 section).
- `rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288` → **DWPose** (RTMPose-large,
  DW-distilled).
- `rtmw-dw-x-l_simcc-cocktail14_270e-{256x192, 384x288}` → **RTMW-x** (newer
  architecture, also DW-distilled). The two files differ **only in input
  resolution** — 384×288 for detail, 256×192 for speed.
- `vitpose-l-wholebody.onnx` → **ViTPose-large**.
- `yolox_m_...` → the **person detector**, required by every rtmlib backend
  (RTMPose/RTMW/ViTPose are top-down: they only place keypoints inside a box handed
  to them). MediaPipe doesn't need it.

## Setup

Requires **Python 3.12** (rtmlib/MediaPipe don't ship wheels for 3.13+). The repo uses
[uv](https://docs.astral.sh/uv/):

```bash
uv sync          # creates the .venv from pyproject.toml + uv.lock and installs deps
```

`torch` is installed but not tracked by `uv` (a real, documented dependency-resolver
conflict — see `project_workflow.md`'s known issues) — avoid casual `uv add`/`uv
remove`/`uv sync` once your environment is working, or reinstall it manually
(`uv pip install torch==2.13.0`) if it disappears.

Model weights live in `models/`:
- **DWPose / RTMW / ViTPose / YOLOX** (`.onnx`) auto-download via rtmlib the first
  time that extractor runs, and are cached in `models/`.
- **MediaPipe** landmarkers (`.task`) are downloaded from the official task pages
  (pose, face, hand landmarker) and placed in `models/`.
- **Trained checkpoints** (`models/embedding_grader/`, `models/ctc_cslr/`) are
  produced by `scripts/train_embedding_grader.py` / `scripts/train_ctc_cslr.py` —
  gitignored, not shipped in the repo.

Reference data:
- **ASL-LEX 2.0** — https://osf.io/zpha4/ → `data/ASL_LEX/`
- **ASL Citizen** → `data/ASL_Citizen/`

Optional: an HF_TOKEN in a repo-root `.env` (see `.env.example`) enables
`--llm-feedback`/`--sentence-prompts`' hosted-API upgrades — both fail open to
templated/no output with no token set.

## Run

The actual product:

```bash
# Single-sign adaptive practice, GPU MediaPipe (recommended if you have one)
.venv/bin/python scripts/diagnose_demo.py --gpu

# Practice a specific sign, or a cycle of them -- [n] still adapts onward from there
.venv/bin/python scripts/diagnose_demo.py --gpu --target father
.venv/bin/python scripts/diagnose_demo.py --gpu --targets you,me,water

# Sentence mode -- forced-alignment grading of a whole sentence, not one sign
.venv/bin/python scripts/diagnose_demo.py --gpu --sentence "I want water."

# No camera: verify the whole grading path offline against cached clips
.venv/bin/python scripts/diagnose_demo.py --selftest
.venv/bin/python scripts/diagnose_demo.py --selftest --sentence "I want water."

# Optional LLM upgrades (needs HF_TOKEN, fails open without one)
.venv/bin/python scripts/diagnose_demo.py --gpu --llm-feedback --sentence-prompts
```

In single-sign mode: `[c]` clears the capture and re-tries the same target; `[n]`
records the attempt and adaptively picks the next one. In sentence mode: `[c]` clears
and retries; `[n]` grades the completed attempt once the capture badge shows
CAPTURED.

Tests: `make test` (== `uv run pytest tests/`), or the faster direct invocation
`.venv/bin/python -m pytest tests/` if `uv run`'s dependency resync feels slow.

## Non-negotiable constraints

These follow from the one principle and hold throughout the build:

- **Grade by distance to a reference, not by classifying** — a learner's attempt is
  often not any valid sign, and a classifier would confidently mislabel it.
- **Retrieve reference video, never generate it** — the learner always imitates real
  Deaf-signer clips.
- **Grammar is a rule engine, not a trained model** — ASL's reorder/drop/non-manual
  rules are enumerable; encoded as reviewable declarative data.
- **Fail closed** — the rule engine refuses out-of-scope input rather than emitting a
  wrong target. Every generated sentence is validated back through the engine.
- **An LLM only ever touches English** (sentence prompts, feedback wording); it never
  authors or judges ASL.
- **Data comes from consented, Deaf-created sources** — ASL Citizen and ASL-LEX, not
  scraped datasets.
- **Deaf review gates any correction** shown to a user as authoritative.

## Scope

In scope and built: isolated-sign recognition, parameter-level feedback, adaptive
practice with spaced repetition and contrastive drills, LLM-written example
sentences validated through the rule engine, and sentence grading via forced
alignment against a known target.

Out of scope by design, not gap: open-vocabulary classifiers, spatial
agreement/productive use of space, free-form translation, any synthesized/generated
sign video, and continuous recognition of an *unknown* sequence (this project always
knows the target it prompted — see `src/aslcv/research/` for why an open-vocabulary
CTC approach was evaluated and deliberately not adopted).
