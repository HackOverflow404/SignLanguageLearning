# ASL Adaptive Tutor — Build Plan

The definitive plan. Supersedes the older workflow (which described a
MediaPipe-holistic, record-your-own-signs, WLASL, classifier-into-SRS pipeline).

## The one principle everything hangs on

**Grounded answer key, dynamic everything else.** Exactly one fixed, sourced
asset exists in the system: a library of *correct sign references* downloaded
from Deaf-created data. Everything the learner experiences — diagnosis,
sequencing, practice items, feedback, difficulty, gap-filling — is generated
live and personalized. Same as how a real instructor works: fully dynamic in
*how* teaching happens, fixed only on *what a correct sign is*. Grounding the
answer key is what makes the live feedback trustworthy instead of a confident
hallucination.

Corollary rules to uphold:
- **Grade by distance to a reference, not by classifying.** A learner's attempt
  is often not any valid sign; an N-way classifier will confidently mislabel the
  attempt.
- **Retrieve reference video, never generate it.** Show real Deaf-signer clips.
- **Grammar is a rule engine, not a trained model.** ASL's reorder/drop/non-manual
  rules are enumerable; encode the ruleset.
- **An LLM only ever touches English** (prompt wording, sentence generation),
  never authoring or judging ASL.
- **Deaf review gates any correction shown as authoritative.**

## The architecture (a closed adaptive loop)

Two grounded inputs — the reference library and the learner's live attempt —
feed a loop:

    assess attempt → diagnose parameters → update learner model
        → re-plan next task → generate task → present + coach → repeat

Components:
- **Reference library** — ASL Citizen video + cached reference pose sequences +
  ASL-LEX phonological features. The answer key.
- **Perception** — the `PoseExtractor` + a normalized feature layer. (Built.)
- **Grading/diagnosis** — distance-to-reference + phonological feature heads +
  temporal alignment. Outputs per-parameter correctness, not a class label.
- **Learner model** — per-sign AND per-parameter mastery, updated every attempt.
- **Adaptive engine** — spaced repetition + gap-targeting + difficulty control.
- **Task generator** — drills, contrastive minimal pairs, (v2) sentence prompts.
- **Feedback presenter** — LLM-worded coaching + the real reference clip.
- **Rule engine (v2)** — English → ordered gloss + non-manual tags.

## Where things stand

Done: `src/aslcv/pose/{base,coco_wholebody,dwpose,mediapipe}.py` — the swappable
extractor layer, both backends working (DWPose via rtmlib as default, MediaPipe
Holistic as alternate), models downloaded locally. Satisfies Phase 2's
perception dependency.

The two small fixes flagged here are complete:
1. `Pose` construction is consistent — `DWPoseExtractor` now sets `width`/`height`
   like `MediaPipePoseExtractor`, so the feature layer can rely on both fields.
2. `Skeleton` carries named anchor indices via `anchor(name) -> index`, so
   normalization reads shoulder/hip/nose indices by meaning instead of hard-coding
   a topology. Both backends expose the same anchor vocabulary — nose, left/right
   shoulder, left/right hip (COCO-WholeBody shoulders = 5, 6; MediaPipe pose = 11, 12).

Phase 0 (curriculum scope) is complete — `curriculum.yaml` holds 60 starter
signs, teaching order, contrastive minimal pairs, and v2 constructions.

Reference data (Phase 1):
- **ASL-LEX 2.0** — downloaded and unzipped at `data/ASL_LEX/`.
- **ASL Citizen** — download in progress; not yet extracted. Blocks Phase 1 and Phase 5a.

Underway: Phase 5b's gloss rule engine (`src/aslcv/production/`), buildable now
against `curriculum.yaml` with no data dependency. Next after the data lands: the
Phase 2 vertical slice.

---

## Build order

Two tracks run in parallel and converge on the app. The recognition track
(Phases 1–4) and production track (Phase 5) are independent until Phase 6.

### Phase 0 — Curriculum scope (do FIRST; no code, no compute) — DONE

Vocabulary defines the label set, which defines which references are needed,
which defines everything downstream. Running last means building against a guess.

- Pick ~60 starter signs **present in ASL Citizen** (check the gloss list),
  chosen for beginner usefulness AND phonological diversity (varied handshapes and
  locations, so contrastive drills have material to work with).
- Draft a rough teaching order.
- List ~4 grammatical constructions for v2: topic-comment, yes/no question
  (brow-raise), wh-question, negation, time-first.
- **Produces:** `curriculum.yaml` (signs, order, constructions).
- **Done when:** a concrete sign list (all confirmed present in ASL Citizen) +
  ordering + construction list exist.

### Phase 1 — Reference library ingestion (recognition track)

- Download **ASL Citizen** (~84k videos, 2,731 signs, 52 consented Deaf/HH signers,
  everyday webcam footage — same domain as the tool) and **ASL-LEX 2.0**
  phonological features. Use ASL Citizen, not WLASL: WLASL is scraped, unconsented,
  from unknown signers; ASL Citizen is consented, IRB-approved, Deaf-involved, and
  domain-matched.
- Build **signer-independent** train/val/test splits — the same signer must never
  appear in two splits, or accuracy inflates and the tool fails on an unseen signer.
- Run the DWPose extractor over the reference videos once; cache pose sequences as
  `.npy`. A batch job — run on Colab or overnight on the 4070.
- Join each sign to the ASL-LEX feature vector (handshape [58 classes], major/minor
  location, path movement, flexion, etc.) → per-sign parameter labels for the
  phonological heads.
- **Produces:** cached `(pose_sequence, gloss, phonological_features, signer_id)`
  records + splits.
- **Done when:** for any curriculum sign the reference pose sequences and
  phonological labels load, with clean signer-independent splits.

### Phase 2 — Thin vertical slice (MILESTONE — before any benchmarking)

Build the crappy version that works end to end on ~20 signs with sensible defaults,
so integration problems surface now, not after two months of component polishing.

- **`features.py`** on the `Pose`/`Skeleton` abstraction: normalized per-frame
  vector — shoulder-anchored (origin = shoulder midpoint, scale = shoulder width,
  read anchor indices from the `Skeleton`), drop legs/feet, keep upper body + both
  hands (+ optional face), append per-keypoint confidence. Add the invariance
  self-test (translate + scale a synthetic pose → identical vector).
- **`dataset.py`**: load cached sequences, pad/crop to fixed length, expose the
  signer-independent split.
- **A minimal grader**: start with nearest-reference by DTW distance over the
  20 signs — no training required — to prove the loop. (A small LSTM classifier is
  an optional sanity check, but the real system is distance-based, so lead with the
  distance grader.)
- **A live loop**: webcam → extract → normalize → compare to references → show the
  closest sign + a crude distance score.
- **Produces:** a runnable end-to-end demo on 20 signs.
- **Done when:** signing one of the 20 signs into the webcam yields the closest sign
  name + a distance score, live. Defaults: DWPose balanced, shoulder normalization, DTW.

### Phase 3 — Benchmarking (now that a system exists to measure)

- **Cheap screen first (no labels):** FPS/latency on the 4070, hand jitter on a
  held-still clip, hand-dropout under occlusion — for each extractor candidate
  (DWPose-l, RTMW balanced/performance, MediaPipe Holistic, Sapiens-0.3B). Any
  candidate unable to hit real-time is disqualified as the *live* extractor, because
  training and serving must use the same extractor. The constraint resolves most
  of the choice. (Sapiens is SOTA — 308 keypoints, beats DWPose-L by ~7 AP — but a
  0.3–2B ViT at 1K resolution is unlikely to run real-time on a laptop 4070; screen
  on FPS before anything else.)
- **Accuracy comparison (uses Phase 2 pipeline):** extract ASL Citizen with each
  surviving extractor, train the same model, compare signer-independent accuracy.
- **Normalizer ablation (an afternoon, not a phase):** shoulder-width vs bounding
  box. Expect bbox to lose — the box grows when the hands raise, making
  normalization sign-dependent and corrupting the location signal.
- **Architectures:** LSTM → Transformer → ST-GCN; compare on the data.
- **Produces:** component decisions backed by numbers.
- **Done when:** extractor, normalizer, and architecture are chosen on measured
  evidence from the actual setup.

### Phase 4 — Grading & diagnosis engine (the real grader)

The instructor's eye. Not a classifier.

- A model over the pose sequence with multiple heads: a **sign embedding** (for
  nearest-reference distance / dictionary retrieval) plus **phonological feature
  heads** (handshape, location, movement, orientation) trained on the ASL-LEX labels
  from Phase 1.
- **DTW alignment** between the learner's attempt and the target reference sequence
  for timing/fidelity.
- Collect a small set of deliberately wrong attempts to calibrate the "how wrong,
  along which parameter" thresholds (the data task everyone skips).
- **Produces:** given `(attempt, target_sign)` → per-parameter correctness + overall
  fidelity + which parameters are off.
- **Done when:** a malformed attempt yields a specific diagnosis ("handshape right,
  location too low"), not a confident wrong label.

### Phase 5 — Production track (parallel to Phases 1–4)

- **5a Reference retrieval:** given a gloss, fetch the real Deaf-signer clip(s) and
  cached pose sequence. Serves both "show correct form" and "compose grading
  target." Retrieval, never generation.
- **5b Gloss rule engine:** English → ordered ASL gloss + non-manual tags. A written,
  inspectable ruleset (drop articles/copula, time-first, topic-comment reorder,
  brow-raise for yes/no, headshake for negation). Compose the target by concatenating
  reference pose sequences per gloss.
- **Scope boundary:** handles reorder/drop/non-manual tagging. Does NOT handle
  classifiers, spatial agreement, or productive use of space — keep out of scope.
- **Produces:** English sentence → (ordered real clips to show, concatenated
  reference pose target to grade against).
- **Done when:** a constrained sentence yields a correct gloss ordering + composed
  target, and the reference clips play in the right order.

### Phase 6 — The adaptive learning app (wire the loop)

- **Learner model:** per-sign + per-parameter mastery, updated every attempt. The
  per-parameter part is what generalizes ("forehead-location signs get missed across
  the board") to signs not yet drilled.
- **Adaptive engine:** spaced repetition (retention) + gap-targeting (weakest signs
  and parameters) + difficulty control. No fixed lesson order, no test bank.
- **Task generator:** targeted isolated drills; contrastive minimal-pair drills that
  fire automatically when two signs differing in one parameter get confused (the
  phonological data supplies the pairs); (v2) sentence prompts via an LLM
  constrained to the current vocabulary and weak items.
- **Feedback presenter:** an LLM phrases the diagnosis as natural coaching (English
  only); the app shows the real reference clip as the model of correct form.
- Wire the closed loop end to end.
- **Produces:** the working adaptive tutor for isolated signs (v1).
- **Done when:** the app chooses what to practice from the learner's gaps, drills,
  corrects at parameter level, and adapts across a session — nothing fixed except
  the reference correctness.

### Phase 7 — Stretch: sentences & continuous recognition (v2)

- **Continuous recognition:** segment a multi-sign sentence (CTC-style) to grade
  sequences, not just isolated signs. The genuinely hard perception step — stage
  only after v1 works.
- Combine with Phase 5's sentence prompts + rule-composed targets.

---

## Cross-cutting

- **Deaf review (ongoing gate):** the rule engine and grammar corrections are an
  *approximation* of a living language and will be confidently wrong at the edges.
  Before any grammar judgment is shown to a user as authoritative — certainly before
  release — get fluent Deaf review of the rules and targets. Build freely; don't
  ship grammar judgments as truth unreviewed.
- **Hardware split:** RTX 4070 for the live tool and everyday dev; Colab for batch
  extraction over ASL Citizen and long training runs.
- **Out of scope (by design, not failure):** classifiers, spatial agreement,
  free-form translation, and any synthesized/generated sign video.
- **Reality check:** ASL Citizen's own SOTA is ~63% top-1 and ~91% recall-at-10 on
  2,731 signs, unseen-signer. A ~60-sign vocabulary will do much better, but
  calibrate expectations: a useful practice aid with real error rates, not an
  oracle — exactly why the grader is distance/retrieval-based and why the learner
  always imitates real Deaf-signer video.

## File map (target)

    src/aslcv/
      pose/            # DONE — extractor layer
      features.py      # Phase 2 — normalized feature vector
      dataset.py       # Phase 2 — cached-sequence loader + splits
      grading/         # Phase 4 — embedding + phonological heads + DTW
      production/      # Phase 5 — reference retrieval + gloss rule engine
      learner/         # Phase 6 — learner model + adaptive engine
      generator/       # Phase 6 — drills, contrastive pairs, sentence prompts
    app/               # Phase 6 — the loop, feedback presenter (LLM English only)
    data/
      asl_citizen/     # Phase 1 — video + cached pose sequences + ASL-LEX features
