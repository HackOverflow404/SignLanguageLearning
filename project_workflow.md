# ASL Adaptive Tutor — Build Plan

The definitive plan. Supersedes the older workflow (which described a
MediaPipe-holistic, record-your-own-signs, WLASL, classifier-into-SRS pipeline).

> **Rev note (this pass):** synced Phase 3 to reality (all four extractors built and
> cached, MediaPipe is the default — earlier text still called DWPose the default and
> RTMW/ViTPose "URL swaps to do"). Added the full Phase 2 normalization/feature design
> (two files: `ShoulderNormalizer` with global + local-hand frames, and a `features.py`
> assembly pipeline; BBox dropped as unsound rather than built). Hardened Phase 4
> (embedding+distance not classification; multi-morphemic signs excluded from
> parameter feedback). One open reconciliation: this pass's manifest count is 1,874
> clips (895/229/750) — authoritative from the built pipeline — vs an earlier ~1,706
> estimate made before the ambiguous variants (dog/baby/drink/how/milk) were resolved.

## Terminology

- **Gloss / ID-gloss** — the canonical written label for a sign (STORE, YESTERDAY).
  An *ID-gloss* is the standardized one, so the same sign always gets the same label
  everywhere in the system. ASL SignBank is the naming authority; ASL-LEX 2.0 is
  cross-referenced to it.
- **NMM — non-manual markers** — grammar carried by anything other than the hands:
  brow-raise (yes/no question), brow-furrow (wh-question), headshake (negation),
  head tilt (topic marking).
- **Phonological parameters** — the parts a sign decomposes into: handshape,
  selected fingers, flexion, major/minor location, movement, orientation.

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
- **Fail closed.** The rule engine refuses to emit a target it isn't confident in,
  rather than emitting a wrong one. A wrong target teaches the learner wrong.
- **An LLM only ever touches English** (prompt wording, sentence generation),
  never authoring or judging ASL.
- **Deaf review gates any correction shown as authoritative.**

## The three data sources (distinct roles — not three of the same thing)

| Source | Role | What it gives |
|---|---|---|
| **ASL Citizen** | the *data* | ~84k videos of real Deaf signers; training/eval data + the reference clips shown to the learner |
| **ASL-LEX 2.0** | the *phonology* | per-sign parameter breakdown (handshape, selected fingers, flexion, major/minor location, movement) + ID-gloss keys |
| **ASL SignBank** | the *naming standard* | canonical ID-glosses so labels never drift |

Day to day the work touches two: **ASL Citizen for videos/data, ASL-LEX 2.0 for
phonological features and gloss keys.** SignBank sits behind those keys (ASL-LEX is
already cross-referenced to it) — consult it to resolve or verify a gloss, not as a
third dataset to pull from.

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
- **Rule engine** — English → ordered gloss + NMM tags.

## Where things stand

**Extractor layer** (`src/aslcv/extractor/`, renamed from `pose/`) — DONE. Four
swappable backends behind one `Extractor`/`Skeleton`/`Pose` interface:
**MediaPipe Holistic** (default; 553 kpts + 52 face blendshapes for NMM), **DWPose**,
**RTMW-x**, and **ViTPose-l** (the latter three share `rtmlib_base.py`, all
COCO-WholeBody 133). Models local. `Pose` carries width/height + an optional
blendshapes channel; `Skeleton` exposes named anchors (nose, left/right
shoulder/hip) so normalization reads indices by meaning, not by hard-coded topology
(COCO-WholeBody shoulders = 5, 6; MediaPipe pose = 11, 12).

**Phase 0** (curriculum scope) — DONE. `curriculum.yaml`: 60 signs + teaching order
+ contrastive pairs + v2 constructions, each resolved to a stable `asllex_code` and
ASL Citizen gloss.

**Phase 1** (reference library) — DONE; the done-when test (`tests/test_dataset.py`)
passes:
- ASL-LEX 2.0 and ASL Citizen (~84k videos) downloaded under `data/`.
- `data/manifest.csv` (`tools/build_manifest.py`): curriculum filtered onto ASL
  Citizen's official signer-independent splits — 1,874 clips, 895/229/750
  train/val/test, zero signer overlap.
- All four extractors run over the manifest and cached as raw per-clip `.npz` in
  `data/cache/{extractor}/` (`scripts/extract_landmarks.py`), integrity-verified
  after several mid-run crashes (`scripts/verify_cache.py`).
- `data/phonology.csv` (`tools/join_phonology.py`): every ASL-LEX phonology +
  frequency parameter joined onto the 60 signs by `asllex_code`; 6 multi-morphemic
  signs flagged (ASL-LEX codes only their first morpheme).
- `src/aslcv/dataset.py`: yields `(pose_sequence, id_gloss, phonological_features,
  signer_id)` per split, plus a lazy-torch fixed-length Dataset + label encoder.

**Phase 5b** (production track) — gloss rule engine built
(`src/aslcv/production/gloss_rules.py`) with its regression test; no data dependency.

**Next: Phase 2** — the thin vertical slice, on the 20-sign subset below:
`normalizer/shoulder.py` (shoulder-global + optional local-hand frames) and
`features.py` (selection → normalize → confidence → concat → velocity → stack →
standardize) → DTW nearest-reference grader → live webcam demo. The mother/father
minimal pair is the first real test. Requires installing **torch** (the loader is
torch-free; training + the Dataset wrapper are not). Only `ShoulderNormalizer` is
being built — BBox is skipped as unsound, not stubbed (see Phase 2/3).

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
  phonological features (OSF: https://osf.io/zpha4/). Use ASL Citizen, not WLASL:
  WLASL is scraped, unconsented, from unknown signers; ASL Citizen is consented,
  IRB-approved, Deaf-involved, and domain-matched.
- Build **signer-independent** train/val/test splits — the same signer must never
  appear in two splits, or accuracy inflates and the tool fails on an unseen signer.
- Run the pose extractor over the reference videos once; cache raw pose sequences as
  `.npz` (MediaPipe is the default; also extract with each Phase 3 candidate so the
  head-to-head has caches ready). A batch job — run on Colab or overnight on the 4070.
- Join each sign to the ASL-LEX feature vector (handshape [58 classes], major/minor
  location, path movement, flexion, etc.) → per-sign parameter labels for the
  phonological heads.
- **Key on ID-glosses** from ASL-LEX (cross-referenced to ASL SignBank), so the
  recognition track, the rule engine's lexicon, and `curriculum.yaml` all name signs
  identically. This join is the seam where label drift would otherwise creep in.
- **Produces:** cached `(pose_sequence, id_gloss, phonological_features, signer_id)`
  records + splits.
- **Done when:** for any curriculum sign the reference pose sequences and
  phonological labels load, with clean signer-independent splits.

### Phase 2 — Thin vertical slice (MILESTONE — before any benchmarking)

Build the crappy version that works end to end on ~20 signs with sensible defaults,
so integration problems surface now, not after two months of component polishing.

**The 20-sign slice** — chosen for phonological spread (all 5 major locations,
~14 distinct handshapes, varied movement), all single-morpheme (the 6
multi-morphemic signs are excluded so parameter-level checks stay clean), and each
has ≥13 training clips. The canonical list lives in `curriculum.yaml` as
`phase2_slice` (checked by `tools/validate_curriculum.py`); this table mirrors it:

| Major location | Signs (handshape) |
|---|---|
| Neutral | you (1), what_1 (5), yes (s), dog (d), green (g) |
| Body    | me (1), please (open_b), tired (c), fine_1 (5) |
| Hand    | name (h), read (v), coffee (s) |
| Head    | who (l), why (y), red (1), eat_1 (o), water (w), **mother (5)**, **father (5)** |
| Arm     | time (bent_1) |

`mother`/`father` are a built-in minimal pair (same handshape + movement, differ
only in minor location — forehead vs chin): the first test of whether the features
+ DTW grader catch a single-parameter difference.

**`normalizer/` and `features.py` are two responsibilities, kept as two files.**
Normalization is the *swappable* strategy the Phase 3 grid ablates; feature
assembly is stable across the grid. Split them so swapping a normalizer is a
one-line injection and the assembly code never moves (same precedent as the
swappable `extractor/` backends).

- **`normalizer/shoulder.py`** — a `ShoulderNormalizer` strategy taking a `Pose` +
  `Skeleton`. Two reference frames, because a sign is two questions at once
  (*where* is the hand vs *what shape* is the hand), and they want different frames:
  - **global block** — origin = shoulder midpoint, scale = shoulder width, applied
    to body + arms + hand *positions*. Carries location and movement. (Anchor
    indices read from `Skeleton.anchor()`, so it's topology-agnostic across the four
    extractors.)
  - **local-hand block** (toggleable) — each hand re-normalized in its own frame
    (origin = wrist, scale = wrist-to-middle-knuckle distance, *not* a hand bbox),
    so handshape is described at full resolution independent of where the hand is.
  - Translation- and scale-invariant, deliberately **not** rotation-invariant — a
    tilted head is grammatical and must be preserved.
- **`features.py`** — the assembly pipeline over the normalized blocks:
  1. **keypoint selection** — keep both hands + upper-body pose (+ face only if NMM
     is on); drop legs/feet. A modeling choice tuned here, never re-extracted.
  2. **normalization** — call the injected normalizer per block.
  3. **confidence handling** — zero/flag points below `conf_thr`; append the
     confidence as an extra channel; a missing hand → zero block + presence flag.
     (Note: DWPose gives graded confidence; MediaPipe hands/face are hard-coded 1.0,
     so this channel is continuous for one backend and binary for the other —
     account for it in the Phase 3 comparison.)
  4. **block concatenation** — `[ body | L-hand | R-hand | face? ]` into one vector,
     preserving slice boundaries so the Phase 4 handshape head reads the hand slice
     and the location head reads the body slice.
  5. **velocity deltas** (toggleable, near-default) — append per-keypoint change
     from the previous frame; movement is a phonological parameter, so hand it to
     the model explicitly.
  6. **sequence stacking** — pile per-frame vectors into `(T, F)`; pad/crop (or
     resample) to a fixed length. DTW is exempt — it handles variable length.
  7. **standardization** (toggleable) — per-feature mean/std computed on **train
     only**, applied to val/test (never fit on val/test — that leaks).
  - Invariance self-test: translate + scale a synthetic pose → identical global
    block; the local-hand block is unchanged by where the hand sits in frame.
- **A minimal grader**: nearest-reference by DTW distance over the 20 signs — no
  training required — to prove the loop. (A small LSTM classifier is an optional
  sanity check, but the real system is distance-based, so lead with the distance
  grader.)
- **A live loop**: webcam → extract → normalize → assemble features → compare to
  references → show the closest sign + a crude distance score.
- **Produces:** a runnable end-to-end demo on 20 signs.
- **Done when:** signing one of the 20 signs into the webcam yields the closest sign
  name + a distance score, live, and the mother/father minimal pair is
  distinguished. Defaults: MediaPipe, shoulder-global + local-hand normalization, DTW.

*(`dataset.py` is already built in Phase 1 — the loader, fixed-length wrapper, and
label encoder exist; Phase 2 consumes it rather than rebuilding it.)*

### Phase 3 — Benchmarking (now that a system exists to measure)

**The filter that decides most of this:** training and serving must use the *same*
extractor, so anything that can't run real-time on the 4070 is disqualified as the
live extractor regardless of its accuracy.

**Candidates — four, all built.** MediaPipe Holistic (current default), DWPose-l,
RTMW-x, and ViTPose-l are all wired behind the `Extractor` interface; the three
rtmlib backends share `rtmlib_base.py` and the same COCO-WholeBody 133 topology,
so nothing downstream changes when swapping among them. All four are already
cached over the manifest (Phase 1), so the head-to-head is ready to run:

| Candidate | Model | Notes |
|---|---|---|
| **MediaPipe Holistic** (current default) | 553 kpts + blendshapes | the *face-detail* / NMM contender |
| **DWPose-l** | `rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288` | RTMPose-large + DW distillation |
| **RTMW-x** | `rtmw-dw-x-l_simcc-cocktail14_270e-384x288` | newer arch; use 384×288 for hand detail |
| **ViTPose-l** | rtmlib ViTPose wholebody (133) | accuracy leader in the family |

All four are built and cached; the head-to-head is a matter of running the screens
below, not new integration.

**Dropped:** Sapiens, AlphaPose/Halpe, SDPose, SMPLest-X. All are separate
integrations that won't hit real-time on a laptop 4070. Sapiens specifically
collapses: its 243-facial-keypoint advantage only exists in its native 308-keypoint
Goliath topology (which would force a feature-layer rewrite), while its
COCO-WholeBody output mode gives the same 68 face points as DWPose — so the variant
that could run live loses the reason to pick it. If facial grammar later proves to
be a measured bottleneck, the answer is **MediaPipe** (already integrated,
semantically labeled blendshapes), not Sapiens.

**Model naming, decoded** (these look interchangeable and aren't):
- `RTMPose` / `RTMW` / `ViTPose` = **architectures**. `DWPose` = a **training
  method** (two-stage distillation), not a network. So `dw` in a filename announces
  the training recipe, not the architecture.
- `rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288` = RTMPose-large + DW distillation
  → **this is "DWPose"**, the 2023 classic. Currently wired up.
- `rtmw-dw-x-l_simcc-cocktail14_270e-{256x192, 384x288}` = **RTMW-x** (newer arch,
  FPN + Hierarchical Encoding Module built so hands/face aren't drowned out by the
  torso) + DW distillation, cocktail14 data. **The two files differ only in input
  resolution** — 384×288 = more pixels on the hands, slower; 256×192 = faster,
  coarser. Use 384×288 as the accuracy contender.
- Filename grammar: architecture + method + size + head (`simcc`) + training data +
  epochs + input resolution.
- `yolox_m_...` is the **person detector, and it is required** — RTMPose/RTMW/DWPose
  are top-down estimators that only place keypoints inside a box handed to them.
  All rtmlib configs share this one file.

**Cheap screen first (no labels, ~a day):** FPS/latency on the 4070, hand jitter on
a held-still clip, hand-dropout under occlusion. Run on the same clips for all four,
including hands-crossing and hands-near-frame-edge. Real-time failures are out.

**Accuracy comparison (needs Phase 2 + labels — do not run early):** extract ASL
Citizen with each survivor, train the same model, compare signer-independent accuracy.

**Normalizer ablation (an afternoon, not a phase):** the real grid is a small set
of toggles on the shoulder-anchored scheme (see Phase 2's normalization design):
local-hand block on/off, velocity deltas on/off, train-set standardization on/off.
BBox normalization is *not* a serious contender — the box grows when the hands
raise, making the scale sign-dependent and corrupting the location signal — so it's
skipped rather than built. Expected winner: shoulder-global + local-hand + velocity
+ standardization, since 40 of the 64 minimal pairs are handshape pairs that the
local-hand block sharpens.

**Architectures:** LSTM → Transformer → ST-GCN; compare on the data.

- **Produces:** component decisions backed by numbers.
- **Done when:** extractor, normalizer, and architecture are chosen on measured
  evidence from the actual setup.

> Perspective: among the good whole-body models the accuracy gaps are small next to
> everything else in this project. Run the cheap screen, pick a survivor, move on.
> Don't let extractor selection balloon.

### Phase 4 — Grading & diagnosis engine (the real grader)

The instructor's eye. Not a classifier.

- **Embedding + distance, never argmax classification.** A learner's attempt is
  often not any valid sign, and an N-way classifier would confidently mislabel it.
  So the model learns a **sign embedding** graded by distance to the target's
  reference (the same framing ASL Citizen itself uses — dictionary retrieval,
  recall@k), which degrades gracefully on malformed input.
- **Phonological feature heads** (handshape, location, movement, orientation)
  trained on the ASL-LEX labels from Phase 1, so feedback degrades *informatively*
  ("handshape right, location wrong"). Read the block slices from `features.py`
  accordingly — handshape head off the local-hand slice, location head off the
  global slice.
- **DTW alignment** between the learner's attempt and the target reference sequence
  for timing/fidelity.
- Collect a small set of deliberately wrong attempts to calibrate the "how wrong,
  along which parameter" thresholds (the data task everyone skips).
- **Exclude the 6 multi-morphemic signs from parameter-level feedback** (or annotate
  their full form): ASL-LEX codes only their first morpheme, so a parameter head
  would diagnose the whole sign against a partial label. `join_phonology.py` already
  flags them.
- **Produces:** given `(attempt, target_sign)` → per-parameter correctness + overall
  fidelity + which parameters are off.
- **Done when:** a malformed attempt yields a specific diagnosis ("handshape right,
  location too low"), not a confident wrong label.

### Phase 5 — Production track (parallel to Phases 1–4)

**5a Reference retrieval:** given an ID-gloss, fetch the real Deaf-signer clip(s)
and cached pose sequence. Serves both "show correct form" and "compose grading
target." Retrieval, never generation. *(Blocked on the ASL Citizen download.)*

#### 5b — Gloss rule engine (underway; no data dependency)

**Don't start from a blank file.** Adapt Moryossef et al.'s open-source
text-to-gloss-to-pose-to-video baseline (arXiv 2305.17714): spaCy for lemmatization,
POS tagging and dependency parsing, then a rule-based word reordering-and-dropping
component. They compared rule-based against neural MT and **chose the rules** — for
the same reasons that matter here: transparency and low-resource robustness. It
targets other signed languages, but the framework and rule structure port; supply
ASL rules and an ASL lexicon.

**Why rules, not the alternatives:** neural MT has no real English↔ASL-gloss parallel
corpus (and the main one, ASLG-PC12, was itself rule-generated — training on it just
re-learns the rules while adding a black box). A prompted LLM is non-deterministic,
unverifiable, can't reliably refuse, and has weak ASL competence, so it fabricates.
Rules win on the four criteria that actually matter: verifiability, fail-closed
behavior, zero parallel-data requirement, and determinism.

**Three cleanly separated layers behind one interface** (this is the future-proofing
— treat it as non-negotiable):

1. **English analysis** — spaCy (POS, dependency parse, lemmatization). Standard,
   maintained, replaceable.
2. **The ruleset** — drop / reorder / NMM-tagging expressed as *declarative data or
   patterns over the parse tree*, **not** logic buried in Python. This is the layer a
   Deaf reviewer or linguist reads and edits without touching code.
3. **Lexicon mapping** — English lemma → ASL **ID-gloss**, keyed to ASL-LEX /
   SignBank naming and to what exists in ASL Citizen. A versioned table.

Wrap all three behind one interface:

    english → GlossSequence(glosses, nmm_tags, in_scope, confidence)

Today rules implement it. Later a learned model (e.g. a dependency-syntax-aware
transformer, which has beaten plain rules on English→ASL gloss) can implement the
*same* interface with nothing downstream changing.

**Fail closed — the reliability principle that matters most.** The engine must
detect when it's out of its depth and **refuse to emit a target** rather than emit a
confidently wrong one. Out-of-scope triggers: any word with no gloss in the lexicon;
any construction not matching a supported rule; anything needing classifiers or
spatial agreement. On those, return `in_scope=False` and produce nothing. Then close
the loop: since generation is controlled, constrain the LLM to supported constructions
and vocabulary, and **run every generated sentence back through the engine as a
validator** — out-of-scope → discard and regenerate. In production the engine then
never faces input it can't handle. The out-of-scope detector is a first-class
component, not an afterthought.

**Golden test corpus.** A regression set of `(English → expected gloss + NMM)` pairs,
one or more per rule/construction, ideally reviewed by fluent signers. Adding a rule
must not silently break another. Version the rules and the lexicon; both evolve under
review.

**In scope:** declaratives, yes/no questions (brow-raise), wh-questions
(brow-furrow), negation (headshake), time-fronting, basic topic-comment, function-word
dropping. Moryossef's concrete dropping rule is the starting point — drop non-content
words (articles, prepositions), keep possessive/personal pronouns plus nouns, verbs,
adjectives, adverbs, numerals.

**Out of scope:** classifiers, spatial agreement/indexing, role shift, aspectual
movement modulation, complex embedding. Reliability isn't "handles all of ASL" — it's
"handles a defined subset correctly and knows its own boundary."

**Exceptions — ASL has them, but not the English kind.** No irregular past tenses or
irregular plurals exist to trip the engine, because ASL doesn't inflect that way
(time is lexical + the timeline; plurals come from reduplication, numbers, or
classifiers). What does exist:
- **Lexical exceptions** — frozen/lexicalized forms, fingerspelled loan signs (#JOB,
  #BACK), fused historical compounds. The **lexicon absorbs these** as entries.
- **The verb-class split** — agreement verbs inflect for who-does-what-to-whom; plain
  verbs don't. Handle with a **lexicon tag** per verb.
- The genuinely hard spatial grammar isn't "exceptions" at all — it's productive
  spatial grammar living outside the engine's scope, which is why it's scoped out
  rather than except-handled.

Note the Moryossef ruleset is deliberately naive (lemmatize, drop, reorder) and does
**not** do sophisticated exception handling. Start small — a handful of well-tested
constructions plus a solid golden corpus — and lean on fail-closed validation rather
than chasing broad English coverage early.

- **Produces:** English sentence → (ordered real clips to show, concatenated
  reference pose target to grade against).
- **Done when:** a constrained sentence yields a correct gloss ordering + NMM tags +
  composed target, out-of-scope input is refused rather than guessed at, and the
  reference clips play in the right order.

#### 5c — Templates (later; a robustness upgrade, not a replacement)

Correct-by-construction generation: author ASL constructions with typed slots (a time
sign, a noun, a verb from the vocabulary), each template encoding its own word order
and NMM scope. Fill slots with curriculum vocabulary → a correct gloss sequence *by
construction*, with the English prompt generated from the same filled template, so
prompt and target are guaranteed consistent. No parse step to fail; fail-closed is
automatic. This is how the constrained generators that produced genuinely fluent ASL
worked (e.g. the weather-report system's sentence stems encoding topic-comment
structure and non-manual morphemes).

Deliberate ordering decision: **build 5b first.** The translator can drive the whole
generation loop on its own, has more structural range, and enables a free-text-input
feature templates can't cover. Templates are the reliability upgrade folded in during
Phase 6, for constructions where a guaranteed-correct target matters — and they
handle exceptions more cleanly (an exception is just another template with the
irregular form baked in). Nothing is wasted: both share the declarative grammar,
ID-gloss lexicon, fail-closed guard, and golden corpus.

Note the dynamicness the tutor needs is **content-driven** and both approaches have
it equally — one template × 60 vocabulary signs is thousands of gap-targetable
sentences. Templates cost arbitrary *structural* variety, not adaptiveness.

### Phase 6 — The adaptive learning app (wire the loop)

- **Learner model:** per-sign + per-parameter mastery, updated every attempt. The
  per-parameter part is what generalizes ("forehead-location signs get missed across
  the board") to signs not yet drilled.
- **Adaptive engine:** spaced repetition (retention) + gap-targeting (weakest signs
  and parameters) + difficulty control. No fixed lesson order, no test bank.
- **Task generator:** targeted isolated drills; contrastive minimal-pair drills that
  fire automatically when two signs differing in one parameter get confused (the
  phonological data supplies the pairs); (v2) sentence prompts via an LLM
  constrained to the current vocabulary and weak items, each validated through the
  rule engine (§5b) and discarded if out of scope.
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
  release — get fluent Deaf review of the rules and targets. Make it a workflow
  (reviewable rule *data* + a signed-off golden corpus), not an afterthought. Build
  freely; don't ship grammar judgments as truth unreviewed.
- **Hardware split:** RTX 4070 for the live tool and everyday dev; Colab for batch
  extraction over ASL Citizen and long training runs.
- **Out of scope (by design, not failure):** classifiers, spatial agreement,
  free-form translation, and any synthesized/generated sign video.
- **Never train the learned gloss model on rule-generated corpora** (ASLG-PC12) — it
  would only re-learn these rules, not real ASL.
- **Reality check:** ASL Citizen's own SOTA is ~63% top-1 and ~91% recall-at-10 on
  2,731 signs, unseen-signer. A ~60-sign vocabulary will do much better, but
  calibrate expectations: a useful practice aid with real error rates, not an
  oracle — exactly why the grader is distance/retrieval-based and why the learner
  always imitates real Deaf-signer video.

## File map (target)

    src/aslcv/
      extractor/       # DONE — 4 backends (mediapipe/dwpose/rtmw/vitpose) + rtmlib_base
      dataset.py       # DONE — cached-sequence loader + splits + label encoder
      features.py      # Phase 2 — feature assembly (select/concat/confidence/velocity/stack/standardize)
      normalizer/      # Phase 2 — ShoulderNormalizer (global + local-hand); BBox skipped as unsound
      grading/         # Phase 4 — embedding + phonological heads + DTW
      production/      # Phase 5
        gloss_rules.py #   5b — gloss + NMM rule engine (BUILT)
        retrieval.py   #   5a — id_gloss → reference clip + pose sequence (later)
        rules/         #   5b — declarative drop/reorder/NMM rules (reviewable data)
        lexicon.py     #   5b — English lemma → ID-gloss, verb-class tags
        templates/     #   5c — correct-by-construction generation (later)
      learner/         # Phase 6 — learner model + adaptive engine
      generator/       # Phase 6 — drills, contrastive pairs, sentence prompts
    scripts/           # DONE — extract_landmarks.py (extract), verify_cache.py (integrity)
    tools/             # DONE — build_manifest, resolve_keys, join_phonology, validate_curriculum
    app/               # Phase 6 — the loop, feedback presenter (LLM English only)
    tests/
      golden/          # 5b — (English → expected gloss + NMM) regression corpus
    data/              # gitignored
      ASL_LEX/         # DONE — phonological features + ID-gloss keys
      ASL_Citizen/     # DONE — videos + official splits
      manifest.csv, phonology.csv, cache/{extractor}/*.npz   # DONE — Phase 1 outputs
