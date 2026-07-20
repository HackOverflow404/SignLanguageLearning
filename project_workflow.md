# ASL Adaptive Tutor — Build Plan

The definitive plan. Supersedes the older workflow (which described a
MediaPipe-holistic, record-your-own-signs, WLASL, classifier-into-SRS pipeline).

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

Done: `src/aslcv/pose/{base,coco_wholebody,dwpose,mediapipe}.py` — the swappable
extractor layer, both backends working (MediaPipe Holistic as default, DWPose via
rtmlib as alternate), models downloaded locally. Satisfies Phase 2's
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
  name + a distance score, live. Defaults: MediaPipe, shoulder normalization, DTW.

### Phase 3 — Benchmarking (now that a system exists to measure)

**The filter that decides most of this:** training and serving must use the *same*
extractor, so anything that can't run real-time on the 4070 is disqualified as the
live extractor regardless of its accuracy.

**Candidates — four, all cheap.** DWPose-l and MediaPipe Holistic are built. RTMW-x
and ViTPose-133 are one-line URL swaps in the rtmlib `Custom` config — same
COCO-WholeBody 133 topology, nothing downstream changes:

| Candidate | Where it comes from |
|---|---|
| **DWPose-l** | `rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288` — already wired |
| **RTMW-x** | `rtmw-dw-x-l_simcc-cocktail14_270e-384x288` — already in `models/` |
| **ViTPose-133** | rtmlib ViTPose wholebody checkpoint |
| **MediaPipe Holistic** (current default) | built; the *face-detail* contender (468 points + blendshapes) |

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

**Normalizer ablation (an afternoon, not a phase):** shoulder-width vs bounding
box. Expect bbox to lose — the box grows when the hands raise, making normalization
sign-dependent and corrupting the location signal.

**Architectures:** LSTM → Transformer → ST-GCN; compare on the data.

- **Produces:** component decisions backed by numbers.
- **Done when:** extractor, normalizer, and architecture are chosen on measured
  evidence from the actual setup.

> Perspective: among the good whole-body models the accuracy gaps are small next to
> everything else in this project. Run the cheap screen, pick a survivor, move on.
> Don't let extractor selection balloon.

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
      pose/            # DONE — extractor layer
      features.py      # Phase 2 — normalized feature vector
      dataset.py       # Phase 2 — cached-sequence loader + splits
      grading/         # Phase 4 — embedding + phonological heads + DTW
      production/      # Phase 5
        retrieval.py   #   5a — id_gloss → reference clip + pose sequence
        gloss.py       #   5b — GlossSequence interface + spaCy pipeline
        rules/         #   5b — declarative drop/reorder/NMM rules (reviewable data)
        lexicon.py     #   5b — English lemma → ID-gloss, verb-class tags
        templates/     #   5c — correct-by-construction generation (later)
      learner/         # Phase 6 — learner model + adaptive engine
      generator/       # Phase 6 — drills, contrastive pairs, sentence prompts
    app/               # Phase 6 — the loop, feedback presenter (LLM English only)
    tests/
      golden/          # 5b — (English → expected gloss + NMM) regression corpus
    data/
      ASL_LEX/         # DONE — phonological features + ID-gloss keys
      asl_citizen/     # Phase 1 — video + cached pose sequences
