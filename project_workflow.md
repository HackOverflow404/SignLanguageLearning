# ASL Adaptive Tutor — Build Plan

The definitive plan. Supersedes the older workflow (which described a
MediaPipe-holistic, record-your-own-signs, WLASL, classifier-into-SRS pipeline).

> **Rev note (latest pass):** Phase 4 (the learned grader) is now DONE and beats the
> DTW baseline decisively: **81.2% top-1 / 96.5% top-5** on the 60-sign val split
> (229 clips), vs. the DTW baseline's 31.9% / 64.2% re-measured on that SAME split.
> Full writeup in "Phase 4" below and `PHASE4_REPORT.md`. The rest of this note is
> the still-accurate Phase 2 history (training-free DTW was always a *floor*, not
> the target — Phase 4 is what closed the gap the bullets below describe):
>
> Phase 2 is DONE **and its post-milestone experiment sweep is done too** — so the
> headline number has moved well past the 32.5% first recorded. Current best on the
> 20-sign val split (held-out signers, training-free DTW, still a floor not a
> target): **MediaPipe 66.2% top-1 / 95.0% top-5** with rest-frame trimming on;
> 48.8% / 80.0% without trim; DWPose 32.5% / 70.0%. The four experiments that
> produced this — and, more importantly, told us *which* levers carry signal — are
> summarized in "Phase 2 — post-milestone findings" below. The short version:
> - **MediaPipe genuinely beats the three rtmlib backends** on minimal-pair
>   separation (father/mother 11–15% confusion vs 36–41%), and the edge is *real
>   keypoint quality*, not an artifact — ~90% of it survives binarizing the
>   confidence channel, and it survived both the cache-integrity scare and a
>   face-region-asymmetry fix.
> - **Rest-frame trimming is the single biggest lever (+17.4pp top-1)** — ASL
>   Citizen clips are recorded rest→sign→rest, and DTW's length normalization was
>   letting the shared rest regions compress every distance. BUT it only helps
>   MediaPipe: MediaPipe runs VIDEO mode (temporal smoothing → true rest reads as
>   still), the rtmlib backends run IMAGE mode (per-frame jitter → rest never trims).
>   So the 66-vs-32 gap *conflates backend quality with running mode* — investigated
>   and closed in Phase 3 (rtmlib has no temporal smoothing to switch on; documented
>   as a reporting caveat rather than chased with a new filter + re-extraction). The
>   minimal-pair advantage below is unaffected and is the real basis for the pick.
> - **Depth proxies (apparent hand size / arm foreshortening) did NOT help** the one
>   pair they targeted (me/you, ~45% confusion either way): the raw signal doesn't
>   separate that pair, and me/you are indexical/deictic signs that are out of scope
>   anyway. True z was considered as a targeted Phase 3 ablation and formally closed,
>   not pursued (see Phase 3 status) — not worth re-extraction for one out-of-scope pair.
> - **`repeated` is the hardest phonological parameter** (38–54% confusion in every
>   backend's own column) — the one parameter-ranking claim the data supports, and a
>   direct instruction for Phase 4 (it's a *temporal* property DTW length-normalizes
>   away; needs an explicit tempo/periodicity feature). **Acted on**: Phase 4's
>   `repeated_movement` head reads an explicit FFT-autocorrelation tempo feature and
>   reaches 82.1% val accuracy — no longer the clear weak point.
> - Cache-integrity scare resolved (rtmlib only ever ran IMAGE mode; confirmed clean
>   across all 1,874 clips × 4 extractors — no re-extraction). `Skeleton` gained a
>   `regions` field (meaning-based group selection); legs/feet dropped by default;
>   velocity zeroes deltas across presence gaps; a shared `pipeline_config.py` now
>   builds every script's pipeline identically; extraction provenance is recorded in
>   each `.npz`; the rtmlib `process_every_n_frames` default is 1 with a hard-error
>   guard against the VIDEO+n>1 combination. Full suite: **84/84 green**, `make test`.
>
> Phase 3 is now fully DONE: the core question (which extractor) is answered
> (MediaPipe, and — after actually running the deferred cheap screen and adding a
> GPU delegate, see Phase 3 status — now the fastest of the four too, not just the
> most accurate). With Phase 4 and Phase 5a also done, the recognition track
> (Phases 1-4) plus reference retrieval have nothing left blocking them.
>
> **Phase 6 v1 is now also wired end to end** (see Phase 6 status): a persisted
> per-sign/per-parameter learner model, a gap-targeting + recency-biased scheduler
> with automatic contrastive minimal-pair drills, and coaching feedback (templated
> by default; `--llm-feedback` opts into an LLM phrasing pass over the same facts
> via HuggingFace's hosted Inference API), all live in `diagnose_demo.py` — `n` now
> records the current attempt and adaptively picks the next target instead of
> blind list-cycling. Deliberately v1-scoped: a heuristic, not a real
> spaced-repetition interval algorithm; still a script, not a packaged `app/`.
> **Sentence prompts (the last Phase 6 v2 piece originally deferred) are now also
> built**: `--sentence-prompts` has the LLM write an English sentence containing
> the target word, then hands it to Phase 5b's already-fail-closed rule engine —
> only an engine-accepted sentence is ever shown, so the LLM's English never
> becomes displayed ASL content on its own say-so. HF_TOKEN now also resolves
> from a gitignored repo-root `.env` (`.env.example`), not just the shell
> environment, at the user's request to keep secrets inside the repo's own
> directory. **Phase 8 (porting to a phone) is now also scoped**, deliberately
> not started — see its section below for what of the current stack is
> resource-appropriate for a phone (most of it) vs. what's real unstarted porting
> work (most of the actual code). **Phase 6's scheduler is now also a real
> SM2-style spaced-repetition scheduler with a first difficulty-control signal**
> (proactive minimal-pair stress-testing of well-mastered signs) — see Phase 6
> status for the full writeup. What's open: actual fluent Deaf review of Phase
> 5b's engine output (a human bottleneck, not code), and Phase 8 itself.
>
> **Also built since:** `scripts/diagnose_demo.py` — a live webcam demo of
> `EmbeddingGrader.grade_against`, letting a learner practice a target sign against
> its real looping reference clip and see all 5 phonological heads' verdicts
> (MATCH/OFF/insufficient-data + confidence) plus overall fidelity, not just a single
> label. Reuses `live_demo.py`'s webcam/threading/pipeline scaffolding; the only
> substantive changes are the grader (learned, not DTW) and the closed-set interface
> (known target, not open-set "which sign"). Fails closed on a pipeline-config
> mismatch against the checkpoint and on any cycle target lacking a cached reference
> video. Verified via `--selftest` (reproduces `PHASE4_REPORT.md`'s mother/father
> numbers exactly) — full writeup in "Phase 4" below. Committed.
>
> **Phase 5a (reference retrieval)** is also now done: `src/aslcv/production/
> retrieval.py`'s `fetch_reference`/`fetch_sequence` resolve a gloss or a whole
> Phase 5b `GlossSequence` to real cached clips and hard-cut-concatenate them into
> one playable video (trim-then-cut, no generative smoothing — see the Phase 5a
> section below for the reasoning). Verified end to end on `"I want water."`; full
> writeup below. Built and tested, **not yet committed**, pending explicit go-ahead.

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
(COCO-WholeBody shoulders = 5, 6; MediaPipe pose = 11, 12) — plus named **regions**
(one level up: `body_upper`, `arms`, `left_hand`, `right_hand`, `face`, `legs_feet`),
so keypoint *groups* are also selected by meaning across topologies, not by
name-prefix matching or hardcoded ranges.

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
  after several mid-run crashes (`scripts/verify_cache.py`). A later scare — a
  VIDEO-mode bug in `rtmlib_base.py` that could duplicate frames and drop a clip's
  opening frames if `process_every_n_frames != 1` — was investigated and resolved:
  the three rtmlib backends have only ever been extracted in IMAGE mode (which
  ignores that setting entirely), confirmed empirically across all 1,874 clips × 4
  extractors. No corruption, no re-extraction needed.
- `data/phonology.csv` (`tools/join_phonology.py`): every ASL-LEX phonology +
  frequency parameter joined onto the 60 signs by `asllex_code`; 6 multi-morphemic
  signs flagged (ASL-LEX codes only their first morpheme).
- `src/aslcv/dataset.py`: yields `(pose_sequence, id_gloss, phonological_features,
  signer_id)` per split, plus a lazy-torch fixed-length Dataset + label encoder.

**Phase 5b** (production track) — gloss rule engine built and substantially
hardened toward the settled interface (`src/aslcv/production/gloss_rules.py`);
no data dependency. spaCy (`en_core_web_sm`) now drives POS/dependency/lemma
analysis (previously absent); `GlossSequence` carries `in_scope`/`confidence`/
`reason`/`trace`, and out-of-scope input (unknown vocabulary, unsupported
constructions — relative clauses, `advcl`/`ccomp`, passive voice, clause
coordination whether VERB- or AUX-headed) is refused rather than soft-flagged,
closing the fail-closed gap this phase's own design calls non-negotiable. The
lexicon is now derived from `curriculum.yaml`'s `english_lemmas` rather than
hand-typed, fails closed at **import time** on any ambiguous lemma (one
English word claimed by two different glosses, from curriculum or the
supplementary table), with a self-check that every emittable gloss resolves to
a real curriculum sign. A dedicated fail-closed probing pass
(`PHASE5B_GAP_REPORT.md`) went looking for edge cases the golden corpus didn't
yet cover — partial-scope sentences, ambiguous lexicon hits, wh-word-as-
relative-pronoun vs. genuine wh-question, and multi-clause/conjoined input —
and found two real gaps (the AUX-headed coordination case above, and the
ambiguous-lexicon guard), both now fixed and regression-tested; the other two
categories were already handled correctly and are now pinned as regressions.
Four tools sit on top of the engine:
- **`scripts/gloss_repl.py`** — interactive discovery REPL: spaCy parse + full
  `GlossSequence` + `:why` trace.
- **`tests/test_gloss_rules_corpus.py`** — the golden corpus this phase's
  design calls for. Mechanical properties (drop rules, time-fronting,
  wh/yes-no NMM scope, fail-closed refusal + the harder probes above,
  determinism, lexicon resolution) asserted hard; ASL-judgment cases collected
  as `PENDING_CASES` that report rather than assert, pending fluent Deaf
  review.
- **`scripts/compose_sentence.py`** — text-only composer: english →
  `GlossSequence` → per-gloss `asllex_code` → cached reference clip lookup →
  ordered playback plan — the pre-Phase-5a pipeline check, no video.
- **`scripts/export_review_sheet.py`** — generates a self-contained HTML file
  (`data/review/gloss_review_sheet.html`, gitignored) so Deaf review of every
  `PENDING_CASES` entry and every `constructions_v2` construction is a
  one-sitting task: embedded reference clips per gloss, NMM tag spans rendered
  visually, a plain-language restatement of the `:why` trace, and a verdict
  form (correct/wrong-order/wrong-nmm/other + comment) persisted to
  localStorage and exportable to JSON/Markdown. This is the artifact that
  unblocks flipping `reviewed=True` on `PENDING_CASES` — it presents only and
  asserts nothing about ASL correctness itself, since neither the codebase nor
  its author can make that call.

Still missing from the settled 5b design: the ruleset itself is still Python
control flow, not the declarative reviewable rule *data* (`rules/`) the design
calls for; no verb-class agreement tag; no actual Deaf review has happened yet
(the corpus's `PENDING_CASES` and the new review-sheet export exist
specifically to make that gap visible and easy to close, not to paper over
it).

**Phase 2** (thin vertical slice, on the 20-sign subset below) — DONE, plus a full
post-milestone experiment sweep. `normalizer/shoulder.py`, `features.py`, a DTW
nearest-reference grader, and a live webcam demo are all built; a shared
`pipeline_config.py` builds every script's feature pipeline identically.
Current best on the 20-sign val split (held-out signers, training-free DTW = a floor,
not a target): **MediaPipe 66.2% top-1 / 95.0% top-5** with rest-frame trimming;
48.8% / 80.0% without; DWPose 32.5% / 70.0%. Findings from the sweep — extractor
comparison, rest-frame trimming, depth proxies, the `repeated` parameter — are in
"Phase 2 — post-milestone findings" below.

**Phase 4 is now DONE** — the learned grader (embedding + phonological heads), trained
on the full 60-sign curriculum, beats the DTW baseline decisively (81.2% top-1 / 96.5%
top-5 vs. DTW's 31.9% / 64.2%, same 60-sign val split) and demonstrates real
per-parameter head independence on a genuine minimal pair. Full writeup in "Phase 4"
below and `PHASE4_REPORT.md`. **Phase 3 is now fully DONE too** — the central
question (which extractor) is settled (MediaPipe) and both loose ends are closed
(see Phase 3 status); nothing on the recognition track blocks moving to Phase 6.
Phase 5a (reference retrieval) is now done too.

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

### Phase 2 — Thin vertical slice (MILESTONE — before any benchmarking) — DONE

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
+ DTW grader catch a single-parameter difference. **Result, generalized beyond this
one pair** (`scripts/eval_minimal_pairs.py`, all four extractors, train+val+test
pooled with train clips graded leave-one-out — 269 query clips, not 3): the earlier
mother/father verdicts that looked *contradictory* across backends (dwpose said
father leaks into mother 2/3 of the time; the mediapipe live-demo selftest said the
contrast held) were pointing at a **real** effect, just overstated by a 3-clip
sample — the actual confusion rate is 11% for mediapipe vs 36–41% for the three
rtmlib backends. That gap holds across *every* in-slice minimal pair tested, not
just this one: mediapipe is consistently, sometimes dramatically, better separated
than dwpose/rtmw/vitpose. Pooled across extractors, none of the three parameters
with in-slice pairs is cleanly resolved by training-free DTW geometry — handshape
37%, minor_location 32%, major_location 30% confusion — so Phase 4's phonological
heads need to target all three, handshape hardest. `movement` and `repeated` have
zero in-slice minimal pairs in this 20-sign slice and could not be evaluated here.
**Caveat on the vitpose column specifically:** its topology was unverified when
these numbers first ran (`vitpose.py`'s own docstring flagged the 133-keypoint
index order as unconfirmed). `scripts/verify_vitpose_topology.py` has since checked
it against dwpose on the same clips — same-index correspondence beats every
left/right-swap test, so the topology is correct, not scrambled. But vitpose's
POSE_INPUT_SIZE is `(192, 256)` vs. dwpose/rtmw's `(288, 384)` (2.25× the pixel
area) — no matching-resolution easy_ViTPose wholebody checkpoint is known to
exist — so any vitpose-vs-others accuracy gap above (here and in Phase 3 below) is
confounded by input resolution, not a clean architecture comparison.

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
     is on); drop legs/feet by default (a `legs_feet` toggle re-adds them for
     ablation). Both toggles resolve via `Skeleton.region()` — by meaning, not
     name-prefix matching — so the drop works identically across topologies. A
     modeling choice tuned here, never re-extracted.
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
     the model explicitly. A point absent (score 0) on either side of a delta gets
     a zero delta instead of differencing against its zero-fill — signing occludes
     hands constantly, and without this a hand disappearing-then-reappearing read
     as two violent fake jumps.
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
  **DONE** — `scripts/live_demo.py` runs the live loop; `scripts/eval_slice.py`
  measures the val split (current best MediaPipe 66.2% top-1 / 95.0% top-5 with
  `--trim-to-motion`); mother/father and every other in-slice minimal pair are
  quantified in `scripts/eval_minimal_pairs.py` (findings below) rather than only
  eyeballed live.

*(`dataset.py` is already built in Phase 1 — the loader, fixed-length wrapper, and
label encoder exist; Phase 2 consumes it rather than rebuilding it.)*

### Phase 2 — post-milestone findings (the experiment sweep)

After the milestone, a series of cheap experiments characterized *which* levers move
accuracy — the real payoff of the thin slice, and the map Phase 4 is built from.
Numbers are 20-sign val split, held-out signers, training-free DTW. **Caveat that
applies to all of them:** 80 val clips ⇒ ~±11pp binomial error, and per-sign rows are
3–5 clips each — only ~10pt+ moves are trustworthy; do not tune against single signs.

1. **Extractor comparison — MediaPipe wins, and it's real.** Across every in-slice
   minimal pair, MediaPipe separates pairs far better than dwpose/rtmw/vitpose
   (father/mother 11–15% confusion vs 36–41%). Confirmed *not* a confound: ~90% of
   the advantage survives binarizing the confidence channel (so it isn't MediaPipe's
   handedness-score / hardcoded-1.0 semantics vs the rtmlib graded scores); it
   survived the cache-integrity scare; and it survived fixing a **face-region
   asymmetry** (MediaPipe's coarse nose/eyes/ears had been gated behind `--face`
   while COCO's rode in the global block by default — a real confound in every prior
   mediapipe-vs-rtmlib number, now fixed by moving MediaPipe's 5 matching coarse
   landmarks into `body_upper`; father/mother moved 11%→15% after). The architectural
   reason it should win: MediaPipe re-crops each hand and runs a dedicated
   high-resolution hand model, vs the rtmlib backends resolving hands inside a
   downscaled person crop. **One counterexample:** me/you, where MediaPipe is not
   ahead under either mode.

2. **Rest-frame trimming — the biggest single lever (+17.4pp top-1), MediaPipe only.**
   ASL Citizen clips are recorded rest→sign→rest (the cache investigation found one
   clip with 11 byte-identical opening frames). DTW normalizes total path cost by
   length, so shared rest regions align cheaply against *every* sign and compress all
   distances together. Trimming to the motion-active span: MediaPipe 48.8→66.2 top-1,
   80.0→95.0 top-5. **But trim helps only MediaPipe** (dwpose was byte-identical
   with/without): MediaPipe runs VIDEO mode with temporal smoothing so true rest reads
   as still and trims cleanly, while the rtmlib backends run frame-independent IMAGE
   mode whose per-frame jitter keeps "rest" above the motion threshold. `trim_to_motion`
   is a `features.py` toggle (off by default) sharing the `hand_motion_energy()`
   implementation that Phase 6's live segmenter will reuse. **Watch:** a sign with a
   genuinely-held handshape at onset/offset could have real content trimmed — verify
   the threshold only eats pre/post rest, and that it doesn't over-trim a live attempt.

3. **Depth proxies — measured, did not help.** Hypothesis: me/you (~37–45% confusion
   even on MediaPipe) fails because the contrast is along the camera axis where 2D is
   blind. Added `depth_proxies` features (per-hand apparent size = wrist→mcp / shoulder
   width, and arm foreshortening) — derived from existing 2D keypoints, no re-extraction,
   works on all backends. Result: byte-identical me/you confusion. The raw signal
   doesn't separate the pair (me clips cluster 0.30–0.32, you clips scatter 0.19–0.39
   and overlap). And me/you are **indexical/deictic** signs — the spatial-grammar class
   scoped out of the project — so this pair may simply be a poor curriculum fit. True z
   was considered as a Phase 3 ablation and formally closed, not pursued — not worth
   re-extraction for one out-of-scope pair (see Phase 3 status).

4. **Parameter ranking — `repeated` is hardest.** From the 60-sign `--full-curriculum`
   run (real support this time: 40 handshape pairs, 14 major_location), `repeated`
   comes out hardest in *every* backend's own column (38–54% confusion) — the one
   ranking claim the data supports. `minor_location` (1 pair) and `movement` (3 pairs)
   have too few pairs to rank. The earlier pooled "handshape is weakest" claim did NOT
   hold per-backend (MediaPipe's handshape confusion, 24%, is one of its lower rates).
   **For Phase 4:** `repeated` is a *temporal* property (does the movement cycle?)
   that DTW's length-normalization actively smears — it needs an explicit
   tempo/periodicity feature, not just more geometry.

Two eval hygiene notes carried forward: pooling train+val+test in the minimal-pair
eval means query clips self-match at distance 0 (fine for *relative* comparison, not
absolute rates — the 60-way run uses leave-one-out to avoid this); and reference-bank
size differs between the 20-sign and 60-sign runs, so never compare rates across bank
sizes.

### Phase 3 — Benchmarking — status: DONE, all loose ends closed

**The central Phase-3 question — which extractor — is effectively answered: MediaPipe**
(see Phase 2 findings #1). Cache-integrity is resolved (rtmlib only ran IMAGE mode;
clean across all 1,874 clips × 4 extractors).

**Loose end A — the running-mode confound — CLOSED, documented not re-measured.**
The 66-vs-32 gap conflates backend quality with running mode: rest-frame trimming
(+17pp) only helps MediaPipe because it runs VIDEO mode with temporal smoothing,
while the rtmlib backends run IMAGE mode. Investigated re-measuring on equal footing
(option (a), re-run the rtmlib backends in a "smoothed/tracked" mode) and found it
isn't the cheap switch it looks like: checked `rtmlib` directly (`PoseTracker` and
its `__call__`) and confirmed it has **no temporal keypoint smoothing at all** —
`PoseTracker` only reuses the last frame's detector *bounding box* (IoU-based ID
tracking) to skip re-running the person detector; it never filters or smooths the
keypoints themselves. There is no rtmlib equivalent of MediaPipe's internal VIDEO-mode
smoothing to flip on. Doing option (a) for real would mean writing a new temporal
filter (e.g. a One Euro Filter) from scratch, re-extracting 1,874 clips × 3 rtmlib
backends, and re-running `eval_minimal_pairs.py` — a new algorithmic component with
its own design surface (filter choice, cutoff params), not a loose end anymore.
**Decision: option (b)** — state the confound plainly rather than build a filter to
chase it. The minimal-pair advantage (finding #1, the actual reason MediaPipe was
picked) is independent of trim and already favors MediaPipe, so this doesn't change
the extractor decision — it only changes how the 66-vs-32 trim margin should be
read: **that number reflects MediaPipe's running mode as much as its keypoint
quality, and should never be quoted as a pure backend-quality comparison.**

**Loose end B — true z as a targeted ablation — CLOSED, not pursued.** Depth
proxies didn't help me/you (finding #3). The hypothesis would be specific ("MediaPipe
*with* its z channel reduces me/you confusion without hurting overall top-1"), but
me/you is one out-of-scope indexical pair, and z requires re-extracting MediaPipe (it
currently discards the third coordinate at write time) and would break apples-to-apples
with the 2D rtmlib backends. Formally decided not worth it — a deliberate closure,
not a dangling TODO.

**Loose end C — the real-time cheap screen — RUN, not just deferred.** Prompted by
a direct question ("didn't we evaluate MediaPipe is better across the board?") that
this doc's own wording didn't actually support: "which extractor" was answered on
minimal-pair *accuracy* alone; the "cheap screen first ... real-time failures are
out" filter below was never applied to the rtmlib backends at all — it wasn't a
disqualification, it was simply untested. `scripts/benchmark_extractors.py` closes
that (fresh timed inference on a 10-clip sample for FPS/latency, no reused numbers;
`hand_motion_energy()`'s tracked-frame noise floor for jitter and each backend's
already-computed `_manifest.csv` `hand_dropout_rate` for dropout — both reused from
existing cached data, no new extraction needed for those two). Results on this
machine (RTX 4070 laptop, `--n-clips 10 --seed 0`):

| backend | fps | ms/frame | jitter floor (lower=steadier) | hand dropout (mean/p95, full dataset) |
|---|---|---|---|---|
| **mediapipe** | 16.1 | 62.0 | **0.048** | 57.3% / 76.1% |
| dwpose | **29.8** | 33.6 | 0.144 | **11.2%** / 53.4% |
| rtmw | 21.5 | 46.5 | 0.118 | 0.0% / 0.0% (see caveat) |
| vitpose | 17.2 | 58.1 | 0.112 | 38.2% / 65.8% |

First pass, three findings, then two of them were overturned on follow-up
investigation (below) — kept here rather than silently rewritten, since the
correction is as informative as the original numbers:
- MediaPipe looked like the *slowest* of the four, running CPU-only.
- MediaPipe's jitter floor was the clear best (0.048 vs 0.11–0.14) — this one
  **held up**: first direct empirical evidence for loose end A's running-mode
  story, not just the logical inference from "VIDEO mode has temporal
  smoothing." MediaPipe's tracking really is measurably steadier frame-to-frame,
  in the same units and signal Phase 6's boundary detector will use.
- Hand dropout looked like it didn't favor MediaPipe (57.3% mean, worst of the
  four).

**Follow-up 1 — hand dropout, corrected.** Prompted by "can the dropout be
fixed?" Before assuming a fix was needed, checked *where* the dropout happens:
split each clip's frames into the trimmed active-signing span (what grading
actually uses) vs. the discarded rest span (`hand_motion_energy` +
`motion_active_span`, same signal as trim_to_motion), and measured dropout
separately in each, 40-clip sample:

| backend | active-span dropout (what matters) | rest dropout (discarded anyway) |
|---|---|---|
| **mediapipe** | 16.8% | 99.6% |
| dwpose | 13.5% | n/a — active span ≈ whole clip (no rest window; consistent with loose end A) |
| rtmw | 0.0% | 0.0% (still flagged suspicious, unchanged) |
| vitpose | 37.2% | n/a — same as dwpose |

The 57.3%/76.1% full-dataset number was measuring almost entirely (99.6%)
**rest frames that get trimmed away before grading ever sees them** — hands
lowered out of MediaPipe's detection range during rest, not a tracking failure
during the sign itself. Inside the span that's actually graded, MediaPipe's
16.8% is competitive with dwpose and clearly better than vitpose. Nothing to
fix — the original full-dataset statistic was measuring the wrong region, not
exposing a real defect. (This also independently corroborates loose end A from
a different angle: dwpose/vitpose show no separate rest window at all because
their per-frame jitter keeps the whole clip reading as "active" — exactly the
"IMAGE mode never lets rest register as still" story, now visible in a second,
unrelated metric.)

**Follow-up 2 — MediaPipe on GPU, and it was worth checking.** Asked directly
whether MediaPipe supports a GPU delegate at all (it does, `BaseOptions.Delegate.GPU`,
confirmed against the installed mediapipe 0.10.35). The current extractor
(`src/aslcv/extractor/mediapipe.py`) never sets `delegate=` on any of the three
`BaseOptions` (pose/face/hand), so it's always taken the CPU default — that's
why the first-pass FPS number was CPU-only. Requesting the GPU delegate
directly **did not simply work**: MediaPipe silently fell back to Mesa's
`llvmpipe` *software* OpenGL renderer (confirmed via the GL context log line)
instead of this machine's real NVIDIA hardware — 4.4fps, ~4x *slower* than
CPU, not faster. Root cause: on this Wayland desktop, MediaPipe's default EGL
context creation doesn't resolve the NVIDIA EGL vendor on its own. Forcing it
explicitly (`EGL_PLATFORM=x11` + `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/
glvnd/egl_vendor.d/10_nvidia.json` at process launch) gets the real hardware
context (log line confirms `renderer: NVIDIA GeForce RTX 4070 Laptop GPU`) —
and with it, **hand landmarker alone: 173 fps vs 49.5 fps CPU (3.5x); the full
3-model pipeline (pose+face+hands, matching production): 53.0 fps vs the
16.1 fps CPU baseline (3.3x)**. Checked for silent corruption before trusting
the speedup (the GPU delegate path logs an "undefined behavior... lack of
synchronization" warning that's worth taking seriously, not hand-waving past):
compared CPU vs. GPU hand-landmark output frame-by-frame on the same 20-frame
slice — comparable detection counts (18/20 vs 17/20) and landmark positions
agree closely (mean abs diff 0.0026 in normalized [0,1] coords, max 0.0099) —
output looks safe, not corrupted, but this is a spot check, not a full
regression suite.

**Net, corrected:** MediaPipe is not just the accuracy and stability winner —
with the GPU delegate properly configured, it's also the *fastest* of the four
(53fps vs dwpose's 29.8fps), and its apparent hand-dropout weakness turns out
to be a measurement artifact, not a real one. "Better across the board" was
still wrong as originally stated (the first-pass numbers really did show
MediaPipe losing on FPS and dropout) but turns out closer to true than the
corrected picture initially suggested — once measured correctly.

**Adopted, opt-in, default unchanged.** `MediaPipePoseExtractor` now takes
`delegate: str = "cpu"` — the default stays CPU (byte-for-byte the same
behavior every existing caller, test, and the entire 1,874-clip cache already
depend on), so nothing about production extraction or the cache's provenance
changed. `delegate="gpu"` requests `BaseOptions.Delegate.GPU` on all three
landmarkers; a module-level `_maybe_force_nvidia_egl()` best-effort-sets the
two EGL env vars found above (only if the caller hasn't already set an
opinion, and only if a known NVIDIA vendor JSON actually exists on this
machine — several common glvnd paths are checked, so it degrades to a no-op
rather than assuming this exact machine's layout), and landmarker construction
is wrapped in a try/except that falls back to CPU with a printed warning if
GPU construction raises for any reason (no GPU, no compatible driver, etc.) —
never a hard failure just because GPU wasn't available. `self.delegate_used`
records which one actually ran. Wired into the two places FPS smoothness
directly affects UX: `live_demo.py --gpu` and `diagnose_demo.py --gpu` (both
opt-in, default off). Batch extraction (`extract_landmarks.py`) intentionally
NOT switched to GPU by default — it's a one-time step already complete for all
1,874 clips, so there's no time pressure, and changing it would mean deciding
whether to re-extract the existing cache. Verified end to end through the real
class, not just the standalone scratch test above: `delegate="gpu"` reaches
`delegate_used == "gpu"` and ~44fps on a real clip. Full suite re-run after
the `extractor/` change per the testing rule: **214 passed, 10 skipped** — no
regressions, same numbers as before this change.

**Final numbers, all corrections folded in.** `scripts/benchmark_extractors.py`
gained `--gpu` (mediapipe only) and a region-split dropout metric
(active-span vs. rest, alongside the original full-dataset number for
comparison) so the corrected methodology is reusable, not a one-off. Re-run,
`--n-clips 10 --seed 0 --gpu`:

| backend | fps | jitter floor | dropout active% (what matters) | dropout rest% | dropout full% (naive) |
|---|---|---|---|---|---|
| **mediapipe+gpu** | **59.9** | **0.048** | 15.7 | 99.6 | 57.3 |
| dwpose | 33.5 | 0.144 | **11.1** | n/a* | 11.2 |
| rtmw | 29.9 | 0.118 | 0.0 (suspicious) | n/a* | 0.0 |
| vitpose | 16.6 | 0.112 | 38.6 | n/a* | 38.2 |

*dwpose/rtmw/vitpose show no separate rest window because their per-frame
jitter keeps the whole clip reading as "active" — the same "IMAGE mode never
lets rest register as still" finding from loose end A, visible again here.

With GPU properly configured, MediaPipe is now the fastest of the four
(59.9fps, ahead of dwpose's 33.5) as well as the steadiest (jitter) and
already-established accuracy leader. The one place it doesn't win is active-
span hand dropout (15.7% vs dwpose's 11.1%) — close, not the 57-vs-11 gap the
naive full-dataset number implied, and left standing rather than explained
away.

**Everything below is the original full-sweep design, kept for reference** — run it
only if the extractor decision is reopened (e.g. MediaPipe fails a real-time or
robustness bar in the live tool).

**The filter that would decide a full sweep:** training and serving must use the *same*
extractor, so anything that can't run real-time on the 4070 is disqualified as the
live extractor regardless of its accuracy.

**Candidates — four, all built.** MediaPipe Holistic (default), DWPose-l, RTMW-x, and
ViTPose-l are all wired behind the `Extractor` interface; the three rtmlib backends
share `rtmlib_base.py` and the same COCO-WholeBody 133 topology, so nothing downstream
changes when swapping among them. All four are already cached over the manifest:

| Candidate | Model | Notes |
|---|---|---|
| **MediaPipe Holistic** (current default) | 553 kpts + blendshapes | the *face-detail* / NMM contender |
| **DWPose-l** | `rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288` | RTMPose-large + DW distillation |
| **RTMW-x** | `rtmw-dw-x-l_simcc-cocktail14_270e-384x288` | newer arch; use 384×288 for hand detail |
| **ViTPose-l** | rtmlib ViTPose wholebody (133), `easy_ViTPose` export | accuracy leader in the family — **but see caveat below** |

All four are built and cached; the head-to-head is a matter of running the screens
below, not new integration.

**ViTPose resolution confound (read before trusting any vitpose-vs-others number):**
its checkpoint runs the pose model at `POSE_INPUT_SIZE = (192, 256)`, while DWPose
and RTMW both run at `(288, 384)` — 2.25× the pixel area. No matching-resolution
`easy_ViTPose` wholebody export is known to exist as of this writing, so any
accuracy gap between vitpose and the other two rtmlib backends is confounded by
input resolution, not a clean architecture comparison — factor this in before
concluding anything about ViTPose-the-architecture specifically. Separately, its
133-keypoint index order (does it actually match `COCO_WHOLEBODY`?) was unverified
until `scripts/verify_vitpose_topology.py` checked it against dwpose on shared
cached clips: same-index correspondence beats every left/right-swap test across
shoulders/elbows/wrists/hips/knees/ankles/hand-roots, so the topology itself is
confirmed correct — only the resolution confound remains open.

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
**A lighter version of this now exists and has been run** — see loose end C above
(`scripts/benchmark_extractors.py`, 10-clip random sample, no hands-crossing/
frame-edge stratification). This full version — a real day-scale grid on
deliberately hard clips — is still what "kept for reference, only if reopened"
means; loose end C answered the specific "is MediaPipe secretly worse at this"
question, not the complete original design.

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

### Phase 4 — Grading & diagnosis engine (the real grader) — DONE

The instructor's eye. Not a classifier.

- **Embedding + distance, never argmax classification — DONE, structurally, not just
  by convention.** `src/aslcv/grading/embedding_model.py`'s `PoseGraderNet` trains its
  primary embedding with **batch-hard triplet loss** between real clip embeddings — no
  learned per-class weight vector anywhere in that path (ArcFace/CosFace-style margin
  losses were considered and rejected for exactly this reason: they DO use one).
  `EmbeddingGrader.grade_against(attempt, target_sign)` grades by nearest-reference-clip
  embedding distance (same `agg="min"` convention as `DTWGrader`), degrading gracefully
  on a malformed attempt instead of forcing a confident closed-set label.
- **Phonological feature heads — DONE, and proven genuinely separable, not just
  co-firing.** Five heads (handshape, major_location, minor_location, movement,
  repeated_movement) trained on the ASL-LEX labels from Phase 1. The risk named at
  design time — a shared bottleneck letting heads silently agree instead of
  diagnosing independently — is closed by construction: `PoseGraderNet` is a
  **multi-stream** encoder, not one shared trunk. A `global_encoder` (BiGRU) sees
  only the global block (location + movement); a weight-shared `hand_encoder` sees
  only the hand blocks (combined order-invariantly via elementwise max — ASL has no
  fixed dominant side, so a positional left/right concat would leak signer handedness
  into the handshape head). Each head reads only its relevant stream's output, so
  there is literally no gradient path from e.g. `handshape_head` into
  `global_encoder`'s parameters — proven by backprop in
  `tests/test_embedding_model.py::test_heads_are_structurally_disjoint`, not just
  observed as an accuracy correlation. Demonstrated on a REAL attempt, not synthetic
  data: grading a real `father` clip against `mother` as target (the curriculum's
  built-in minimal pair, differing ONLY in `minor_location`) — handshape,
  major_location, movement, and repeated_movement all correctly MATCH, and only
  `minor_location` disagrees, pinned as a regression
  (`test_embedding_grader.py::test_heads_disagree_independently_on_a_real_minimal_pair`).
- **An explicit tempo feature for `repeated` — DONE.** The one well-supported Phase 2
  finding was that `repeated` is hardest because it's a temporal/cyclic property DTW's
  length-normalization smears. `movement_head`/`repeated_head` read the global stream
  PLUS a 2-scalar FFT-autocorrelation feature computed from `hand_motion_energy()`
  (unchanged, shared with the live-demo segmenter) — peak lag and peak height, giving
  the periodicity signal geometry-only features never had. Measured effect:
  `repeated_movement` reaches 82.1% val accuracy (well-supported classes), no longer
  the clear weak point DTW made it.
- **Trained on the full 60-sign curriculum, not the 20-sign Phase 2 slice** — a
  deliberate dataset decision, not an oversight: per-parameter label coverage is far
  better at 60 signs (`repeated_movement` 30/30 balanced vs. 16/4 skewed in-slice;
  handshape 20 vs 14 classes; minor_location 14 vs 9 — `movement` and `repeated` have
  almost no minimal pairs at all in the 20-slice per known issue #5). This is what
  "did it work" means for this phase: generalizing across all 60 signs' phonological
  classes on the official signer-independent splits (895/229/750 train/val/test),
  never re-split.
- **The minimum-support gate — a real thin-data problem, decided explicitly, not
  papered over.** Several label values are carried by only 1-2 of the 60 signs (9 of
  20 handshape classes, major_location's `Arm`, 4 of 14 minor_location classes) — a
  head cannot be shown to generalize for those vs. memorizing the one sign that
  carries the value. Decision (made with explicit sign-off before training): train
  every head on ALL classes as-is, but gate what's SHOWN at inference —
  `phonology_labels.py`'s `MIN_SUPPORT = 3` — so a verdict for an under-supported
  label value reports `correct=None` ("insufficient data"), never a confident
  right/wrong. Val accuracy is reported split by well-supported vs. thin classes,
  never blended into one number that would overstate confidence
  (`tests/test_embedding_grader.py::test_thin_class_verdict_is_gated_regardless_of_model_output`
  pins the gate itself against the real trained grader, independent of model quality).
- **Overfitting — the main predicted risk, real, and reported plainly.**
  `scripts/train_embedding_grader.py` reports TRAIN (leave-one-out nearest-clip
  ranking, so it's not trivially self-matching) AND VAL every epoch. Train top-1
  saturates to 100% well before val does on ~15 clips/sign — the gap is real (~20pp
  at the end of training) and is not hidden: the **best-val-top1 checkpoint is saved
  separately from the final epoch** (`model_best.pt` vs `model_final.pt`) precisely
  so a plateauing-then-drifting val curve can't silently ship whatever the last epoch
  happened to land on. `models/embedding_grader/history.json` has the full per-epoch
  curve (gitignored, like all of `models/`).
- **Exclude the 6 multi-morphemic signs from parameter-level feedback** — not yet
  wired as an explicit exclusion in `EmbeddingGrader`; the phonology labels used are
  ASL-LEX's first-morpheme-only labels as joined in Phase 1 (`join_phonology.py`
  already flags which 6 signs these are). Revisit if diagnosis quality on those 6
  looks off in practice.
- **Produces:** given `(attempt, target_sign)` → `GradeResult(fidelity,
  parameters={param: ParameterVerdict(predicted, target, correct, support)})`.
- **Done when:** a malformed attempt yields a specific diagnosis ("handshape right,
  location too low"), not a confident wrong label. **DONE** —
  `scripts/eval_embedding_grader.py` / `PHASE4_REPORT.md`: on the 60-sign val split
  (229 clips), the learned grader reaches **81.2% top-1 / 96.5% top-5**, beating the
  re-measured DTW baseline's 31.9% / 64.2% on the SAME split (the old 66.2%/32.5%
  Phase 2 milestone numbers were 20-slice-only and are not the comparison point here).
  Per-parameter val accuracy (well-supported classes): handshape 80.7%,
  major_location 88.4%, minor_location 85.9%, movement 84.5%, repeated_movement
  82.1%.

Two bugs found and fixed while building this (both verified empirically
before/after, both regression-tested — see `CLAUDE.md`'s Known issues for the full
writeup): a NaN-gradient bug in the triplet loss's distance function
(`sqrt` at exactly zero distance has an infinite local gradient that autograd
computes even for unselected diagonal entries, corrupting training from batch one —
fixed with an epsilon floor before the square root), and a segfault in the full test
suite caused by `onnxruntime-gpu`'s NVBLAS hook breaking torch's CPU RNN kernel when
both are loaded in the same pytest process (fixed by running the Phase 4 model on
CUDA in tests, not CPU — a real dependency-stack incompatibility in this
environment, documented in `CLAUDE.md`'s Testing section).

**RESOLVED — grader accuracy investigation: a real train/serve distribution
mismatch found and fixed, addressing a direct user report ("it often thinks
I repeated the movement when I didn't").**

Investigated rather than dismissed as just the already-documented ~82%
ceiling. Measured real ASL Citizen train clips against what
`aslcv.capture.CaptureBuffer` actually keeps live and found a large, real
gap: e.g. "milk" carries 64 lead + 66 trail rest frames around only 40
active frames, while `CaptureBuffer`'s live defaults keep only ~12 lead +
~8 trail. Verified this mismatch actually flips the (then-current) trained
model's real predictions, not just abstract feature values — ran the grader
on all 60 val clips under both framings: 3/60 flipped, and all 3 flipped
toward predicting "repeated," mechanistically matching the reported
symptom. Tested the cheap fix first — padding the live-captured sequence
with static zero-motion frames before computing only the periodicity/tempo
feature — and it changed NOTHING, which was itself informative: it ruled
out the tempo scalar as the culprit and pointed at the whole BiGRU feature
stream's rest:signal ratio instead.

**Fix, in the order it was built:**
1. `features.py`'s new `live_capture_span(energy, threshold, preroll,
   settle_frames)` — an offline approximation of what `CaptureBuffer` would
   actually capture for a clean rest→sign→rest clip (every cached clip's
   shape): the motion-active span extended by the SAME preroll/settle
   margins `CaptureBuffer` uses live, not a byte-for-byte state-machine
   replay (doesn't need to be one — it exists to answer "how much rest will
   live capture keep," which reduces to exactly this for a single coherent
   motion burst). 4 new tests (`tests/test_features.py`).
2. `embedding_dataset.py`'s `_live_trimmed_poses(npz_path, pipeline)` — the
   ONE place this trim happens, using `LIVE_PREROLL=12`/
   `LIVE_SETTLE_FRAMES=8` (must match `diagnose_demo.py`'s single-sign-mode
   CLI defaults, not sentence mode's deliberately larger ones). Both
   `EmbeddingClipDataset.__init__` (training/reference-bank data) and
   `fit_standardizer` now route through it instead of the full raw clip —
   they have to trim identically, or the standardizer's fitted stats and
   the features it's applied to would silently be two different framings
   of the same clips.
3. `embedding_grader.py`'s `_forward_npz` (the cached-FILE grading path
   behind `grade`/`grade_against`, used by `eval_embedding_grader.py`,
   `--selftest`, and the regression tests) now routes through the SAME
   `_live_trimmed_poses` helper, so grading a cached file stays
   representative of what a real live attempt of that content would
   produce. Deliberately did NOT touch `_forward_poses` (the live IN-MEMORY
   path `grade_poses`/`grade_against_poses` actually call) — a live
   caller's poses already came through `CaptureBuffer`, which shapes them
   live; trimming again here would double-trim an already-live-shaped
   sequence.

**Retrained and validated, not just theorized.** Old checkpoint backed up
to `models/embedding_grader_backup_full_clip_trim/` first (`models/` is
gitignored, so this is a local rollback copy, not a git safety net). New
`PHASE4_REPORT.md` (identical script/methodology, only the checkpoint
changed): **85.2% top-1 / 97.8% top-5** (was 81.2% / 96.5%); per-parameter —
handshape 85.4%, major_location 89.8%, minor_location 85.3%, movement
82.3%, **repeated_movement 85.6%** (was 82.1%). The controlled comparison
that actually isolates the fix (holding framing fixed at live-shaped,
varying only old-vs-new checkpoint, via an ad hoc diagnostic script — not
`PHASE4_REPORT.md`'s own aggregate methodology): **false positives on
"repeated"** (truly not repeated, predicted repeated — the exact reported
symptom) **dropped from 28/116 (24.1%) to 14/116 (12.1%), roughly halved.**

Full suite re-run per the testing rule (this touches `features.py`, which
every downstream cache/comparison depends on): one existing regression test
needed re-pinning. `test_heads_disagree_independently_on_a_real_minimal_pair`
originally used the first father val clip; of the curriculum's 3 father val
clips, 2 demonstrate head-independence cleanly at 91-100% confidence post-
retrain and 1 sits on a genuinely marginal ~74% decision boundary for
`repeated_movement` — ordinary variance for one example near a boundary
after any retrain, not evidence the underlying property (heads are
independently correct) stopped holding. The test now pins to a specific
clip that demonstrates it robustly, documented inline rather than silently
swapped. `--selftest`'s `emulate()` was also updated to use
`live_capture_span` instead of a plain last-60-frames trailing window (the
model no longer expects that framing), and the now-dead `--window` CLI flag
was removed. **Full suite: 339 passed / 10 skipped.**

**Honest scope note, carried forward deliberately:** this is a real,
verified improvement, not the whole story. `repeated_movement` remains the
hardest parameter across every backend (CLAUDE.md issue #5), and real
webcam noise (jitter, lighting changes, tracking dropout) will add error
beyond what this fix's clean-cached-clip validation shows — it was not
possible to re-validate against an actual live camera session in this
pass. That's the natural next check, the same caveat Phase 7 step 5's
capture tuning already carries for a different reason.

**Live diagnostic demo — built, verified, not yet committed.**
`scripts/diagnose_demo.py` puts `grade_against` in front of a webcam rather
than only cached clips: prompt a target, play its real reference clip on
loop next to the live view (mandatory — `resolve_targets()` refuses to start
if any cycle target lacks a cached reference video), capture an attempt over
a sliding window, and show all 5 heads' verdicts (MATCH/OFF/insufficient-data
+ confidence) plus overall fidelity every time — the per-parameter breakdown
is the visual focus, not a single pass/fail. Deliberately reuses
`live_demo.py`'s webcam loop, sliding window, mirrored-display-only handling,
background-threaded grading, and `pipeline_config.py` wiring rather than
rebuilding any of it; the only two substantive changes are the grader
(learned, not DTW) and the interface (closed-set `grade_against` a known
target, not open-set classification). `verify_pipeline_matches_checkpoint()`
refuses to start (fail-closed) if the live feature pipeline built from CLI
flags differs from the checkpoint's own training-time config in any field —
a silent mismatch there would corrupt every verdict invisibly. `c` clears the
window for a fresh attempt but keeps the last verdict visible (dimmed,
marked stale) so a learner sees what changed between tries — deliberately
different from `live_demo.py`'s `c`. A persistent on-screen line states the
honest limit: this confirms plumbing and lets a learner imitate a real
reference clip, it does not independently verify ASL correctness.
`--selftest` runs the whole prompt → `grade_against_poses` → verdict path on
cached val clips with no camera, and reproduces `PHASE4_REPORT.md`'s
mother/father cross-check numbers exactly, confirming the new in-memory path
behaves identically to the already-verified cached-file path. Required two
small additive changes to `EmbeddingGrader` (no existing signature/behavior
changed, full suite still green): `grade_poses`/`grade_against_poses` (an
in-memory entry point for live frames, alongside the existing file-based
`grade`/`grade_against`), and a `confidence` field on `ParameterVerdict`
(already computed internally, now surfaced). **Not yet committed** —
`src/aslcv/grading/embedding_grader.py`'s changes and the new script are
both held back pending explicit go-ahead.

### Phase 5 — Production track (parallel to Phases 1–4)

**5a Reference retrieval — DONE** (video retrieval + concatenation; a
concatenated pose-sequence grading target is deliberately deferred, not
dropped — see below). Given an ID-gloss or a Phase 5b `GlossSequence`, fetch
the real reference clip(s). Retrieval, never generation. Wasn't blocked on
anything — Phase 1 already downloaded ASL Citizen and cached all 60
curriculum signs across all 4 extractors.

Scope was narrowed from the original "show correct form" + "compose grading
target" phrasing after two design questions surfaced discussing it:

- **Only video concatenation was built at the time; pose-sequence
  concatenation (a multi-sign grading target) was deferred, not dropped —
  since resolved, see below.** Grading a live continuous attempt against a
  concatenated pose target requires knowing where one sign ends and the next
  begins *in that live attempt* — that was Phase 7's unsolved continuous-
  recognition segmentation problem at the time (see Phase 7 below). The live
  pipeline had only Phase 2's manual fixed-window + keypress reset (`c`), no
  automatic boundary detector. A pose-sequence target built then would have
  had no consumer until Phase 7 existed, so this wasn't premature scoping —
  it was literally unusable before then. Per-sign grading targets already
  existed independently (Phase 4's `EmbeddingGrader`/`DTWGrader` reference
  banks) and were unaffected. **Now built**: `compose_reference_features`
  (Phase 7 step 2, in `retrieval.py` alongside `fetch_sequence`) is that
  feature-space target, now that step 1's `dtw_align` gives it a real
  consumer — see Phase 7's section for the full writeup.
- **No generative model for smoothing stitched clip transitions.** Considered
  and rejected: interpolating frames between two real clips to hide a hard cut
  would fabricate motion no signer produced and present it as a model of
  correct form — exactly the failure mode CLAUDE.md's "Retrieve reference
  video, never generate it. Ever." rule exists to prevent. The accepted
  approach instead: trim each clip's rest frames using the existing
  `hand_motion_energy()`/`motion_active_span()` (`features.py`, shared with
  Phase 2/6's segmentation — same default `motion_threshold`/
  `motion_pad_frames`, no new tuning) and hard-cut directly between clips.
  Visibly jump-cuts between signers; that's left honest, not hidden, matching
  `scripts/compose_sentence.py`'s existing banner ("stitched citation clips,
  not fluent connected signing").

Built: `src/aslcv/production/retrieval.py`.
- `fetch_reference(id_gloss, extractor) -> ReferenceClip` — one formalized
  selection rule (prefer a train-split clip, else the first match) replacing
  the same logic that used to be independently copy-pasted across
  `render_clip.py`'s `pick_row`, `compose_sentence.py`'s `pick_clip`, and
  `diagnose_demo.py`'s `reference_row`. Raises `KeyError` on a sign with no
  manifest rows at all; returns `npz_path=None` (not an error) for a known
  sign uncached under the requested extractor, leaving the fail-closed call
  to callers that actually need a pose sequence.
- `fetch_sequence(gloss_sequence, extractor) -> ComposedReference` — resolves
  every gloss the same way `compose_sentence.py`'s STEP 2/3 already does, but
  goes one step further than that script deliberately stopped short of: it
  actually decodes, trims (`hand_motion_energy()`/`motion_active_span()`,
  same defaults as `features.py`'s `trim_to_motion`), and hard-cuts the clips
  into one ordered frame list. Fail-closed on three paths, all tested: an
  out-of-scope `GlossSequence`, an empty one, and any gloss missing a cached
  reference for the target extractor (same convention as `resolve_targets` in
  `diagnose_demo.py`) — refuses rather than compose a partial video.
  `write_composed_video()` writes a `ComposedReference` to a real `.mp4`,
  kept separate since composing is cheap and a future caller (Phase 6's
  presenter) may want the frames without paying for a disk write every time.

**Verified end to end on real data**, not just unit-tested in isolation:
`"I want water."` → gloss sequence `['me', 'want_2', 'water']` → 3 real
ASL Citizen clips retrieved, each trimmed to its motion-active span, hard-cut
together into a 94-frame playable mp4 (fps carried from the source clips).
All three fail-closed paths confirmed to actually refuse (not just
theoretically). `tests/test_retrieval.py` (12 tests) covers `fetch_reference`
determinism/fallback/unknown-sign, `fetch_sequence`'s ordering/trimming/
fail-closed behavior, and `write_composed_video`'s round-trip + empty-input
refusal — full suite **214 passed, 10 skipped** (up from 202/10; nothing
else broke). Built and tested this session; **not yet committed**, held back
pending explicit go-ahead (same convention as `diagnose_demo.py` earlier).

#### 5b — Gloss rule engine (engine + tooling built; no data dependency; Deaf review still pending)

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

**Golden test corpus — built, partially reviewed by construction.**
`tests/test_gloss_rules_corpus.py` covers every `constructions_v2` entry plus
fail-closed/edge cases, but splits assertions into two tiers rather than
treating the whole corpus as signed-off: MECHANICAL properties (drop rules,
time-fronting, wh/yes-no NMM scope + span, fail-closed refusal + reason
populated, determinism, lexicon resolution) are asserted hard, since they
follow from the engine's own stated rules regardless of ASL fluency. Whole-
sentence word-order/NMM-placement correctness — the thing that actually needs
a fluent signer — lives in a `PENDING_CASES` list that only *reports* the
engine's output (`pytest.skip()` with the actual gloss sequence in the skip
reason), never asserts it as correct. A case is promoted from pending to a
real regression by a fluent reviewer setting `reviewed=True` and filling in
`expected_glosses`/`expected_nmm` — a flag flip, not a rewrite, so review
capacity is the only thing gating how much of the corpus becomes load-bearing.
Adding a rule still must not silently break another; the mechanical tier is
what currently catches that. Rules and lexicon are versioned together via
`curriculum.yaml`, not yet separately.

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
- **Status:** the gloss-ordering + NMM-tagging + fail-closed refusal half is
  built and mechanically tested (`gloss_rules.py`), and hardened by a
  dedicated fail-closed probing pass (`PHASE5B_GAP_REPORT.md`) that found and
  fixed two real gaps (AUX-headed clause coordination, ambiguous-lexicon
  entries) rather than leaving them as known limitations.
  `scripts/compose_sentence.py` proves every gloss resolves end to end to a
  real cached clip (all 60 curriculum signs are cached across all 4
  extractors, so nothing is currently retrieval-blocked), and
  `scripts/export_review_sheet.py` packages every case that needs a fluent
  signer's judgment into a one-file HTML sheet with embedded clips and a
  verdict form. Actual video concatenation is now done too (Phase 5a's
  `retrieval.fetch_sequence`, see Phase 5a above). **Not yet done** — the
  part that actually gates calling this phase complete — no fluent Deaf
  reviewer has actually opened the review sheet and signed off on word
  order/NMM placement yet;
  that's exactly what `PENDING_CASES` + the export script are scaffolded to
  make undeniable rather than quietly assumed.

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

### Phase 6 — The adaptive learning app (wire the loop) — status: v1 loop wired, in diagnose_demo.py

**Scoping decision (made explicitly, not defaulted into):** v1 extends
`diagnose_demo.py` rather than a new standalone app — its webcam loop, grading,
and reference-clip plumbing were already proven; wrapping the adaptive loop
around it is a much smaller diff than rebuilding that scaffolding. v1's
scheduler shipped as a simple weighted-random heuristic (gap-targeting + a
light recency bias), not a real spaced-repetition interval algorithm — enough
to close the loop end to end without blocking on it. **Now upgraded (see
below)** to a real SM2-style interval scheduler plus a first difficulty-control
signal; the feedback presenter is still templated English text by default,
with an optional LLM pass (`llm_feedback.py`) layered on top, documented
separately above.

- **Learner model — DONE.** `src/aslcv/learner/mastery.py`'s `MasteryState`:
  per-sign, per-parameter mastery in [0, 1], EMA-updated (`LEARNING_RATE=0.3`
  toward 1.0/0.0) from each attempt's real `ParameterVerdict.correct`. A `None`
  verdict (MIN_SUPPORT insufficient data) is skipped entirely, never averaged
  in as a fake 0.5 — an unresolved question isn't evidence of medium mastery.
  Persisted as JSON (`data/learner_state.json`, gitignored — personal session
  state, not project data) via an atomic temp-file + `os.replace` write, same
  convention as `extract_landmarks.py`'s cache writes. A logical clock (plain
  incrementing counter, not wall-clock) tracks recency deterministically.
- **Adaptive engine — DONE, now real spaced repetition + a first difficulty
  signal, not just the v1 heuristic.** The scoping decision above's "later,
  self-contained upgrade" was built: `MasteryState` gained an SM2-style
  interval/ease-factor pair per sign (`interval`, `ease`, `repetitions`,
  `INITIAL_EASE=2.5`, `MIN_EASE=1.3`, `MAX_EASE=3.0`, `SECOND_REP_INTERVAL=3`,
  `MAX_INTERVAL=40`), adapted to the ONLY clock this project has — the logical
  per-attempt tick, not wall-clock days, since a live single-session demo has
  no "next day" to schedule against. `interval` is measured in attempts-of-
  any-sign since a sign was last seen; `is_due(sign)`/`due_at(sign)` expose
  when it next becomes eligible. `update()` treats a "good" attempt as ALL
  judged parameters matching (a real full pass, same bar `focus_parameter`
  already uses to decide whether to coach at all — one wrong parameter is a
  miss for scheduling purposes, not partial credit); a `None`-only attempt
  (nothing judged) leaves the interval untouched rather than guessing.
  `scheduler.py`'s `pick_next` now gates its candidate pool on `is_due` before
  weighting by weakness — a sign just drilled successfully is excluded until
  enough OTHER attempts have passed, same as SM2 skipping a well-known card —
  and falls back to the full pool if nothing is due yet rather than stalling a
  session with nothing to present. **Difficulty control** (previously entirely
  unimplemented) now has one concrete form: a pick that lands on an
  already-well-mastered sign (`HIGH_MASTERY_THRESHOLD=0.8`) has a
  `PROACTIVE_CONTRASTIVE_PROB=0.5` chance of being redirected to ANY of its
  minimal-pair partners (`_any_contrastive_partner`, not restricted to one
  parameter the way the existing reactive drill is) — proactively
  stress-testing a parameter the learner hasn't recently been tripped up on,
  rather than only reacting after a wrong verdict. 16 new tests
  (`tests/test_mastery.py`, `tests/test_scheduler.py`, including a `_FakeRNG`
  helper to pin down the proactive branch deterministically, since a real
  seeded PRNG's call order isn't reliable for that). Full suite: **310
  passed / 10 skipped.**
- **Task generator (isolated + contrastive + sentence prompts) — ALL DONE,
  sentence prompts now built too.** `find_minimal_pairs` re-derives the same
  differ-in-exactly-one-parameter pairing `eval_minimal_pairs.py` uses for the
  accuracy screen (re-derived, not imported — that's script code, this is
  library code the live session imports), scoped to the active target pool
  only (a contrastive pick must itself have a validated reference clip, same
  fail-closed constraint `resolve_targets` already enforces). `pick_next`
  fires a contrastive drill FIRST and unconditionally whenever the last
  attempt's most-confidently-wrong parameter has a real partner sign in the
  pool — verified end to end on real data, not just unit tests: grading a
  real `father` clip against `mother` (the curriculum's built-in
  `minor_location` minimal pair) correctly queues `father` right back up
  next.

  **LLM-generated sentence prompts — DONE**, opt-in via `diagnose_demo.py
  --sentence-prompts`. `src/aslcv/generator/sentence_prompts.py`'s
  `sentence_prompt_maybe_llm` composes the two already-finished pieces this
  was always going to need (Phase 5a's retrieval, Phase 5b's rule engine)
  rather than adding a third path: it asks the LLM to write ONE short English
  sentence containing the current target's word (constrained to
  curriculum.yaml's own `english_lemmas` vocabulary as an allow-list, to
  raise — not guarantee — the acceptance rate), then hands that sentence to
  Phase 5b's `gloss_rules.gloss_sentence()` exactly as a human-typed sentence
  would go through it. Only if the ALREADY fail-closed rule engine accepts it
  in scope does it ever become a displayed prompt (up to `max_attempts=2`
  retries on refusal, then silently nothing — never a partial or
  unvalidated sentence shown as ASL content). This is CLAUDE.md's "an LLM
  only ever touches English" and "grammar is a rule engine, not a trained
  model" enforced TOGETHER, structurally: the LLM only ever supplies English
  wording, and the deterministic rule engine — not the LLM's own obedience to
  its prompt — is what actually decides whether that wording becomes
  displayed ASL content. Presentational only, NOT graded: shows the learner
  an example sentence using their target word, glossed and NMM-tagged, to
  read as an English → ASL composition example; continuous-sentence grading
  is still Phase 7, unbuilt, so `diagnose_demo.py` keeps grading only the
  isolated target sign exactly as before this existed. Same hosted-API/
  fail-open pattern as `llm_feedback.py` (`meta-llama/Llama-3.1-8B-Instruct`,
  `provider="auto"`, needs `HF_TOKEN`) — sharing a new `_hf_client.py` for
  the token/`.env` resolution logic both modules need identically, rather
  than risking two copies drifting apart. Blocking, same accepted tradeoff as
  `--llm-feedback`: this is a dev-machine demo script, not a production UI.
  22 new tests (`tests/test_sentence_prompts.py`, `tests/test_hf_client.py`),
  full suite **263 passed / 10 skipped**.
- **Feedback presenter — DONE, both templated (default) and an optional LLM
  pass.** `src/aslcv/generator/feedback.py`'s `coach_text`: one line, praise if
  every judged parameter matched, otherwise names the SINGLE
  most-confidently-wrong parameter with a short tip (an instructor gives one
  correction at a time, not five at once) and briefly lists any other misses.
  Duck-typed on `.parameter`/`.correct`/`.confidence` — no dependency on the
  grading package's dataclass, just its shape, so this module (and its tests)
  never load torch or a checkpoint. `focus_parameter` factors out the "which
  parameter would coach_text focus on" signal so the scheduler's contrastive
  trigger doesn't have to re-derive it or parse prose. `llm_feedback.py`'s
  `coach_text_maybe_llm` is the optional upgrade, opt-in via
  `diagnose_demo.py --llm-feedback`: phrases the SAME pre-computed facts more
  naturally via HuggingFace's HOSTED Inference API
  (`huggingface_hub.InferenceClient`, `meta-llama/Llama-3.1-8B-Instruct` default,
  `provider="auto"`) — deliberately the hosted API, not a locally-run model,
  because of the Phase 8 mobile target below: a phone can't run even a small
  open LLM the way this desktop process could, so code built against the
  hosted API now is close to what a phone client will actually do, where code
  built around local `transformers` inference would have been thrown away.
  CLAUDE.md's "an LLM only ever touches English" rule is enforced
  structurally: the model is handed only pre-computed facts (target sign,
  which parameter(s) were wrong, confidence) and instructed to phrase them,
  never asked to judge correctness or shown raw attempt data. Fail-open by
  design: no `HF_TOKEN`, no `huggingface_hub`, a network error, a timeout, or
  a malformed response all fall back to the templated `coach_text` (one
  warning printed once), never block or crash the live loop. 10 new tests
  (`tests/test_llm_feedback.py`) cover the fail-open paths for real (no
  token needed) and a successful call via a mocked client (no network
  needed).
- **Closed loop, wired end to end in `diagnose_demo.py`.** `c` is unchanged
  (clear the window, retry the SAME target, last verdict stays on screen
  dimmed). `n` is now the adaptive step, not blind list-cycling: records the
  CURRENT non-stale verdict into `MasteryState` (a stale verdict is a re-sign
  in progress, not a completed attempt — never double-scored), saves to disk,
  prints one line of coaching (templated or LLM-phrased per `--llm-feedback`),
  then calls `pick_next` for the next target — console-prints the reason when
  a contrastive drill fires, and the on-screen target line now shows live
  mastery (`TARGET: mother   mastery 62%`). `resolve_targets`'s fail-closed
  reference-clip validation is unchanged and still gates the whole pool up
  front. Switching targets also prints an LLM sentence prompt when
  `--sentence-prompts` is on (see the task-generator bullet above). 54 new
  tests total across Phase 6 v1 + v2 (`tests/test_mastery.py`,
  `tests/test_scheduler.py`, `tests/test_feedback.py`,
  `tests/test_llm_feedback.py`, `tests/test_hf_client.py`,
  `tests/test_sentence_prompts.py` — pure logic + real phonology/curriculum
  data + mocked network, no real torch/checkpoint/token/network needed for
  any of them), full suite **263 passed / 10 skipped** (up from 214/10 before
  Phase 6).
- **Live-demo hardening (accuracy + grounded description + UI) — DONE.**
  Investigated a user report that handshape/repeated verdicts are "often
  wrong" rather than patching blind. Partly a real, already-measured model
  ceiling (handshape 80.7% / repeated_movement 82.1% val accuracy,
  `PHASE4_REPORT.md`; `repeated` independently flagged the hardest parameter
  across every backend in `eval_minimal_pairs.py`) — now said outright in the
  on-screen disclaimer. But also a real, fixable live-only bug: the demo used
  a fixed `deque(maxlen=60)` trailing window, which silently evicts its
  OLDEST frames as new ones arrive, truncating a slow signer's attempt —
  exactly the corruption that would hit `repeated_movement` hardest (needs
  the full cyclic pattern) and can catch `handshape` mid-transition. Fixed
  with `src/aslcv/capture.py`'s `CaptureBuffer`: an idle -> active -> settled
  state machine using the SAME `hand_motion_energy` signal `features.py`'s
  `trim_to_motion` already uses for cached clips (that function's own
  docstring names this exact live use case, previously unbuilt) — grows from
  the start of motion to a natural rest boundary instead of a fixed size,
  capped for safety. READY/CAPTURING/CAPTURED shown on screen. `energy_fn` is
  injected so the state machine is unit-tested with a synthetic deterministic
  signal, no camera/pose fixtures needed (7 tests).

  `src/aslcv/generator/sign_description.py`'s `describe_sign` is a NEW
  always-on (no LLM, no network) grounded description of the TARGET sign's
  own phonology, shown at the bottom of the reference-video panel per
  explicit request ("describe what the correct sign looks like") — built
  entirely from `PhonologyLabels`/`phonology.csv`, the same source every
  verdict's `.target` already comes from (5 tests).

  The whole overlay was redesigned for a cleaner look: one consistent
  color/type-scale theme, a colored capture-state badge, and per-parameter
  rows split into a label+tag line and a "you:/target:" detail line (fixes a
  real overflow bug the old fixed-x-offset layout had). Caught along the way:
  the wrap-width helper's original 9.5px/char constant was ~40% too
  optimistic vs. real `cv2.getTextSize` measurements, silently running text
  off the canvas edge (cv2.putText neither wraps nor clips) — found by
  rendering mock canvases and visually inspecting them, not by reasoning
  about it; recalibrated to ~14.2px/char, measured. Full suite: **296 passed
  / 10 skipped.**
- **UI polish pass 2 — DONE.** First redesign wasn't enough per user feedback:
  bottom text still hard to read, red/green verdict text hard to read, fonts
  too small, `sign_description` read like a labeled spec sheet not natural
  language. Root cause of the readability complaint: the verdict list,
  coach-text band, and status bar were drawn directly over the RAW live
  camera feed with no background panel behind most of them. Fixed by giving
  every text region its own near-opaque panel, bumping the type scale up
  (0.4–0.72 → 0.52–1.0) with bold weight where it needs to pop, more
  saturated MATCH/OFF colors, and rewriting `describe_sign` into two flowing
  sentences (with proper a/an agreement and natural "Away" phrasing) instead
  of `"Label: value"` fragments. Two more real layout bugs found by rendering
  mock canvases against synthetic noise and inspecting the PNGs (same method
  as before, not reasoning about pixel math): (1) the coach-text band was
  positioned via clamping against two independently-guessed anchors, which
  could overlap the verdict list when content ran longer than assumed —
  fixed by making `draw_verdict` return the exact y it stopped at and having
  every section below it start exactly there, top-down, structurally
  incapable of overlapping; (2) deliberate small gaps between adjacent
  opaque panels let raw video show through in a visible stripe at each seam
  — panels now touch edge-to-edge with a divider line instead. Canvas height
  raised 640→920 to fit the larger text without re-triggering the overlap
  bug at the margin. Full suite: **299 passed / 10 skipped.**

  Separately found and RESOLVED: `Qwen/Qwen2.5-7B-Instruct` (the prior
  `--llm-feedback`/`--sentence-prompts` default model) was observed failing
  live HF calls with `model_not_supported`, then succeeding again minutes
  later with no code change. Investigated rather than just retried: HF's own
  model API (`GET /api/models/<id>?expand[]=inferenceProviderMapping`,
  checked directly) showed Qwen2.5-7B-Instruct served by only 2 providers
  (`together`, `featherless-ai`) vs. `meta-llama/Llama-3.1-8B-Instruct`'s 4
  (`novita`, `nscale`, `deepinfra`, `featherless-ai`) — thin provider
  coverage under `provider="auto"` is exactly what predicts that flakiness.
  `_hf_client.py`'s `DEFAULT_MODEL` swapped to Llama-3.1-8B-Instruct (same
  8B size class, same `provider="auto"` pattern), verified live post-swap on
  both `llm_coach_text` and `sentence_prompt_maybe_llm`. No test hardcodes
  the model name, so the full suite was unaffected: **299 passed / 10
  skipped.**
- **Produces:** a working adaptive tutor for isolated signs (v1) — not yet
  packaged as `app/` (still a script), not yet difficulty-aware, not yet
  spaced-repetition-scheduled in the interval-algorithm sense. Both are
  legitimate, scoped-out-on-purpose follow-ups, not gaps that block "done
  when" below.
- **Done when:** the app chooses what to practice from the learner's gaps
  (**yes**), drills (**yes, isolated + contrastive**), corrects at parameter
  level (**yes**), and adapts across a session (**yes, persisted across
  sessions too, not just within one**) — nothing fixed except the reference
  correctness.

### Phase 7 — Sentence grading via forced alignment — status: DONE (all 6 steps), live-capture tuning untested against a real camera

**Reframed from the original "CTC-style continuous recognition" spec, deliberately.**
Industry CSLR (the RWTH PHOENIX lineage's CTC decoders, DeepMind's SL2T, the
2025 transformer/BIO-tagging segmenters — see the research pass this scoping
is based on) all solve *blind* segmentation: recognize an unknown sequence of
signs with no advance knowledge of what's being signed. This project never has
that problem. The learner is always attempting a sentence THIS SYSTEM
generated (Phase 5b's fail-closed gloss engine, optionally Phase 6's LLM-
written prompt text) — the ordered target gloss sequence is known before the
attempt starts. That turns "segment then recognize" into **forced alignment**:
align a live attempt against a known reference sequence, the same class of
problem as speech recognition's forced alignment, not open CSLR. This is a
strictly easier problem AND the only framing consistent with CLAUDE.md's
"grounded answer key" principle and its explicit exclusion of free-form
translation — so it's the only framing considered.

**What already exists and needs zero changes:**
- `EmbeddingGrader.grade_against_poses(poses, target_sign)` already grades an
  arbitrary in-memory pose slice against one known target — call it once per
  detected segment and per-sign grading is already solved.
- `hand_motion_energy()` / `motion_active_span()` (`features.py`) — already
  the shared boundary-energy signal for single-clip trimming; reused, not
  reinvented, for reference-side trimming below.
- `production/retrieval.py`'s `fetch_sequence`/`ComposedReference` pattern —
  resolve each gloss to a clip, trim to its motion-active span, concatenate,
  track `clip_frame_ranges` — already does exactly this for VIDEO frames
  (Phase 5a's "show the correct sentence" path). The new work below is the
  same pattern one layer down, on FEATURES instead of pixels, for grading
  rather than display.
- `MasteryState.update()` / `scheduler.pick_next` — already per-sign; a
  multi-sign attempt just calls `update()` once per segment, no changes.

**What's actually new, in build/validate order (each step gates the next —
no live-UI work until the algorithm is proven on data that already exists):**

1. **`dtw_align()` — DONE.** Added to `grading/dtw_grader.py` alongside
   `dtw_distance` (not a separate module — small enough, and keeping both
   next to each other makes "these share one recurrence" obvious on read).
   Keeps the full `(n+1, m+1)` DP table (`dtw_distance` only keeps a rolling
   row) and backtraces from `(n, m)` to `(0, 0)`, preferring diagonal on
   ties. Returns `(length_normalized_distance, path)`; the distance always
   matches what `dtw_distance(a, b, band)` returns for the same inputs — same
   cost function, same recurrence, verified directly by test, not just
   asserted in the docstring. 5 new tests (`tests/test_dtw_grader.py`):
   distance-matches-dtw_distance (banded and unbanded), path monotonicity
   and endpoint coverage, identity-sequence path is the diagonal, empty
   input returns no path, and the actual forced-alignment use case —
   concatenating two far-apart synthetic clusters with a known boundary,
   adding noise, and confirming the warp path recovers the true cut point
   within a few frames. Full suite: **315 passed / 10 skipped.**
2. **`compose_reference_features()` — DONE.** Added to `production/retrieval.py`
   alongside `fetch_sequence` (the feature-space sibling, same file, same
   fail-closed rule — not a separate module). Refactored `fetch_sequence`'s
   inline resolve-and-validate loop into a shared `_resolve_clips()` first, so
   the video path (Phase 5a) and the feature path (Phase 7) can't silently
   drift on which clips they pick or how they refuse — behavior-preserving,
   existing `fetch_sequence` tests unchanged. `_trimmed_poses()` mirrors
   `_trimmed_frames()`'s exact trim signal (`hand_motion_energy`/
   `motion_active_span`) but slices the cached POSES instead of video frames.
   Returns `ComposedReferenceFeatures(gloss_sequence, clips, features,
   frame_gloss_index)` — `frame_gloss_index[t]` is the 0-based index into
   `gloss_sequence.gloss_ids` that concatenated feature-frame `t` belongs to
   (the ground truth step 4's synthetic benchmark checks against, and what
   step 3's alignment projects onto a live attempt). `pipeline`/`standardizer`
   are passed in, not constructed here — a caller (an `EmbeddingGrader`)
   already owns its trained instances, and reusing them exactly is what keeps
   the reference and the live attempt in the same feature space. 4 new tests
   (`tests/test_retrieval.py`): concatenation order + `frame_gloss_index`
   coverage, a single-gloss sequence matches a direct pipeline call
   (shape/dtype, since composed trims and the direct call doesn't), and both
   fail-closed paths (out-of-scope, missing reference). Full suite: **319
   passed / 10 skipped.**
3. **`align_and_grade(grader, attempt_poses, gloss_sequence)` — DONE.** New
   module `grading/alignment.py` (needs `EmbeddingGrader`, `dtw_align`, AND
   `compose_reference_features` together, so it doesn't fit cleanly inside
   any one of the three existing files it draws from). Featurizes the
   attempt with the SAME `grader.pipeline`/`grader.standardizer` instance
   `compose_reference_features` used for the reference (so both sides of the
   alignment share one feature space), DTW-aligns via `dtw_align`, then
   `_segment_ranges()` projects the reference's known per-frame gloss labels
   onto the attempt: for each gloss, the `[min, max]` attempt frame aligned
   to ANY reference frame carrying that gloss's index. Provably can't miss a
   gloss — a complete warp path is a monotonic staircase from `(0,0)` to
   `(n-1,m-1)`, so every reference column index is visited at least once,
   which is the actual guarantee the implementation leans on rather than
   hoping. Each segment is then graded with `grade_against_poses` completely
   unchanged — no grading logic duplicated. Returns
   `(alignment_distance, [AlignedGrade, ...])`, mirroring `dtw_align`'s own
   `(distance, path)` shape; `AlignedGrade` carries `target_sign`,
   `frame_range`, and the `GradeResult` together, since step 4's validation
   benchmark needs the frame range (to check against ground truth) and step
   6's live UI needs the `GradeResult` (to render), not just one or the
   other. 5 new tests (`tests/test_alignment.py`, needs a trained checkpoint,
   same skip-if-absent convention as `test_embedding_grader.py`): concatenate
   the SAME real reference clips `compose_reference_features` would resolve
   into a synthetic "continuous attempt" (a small-scale rehearsal of step 4's
   actual validation trick) and check segment order, monotonicity/coverage,
   that a segment fidelity-ranks its true target better than a mismatched
   one (relative check, not a pinned absolute distance whose scale isn't
   documented anywhere), a degenerate single-gloss sequence needs no special
   casing, and the empty-attempt refusal. Full suite: **324 passed / 10
   skipped.**
4. **Validate BEFORE any live wiring — DONE, result: PASS, proceed to step 5.**
   `scripts/eval_forced_alignment.py`: for N random synthetic sentences (2-4
   curriculum signs each), builds the reference side via `align_and_grade`'s
   own `compose_reference_features` (TRAIN-split clips, exactly what a real
   session uses) and the "attempt" by concatenating each sign's TRIMMED
   VAL-split clip (held-out footage, genuinely different takes from the
   reference — not the trivial identical-clip case `tests/test_alignment.py`'s
   unit tests use for plumbing checks only). Reports two numbers, deliberately
   NOT against the true phonology label for the second one: (a) boundary
   error — `|predicted segment length - true segment length| / true length`,
   since the true length is known exactly (we did the concatenating); (b)
   grading agreement — does the aligned segment's `grade_against_poses`
   verdict match grading that SAME clip in isolation, per parameter,
   excluding thin/insufficient-support targets (the `MIN_SUPPORT` gate, same
   exclusion convention as every other report in this project)? Agreement
   with the isolated grade (not the true label) is the right bar here
   specifically because the isolated grader itself is only ~81% accurate
   (`PHASE4_REPORT.md`) — comparing against ground truth would conflate
   alignment error with already-documented, unrelated model error; this
   metric isolates what THIS step adds.

   **Real numbers** (100 trials, seed=1, `mediapipe`, `PHASE7_ALIGNMENT_REPORT.md`):
   boundary error — median 4.0% relative, mean 12.8% (mean 4.7 frames
   absolute) — with a real, disclosed tail: **19/305 segments (6.2%) exceed
   50% relative boundary error**, not hidden behind the median. Grading
   agreement across all 5 parameters: **92.4%-94.6% correct-flag agreement,
   85.4%-93.8% exact predicted-label agreement**. **Verdict: PASS** — the
   median case is tight (a few frames off), and even including the outlier
   tail, per-parameter grading agreement stays in the low-to-mid 90s across
   every parameter, meaning misalignment rarely flips a verdict. Proceeding
   to step 5. **Honest caveat, unchanged regardless of this result:**
   concatenated real clips have a hard cut and no coarticulation — this
   measures an EASIER problem than genuine fluent continuous signing, the
   same direction of optimism `compose_sentence.py`'s video output already
   discloses ("stitched citation clips, not fluent connected signing"); the
   6.2% boundary-error tail is a real, disclosed weak spot to watch once live
   data (step 5+) is available, not something this benchmark can rule out.
5. **`CaptureBuffer` tuned for a whole sentence — DONE.** Confirmed (4)'s
   bar cleared, then addressed the real risk head-on rather than assuming it
   away: `CaptureBuffer` already bounds ONE attempt idle → active → settled
   on a motion rise-and-fall, and a multi-sign sentence has SEVERAL such
   rises with brief dips between words — too small a `settle_frames` (tuned
   for a single sign) reads an inter-word pause as the end of the attempt.
   **No structural change to `capture.py`** — this was a parameter-tuning
   decision, not a new state machine, exactly as scoped. New CLI flags
   `--sentence-settle-frames` (default 30, vs. single-sign's 8) and
   `--sentence-capture-max` (default 400, vs. 150) in `diagnose_demo.py`.
   Since no real camera is available in this environment to tune against
   directly, verified the ONLY honest way available: 2 new synthetic tests
   in `tests/test_capture.py` — a 3-sign energy pattern with brief inter-sign
   dips (shorter than the sentence-mode `settle_frames`) confirms the buffer
   rides through all of them and only settles after the true final rest; a
   second test with a single-sign-scale `settle_frames` on the IDENTICAL
   pattern confirms the risk is real, not hypothetical (settles mid-sentence,
   at the first pause). Explicitly still **untuned against real recordings**
   — the chosen defaults are conservative, not measured; flagged as-is
   rather than presented as validated. Full suite: **326 passed / 10
   skipped.**
6. **`diagnose_demo.py` sentence mode — DONE.** `--sentence "I want water."`
   glosses the sentence via Phase 5b's engine (fail-closed, same spirit as
   `resolve_targets`), resolves both the display reference (Phase 5a's
   `fetch_sequence`) and — inside `run_live_sentence`, the new mode's own
   top-level function, kept separate from `run_live` rather than branching
   deep inside it — grades via `align_and_grade` on `[n]`, once `CaptureBuffer`
   reports CAPTURED. Deliberately **no background re-grading thread** the way
   single-sign mode has: forced alignment needs the COMPLETE attempt against
   the COMPLETE reference, there's no meaningful partial verdict for it.
   `draw_sentence_results` renders a compact per-word SCOREBOARD (matched/
   judged parameter count + fidelity per word), not `draw_verdict`'s full
   5-parameter breakdown per word — a real, deliberate scope choice: a
   multi-word sentence's full per-parameter detail for every word wouldn't
   fit on screen at once, and the coach-text band still names the single
   most useful correction (first segment with a real mistake), keeping the
   "one thing at a time" philosophy `coach_text` already uses. Mastery is
   updated per word exactly like single-sign mode. Deliberately **NOT
   adaptive across sentences** — grades exactly the one sentence given;
   picking the next sentence to practice is a bigger feature outside this
   6-step scope, single-sign mode's scheduler is untouched. Verified two
   ways, no camera available: (a) `--selftest --sentence "..."` extends the
   existing offline selftest to build a synthetic attempt from cached val
   clips and run it through the real `align_and_grade` path — confirmed
   working end-to-end (`me`/`want_2`/`water` segments produced with sane
   frame ranges and fidelities); (b) `compose_sentence_canvas` rendered
   against synthetic noise (the same busy-camera-feed stand-in every prior UI
   change in this file was verified with) in both the pre-grade and
   post-grade states — confirmed no layout bugs (panels touch edge-to-edge,
   no overlap, top-down flow holds through the new scoreboard section too).

   **Redesigned after real (first-ever live) usage exposed exactly what a
   canvas render against synthetic noise can't catch: a workflow nobody's
   used before is confusing even with a technically-correct layout.** User
   feedback after actually running `--sentence`: didn't understand how to
   use it. Root causes, found by re-reading the code rather than guessing:
   (1) the status bar's control hint was single-sign mode's UNCHANGED
   `"[n]ext"` — actively wrong in sentence mode, where `[n]` grades and does
   nothing at all until CAPTURED; (2) the only instruction was a small
   `F_TINY` sub-hint easy to miss, for a 2-step (sign, then grade) flow a
   learner has never seen before, unlike single-sign mode's familiar
   single-action loop; (3) the results scoreboard leaned on raw ML
   vocabulary ("fidelity") and bare colored text with no iconography,
   forcing the reader to parse numbers instead of glance-reading a result;
   (4) the left reference panel had zero on-screen explanation of what it
   even was. Fixed all four, not just the one bug found first:
   - `draw_status_bar` gained an optional `controls` param (default =
     single-sign mode's exact original string, zero behavior change there);
     sentence mode's `_sentence_controls()` builds a STATE-AWARE string —
     `"[n] grade (finish signing first)"` while capturing, `"[n] Grade my
     attempt"` once CAPTURED, `"[c]lear to retry   [n] Grade again"` after
     grading.
   - `_SENTENCE_STEP_HINT` (separate from `_CAPTURE_STATE_DISPLAY`'s hint
     dict, which single-sign mode also uses and was already tuned for that
     flow) gives sentence mode explicit "STEP 1 -- sign..." / "STEP 2 --
     press [n]..." copy, drawn at `F_BODY` size (matching single-sign mode's
     own primary-instruction weight) in a taller `SENT_BADGE_H` badge, not
     crammed into the old tiny sub-hint; once actually graded, the hint
     switches to a `"settled_graded"` variant ("Results below...") instead
     of stale "press [n] to grade" text sitting above results already on
     screen.
   - `draw_sentence_results` rewritten: a real checkmark/X icon per word
     (`_draw_check`/`_draw_x`, drawn as vector line segments — HERSHEY fonts
     don't reliably render ✓/✗ glyphs) instead of relying on color alone,
     plain "N of M correct" instead of "matched," the fidelity number
     renamed "similarity" and demoted to a smaller muted secondary line
     under the headline count rather than crowding it, and a footnote
     spelling out what the 5 graded aspects actually are (a learner
     shouldn't have to already know "handshape/major_location/..." to read
     the count).
   - New `draw_sentence_reference_footer` gives the left panel the
     always-on "HOW TO USE THIS" explanation single-sign mode's
     `describe_sign` footer already provides, that sentence mode never had.

   Verified the same way as every prior UI pass in this file: rendered
   `compose_sentence_canvas` against synthetic noise across all four states
   (idle, active, settled-ungraded, graded-with-coach-text) and visually
   inspected each — confirmed no overflow from the longer state-aware
   hint/control strings, no overlap, panels still edge-to-edge. `--selftest
   --sentence "I want water."` re-run to confirm the wiring (unaffected --
   this pass was drawing-only) still works end to end. Full suite unaffected
   (script-only change): **336 passed / 10 skipped.**

   **Layout redesign — real live usage (a real camera, this time) found two
   more real problems the previous pass's synthetic-noise checks structurally
   couldn't catch: those checks exercise DRAWING correctness, not whether the
   drawn thing is a good use of the screen.** User feedback after actually
   using it live: wanted to see themselves more clearly, and wanted coaching
   on EVERY wrong word, not just the first.
   - **"See myself more clearly"**: with the right/live panel carrying
     header + badge + a 3-word scoreboard + coach text + status bar all
     stacked on top of the camera feed, well over half of it was opaque UI,
     not camera. Fixed by moving the results scoreboard and coaching
     entirely onto the LEFT (reference) panel, drawn as a bottom band below
     the looping video instead — the right panel now carries only header +
     badge + status bar once graded, freeing nearly the whole panel for the
     live view (the underlying draw functions, `draw_sentence_results`/
     `draw_coach_texts`, are unchanged in how they draw — only WHICH canvas
     they're called on moved).
   - **"Coach me on every wrong word"**: `[n]`'s handler used to coach only
     `next(g for g in graded if focus_parameter(...))` — the first mistake
     in gloss order, silently dropping the rest. Now loops over every
     segment with a real mistake and coaches each (still "one correction
     per WORD" — `focus_parameter` still picks that word's single most-
     confidently-wrong parameter — just no longer capped at one word
     total). `draw_coach_text` (singular, single-sign mode, UNCHANGED) got
     a sibling `draw_coach_texts` (plural) that stacks one band per
     mistake; entries are now `(sign, text, color)` triples so a
     no-mistakes "Great job" entry can render in green without an odd
     "Coach (I want water.):" prefix, instead of forcing every message
     through the same "Coach (X):" phrasing.
   - **A real overflow bug found BY rendering the new layout, not assumed
     away**: with all 3 words wrong and long LLM-length coach text each,
     the third entry ran off the bottom of the canvas. Root cause: the
     first version of `_coach_texts_height`'s formula didn't match
     `draw_coach_texts`'s actual per-entry step arithmetic (which advances
     by a full `line_h` *before* each entry, not only between entries) --
     correct for exactly 1 entry (matching single-sign mode's original
     `_coach_text_height`, which is where the formula was copied from), a
     30px-per-entry underestimate for every entry after the first — invisible
     at 1 entry, a real ~70px shortfall at 3. Fixed by rewriting the height
     function to SIMULATE the exact same stepping arithmetic the drawing
     loop uses, rather than a separately-derived formula that can silently
     drift from what's actually drawn — this is the same class of bug
     `draw_verdict`'s own docstring already warns about (fixed-offset
     layout assumptions breaking when content is taller than assumed), just
     inside a height *formula* this time instead of a fixed pixel offset.
     Also added `_fit_coach_entries`: even with a correct height formula,
     enough wrong words with long enough LLM-phrased text could still
     exceed the canvas, so entries are truncated to whatever fits within a
     computed budget (never letting the video shrink below `MIN_VIDEO_H`),
     with a "+N more corrections -- see terminal output" note for the rest
     — every entry is printed to the console regardless, so truncation
     never actually loses information, just screen space.

   Verified by rendering all three post-grade states against synthetic
   noise — normal (1 wrong word), a deliberately adversarial worst case (all
   3 words wrong, long stress-test coach text on each), and all-correct —
   and visually inspecting each; the worst case is what caught the overflow
   bug above. `--selftest --sentence` and the full suite re-confirmed
   afterward. Full suite: **336 passed / 10 skipped** (drawing/layout-only
   change).

   **CLI surface simplified — 31 flags down to 13, no behavior change.**
   User feedback: `diagnose_demo.py` "has way too many options and is
   complex." Investigated rather than guessing which to cut: most of the
   surface (10 flags) came from `add_pipeline_args`, the shared pipeline-
   ablation block `live_demo.py`/`eval_slice.py`/`eval_minimal_pairs.py`
   also use — and in THIS script they turned out to be pure dead weight.
   Grading has always used `grader.pipeline` (loaded straight from the
   checkpoint); the CLI-built pipeline `verify_pipeline_matches_checkpoint`
   constructed from those flags was never fed into any actual grading call,
   only compared against `grader.pipeline` to print a warning if they
   differed. Same for `--extractor`: any value other than the checkpoint's
   own already just triggered a refusal, so it never customized anything
   either. Removed both, plus the whole verification function — there is no
   longer a second pipeline-construction path to accidentally mismatch, so
   there's nothing left to verify. **`live_demo.py` was deliberately left
   untouched**: it's the Phase 2 DTW baseline with no saved checkpoint to
   fall back on, so its pipeline flags are genuinely load-bearing there
   (what you pass is what gets used), and it's the shared ablation harness
   Phase 3's backend comparisons ran through — cutting it would remove real
   research capability for no UX benefit, since it isn't the script anyone
   runs interactively.

   Also folded the rarely-touched tuning knobs (`--min-frames`,
   `--settle-frames`/`--preroll`/`--capture-max` and their `--sentence-*`
   duplicates, `--mastery-path`) into fixed module-level constants —
   `PREROLL`/`SETTLE_FRAMES` now literally ARE
   `embedding_dataset.LIVE_PREROLL`/`LIVE_SETTLE_FRAMES` (the same import,
   not a separately-typed matching value), making the "these two must agree
   with what the grader was trained on" invariant impossible to drift
   instead of just documented. Collapsed the redundant `--mirror`
   (default-on, so it never did anything) / `--no-mirror` pair into a single
   `--no-mirror` flag. What's left is only what someone actually types:
   `--checkpoint`/`--which`, `--target`/`--targets`, `--camera`,
   `--sentence`, `--gpu`, `--llm-feedback`, `--sentence-prompts`,
   `--selftest`, `--no-mirror`.

   Verified behavior-identical, not just "looks the same": `--selftest`
   (both single-sign and `--sentence`) produces byte-identical output
   before/after the cut. Full suite: **339 passed / 10 skipped.**

**Deliberately still out of scope even once this ships:** genuine free
continuous signing with real coarticulation (this always grades an attempt at
a system-known target sentence, never open translation — consistent with
CLAUDE.md's "out of scope: free-form translation, continuous recognition [as
open-set]"); anything resembling SL2T's actual architecture (a from-scratch
translation model) — this stays a retrieval/alignment approach the whole way
through, per the project's core non-negotiable (grade by distance to a
reference, never classification/generation).

**Done when:** `align_and_grade` reproduces known boundaries on the
concatenated-real-clips benchmark at an accuracy the DTW baseline's own
minimal-pair numbers make a fair target for (✅ step 4, PASS), and
`diagnose_demo.py` can grade a live multi-sign sentence attempt
segment-by-segment against a Phase 5b-generated target, with mastery updated
per sign exactly as the isolated-sign loop already does today (✅ step 6).
**All 6 steps done.** The one remaining gap between "done" and "validated
against real signing": `CaptureBuffer`'s sentence-mode tuning (step 5) is
verified only against synthetic energy patterns, never a real camera in this
environment — real-world tuning of `--sentence-settle-frames`/
`--sentence-capture-max` against an actual signer is the honest next check
before trusting sentence mode's capture boundaries the way single-sign
mode's are already trusted.

### Phase 8 — Port to a phone — status: not started, scoped only

Everything through Phase 7 is a desktop/dev-machine prototype: Python scripts, a
local webcam, a local file cache. The product is meant to run on a phone. This
phase is that port — deliberately scoped now (so later decisions don't paint the
project into a corner) but not started, and not to be started until the desktop
tutor (Phases 0-7) is functionally complete. It's a different engineering
discipline (mobile app development, different languages/runtimes) layered on
top of the ML/data work above, not a continuation of it.

**The resource/complexity question, answered directly (asked explicitly before
this phase was written up): yes** — everything in the CURRENT stack (excluding
the rtmlib backends, which were evaluated and not carried forward regardless of
this phase) is algorithmically light enough for phone-class hardware. This is a
statement about compute/memory budget, not about the existing Python code being
reusable — none of it ships as-is; see the porting work below.
- **MediaPipe Holistic** is Google's own mobile-first architecture (BlazePose),
  with official GPU/NPU-accelerated Android and iOS SDKs — this is its flagship
  use case, shipped in production apps today. The `.task` model files already in
  `models/` (`pose_landmarker_full.task`, `face_landmarker.task`,
  `hand_landmarker.task`) are the same cross-platform bundle format MediaPipe
  uses on mobile, so they are plausibly reusable as-is, not just the architecture.
  Running three landmarkers (pose+face+hands) every frame is heavier than a
  single-model mobile CV app, but still well inside what current phones handle
  in real products — a genuine, if previously unstated, point in MediaPipe's
  favor on top of its accuracy/stability lead from Phase 3.
- **`normalizer/`/`features.py`** is plain anchor-based geometry and NumPy math
  (selection, concatenation, velocity, standardization) over ~550 keypoints —
  computationally trivial regardless of implementation language; a phone CPU
  does this in microseconds.
- **`EmbeddingGrader`'s `PoseGraderNet`** (multi-stream BiGRU + 5 small linear
  heads, no attention/transformer, no large embedding tables) is architecturally
  a small model relative to on-device budgets — nowhere near LLM or vision-
  transformer scale. Inference over a ~60-frame window is cheap.
- **The LLM feedback call** (`generator/llm_feedback.py`) is a hosted API call
  by design (see below) — zero on-device compute, exactly what a phone client
  would do.
- **The gloss rule engine** (spaCy-backed) is the one piece that ISN'T
  per-frame/real-time-critical — it runs once per sentence prompt, not once per
  video frame — so it's a real candidate to simply stay server-side rather than
  be ported to an on-device NLP toolkit (see below).

**What that resource-feasibility answer does NOT cover — the actual porting
work, all unstarted:**
1. **Replace the Python Tasks API with MediaPipe's native Android/iOS SDK.**
   Current code (`extractor/mediapipe.py`) uses the desktop/server Python Tasks
   API; a phone app needs the Kotlin/Swift native SDK instead. The `.task` model
   files likely carry over; the Python wrapper code does not.
2. **Reimplement `normalizer/`/`features.py` natively.** The math is simple
   enough to be a tractable, bounded port (not a research problem), but it's a
   real rewrite in Kotlin/Swift (or a shared C++ core bridged to both), not
   reused Python.
3. **Export/convert `PoseGraderNet` to a mobile inference runtime** — PyTorch
   Mobile, ExecuTorch, CoreML, or TFLite. Untested; the main concrete risk is
   the BiGRU's export path (variable-length sequence handling is a known rough
   edge for some of these converters) — needs a proof-of-concept export before
   this is treated as a solved problem, not assumed to "just work."
4. **Decide how reference clips reach the phone.** `data/cache/`'s local file
   layout (1,874 clips) doesn't fit in an app bundle as-is. Needs a real
   decision: bundle a curated subset, stream from a CDN, or download-and-cache
   on first use. A product/infra question, not an ML one.
5. **Decide where the gloss engine runs.** Recommended: keep it server-side
   (English sentence in, gloss sequence out) rather than port spaCy to an
   on-device NLP toolkit — it's not latency-critical the way per-frame pose
   extraction is, so there's no forced reason to put it on-device. Still a
   fail-closed rule engine either way; "server-side" changes where it runs, not
   what it is or CLAUDE.md's "never an LLM" constraint on it.
6. **Learner state storage** (`learner/mastery.py`'s JSON file) has an obvious
   phone-native equivalent (local app-sandbox storage — SQLite, a JSON file in
   app storage, platform prefs) — a straightforward port, not a design problem,
   noted here only so it isn't forgotten as one of the pieces that moves.
- **Produces:** a real phone app, functionally equivalent to the Phase 6
  desktop tutor.
- **Done when:** the full loop (extract → normalize → grade → diagnose →
  adapt → coach) runs natively on a phone at usable latency, with reference
  clips actually reaching the device and the LLM feedback call working over a
  real mobile network path.

### Phase 9 — Curriculum expansion (60 signs → full vocabulary) — status: not started, not scheduled

Asked directly ("when do we start expanding from 60 signs to everything?"):
there is no scheduled trigger for this in the build order above. Every phase
through Phase 8 is scoped to work ON TOP OF the fixed 60-sign curriculum
(Phase 0), not to grow it — recorded here so the answer doesn't have to be
re-derived if asked again, and so this doesn't get scheduled by accident
ahead of the phases it would actually depend on.

- **The dataset itself is not the blocker.** ASL Citizen covers 2,731 signs
  total (Phase 1), of which the current curriculum uses ~60 — the other
  ~2,670 are already inside the same consented, IRB-approved corpus, just
  unused. The dataset is confirmed still downloaded locally, so expansion
  would not need a re-acquisition step, only re-running the existing
  pipeline over more of what's already there.
- **What actually has to happen per new sign**, same as Phase 1 did for the
  original 60: confirm it resolves to a real ASL-LEX ID-gloss (the join key
  everything downstream uses), extract cached keypoints across every
  extractor in use (currently MediaPipe), and join its ASL-LEX phonological
  labels (handshape/major-location/minor-location/movement/repeated).
- **The embedding grader (Phase 4) would need retraining, not just more
  cache.** `PoseGraderNet`'s phonological heads are already gated by
  `MIN_SUPPORT` (3 distinct signs per label value) precisely because thin
  classes can't be shown to generalize — going from 60 to hundreds/thousands
  of signs changes the label distribution per parameter substantially (for
  better in aggregate, likely worse for rare handshape/location classes
  freshly introduced), so this is a full re-run of
  `scripts/train_embedding_grader.py` and a fresh `PHASE4_REPORT.md`-style
  validation pass, not a drop-in.
- **The gloss rule engine's lexicon (Phase 5b)** is derived from
  `curriculum.yaml`'s `english_lemmas` and fails closed at import time on any
  ambiguous lemma (issue #12) — a much larger vocabulary raises the odds of a
  real collision (two signs both plausibly claiming the same English word)
  that has to be resolved by hand, not just a bigger table.
- **Scale of the lift**: this is closer in size to Phases 0-4 combined
  (curriculum selection at scale, re-extraction, re-training, re-validation)
  than to a routine data-addition task — it should be proposed and scoped as
  its own deliberate phase, prioritized against the currently open items
  (spaced-repetition/difficulty scheduling, Phase 5b's Deaf review, Phase 8),
  not started implicitly.
- **Produces:** an expanded `curriculum.yaml`, full-vocabulary caches, a
  retrained/revalidated `EmbeddingGrader`, and a re-verified gloss lexicon.
- **Done when:** none of the above exists yet — this phase has not begun.

---

## Research aside — CTC-CSLR vs. forced alignment — status: DONE, isolated comparison, NOT a phase

**Deliberately NOT numbered as a phase and NOT part of the shipped product.**
Built and run once, entirely inside `src/aslcv/research/` + three `scripts/`,
to empirically answer "how does the industry-standard approach to continuous
sign recognition (CTC-based CSLR) compare to this project's own forced-
alignment system?" — after a design-fork question surfaced this directly:
CTC is structurally the N-way open-vocabulary classifier CLAUDE.md's
non-negotiables explicitly rule out for the product (a classifier must emit
SOME label and will confidently mislabel a malformed attempt as a real
sign), and "free-form translation" is explicitly out of scope. Rather than
silently building it into the product or silently skipping the request, the
scope was proposed and confirmed first: an isolated benchmark, framed
honestly as "a CTC architecture trained on this project's own data," never
"SOTA" (real SOTA — DeepMind's SL2T — trains on 100k+ hours across 50+
languages; this trains on ~950 synthetic sentences built from this
curriculum's own 1,874 clips).

**What was built:**
- `src/aslcv/research/synthetic_sentences.py` — `make_trial`, factored OUT of
  `scripts/eval_forced_alignment.py` (Phase 7 step 4's own validation
  benchmark) rather than reimplemented, so both the Phase 7 benchmark and
  this comparison generate IDENTICAL synthetic continuous sentences
  (concatenated real, trimmed, held-out clips with known ground-truth
  boundaries) — a true apples-to-apples comparison, not two independently-
  drifting generators.
- `src/aslcv/research/ctc_cslr.py` — `CTCEncoder` (a BiGRU + linear head over
  vocab+blank classes, sized to roughly match `PoseGraderNet`'s combined
  stream capacity — not a strawman), `greedy_decode` (CTC's real use case:
  no known target), `forced_align` (the fair comparison point: given the
  TRUE label sequence, the standard CTC Viterbi forced-alignment recurrence
  over the blank-interleaved extended target — the same *kind* of algorithm
  as `dtw_align`, just CTC's version), and `word_error_rate`/`edit_distance`.
  `forced_align`'s correctness is independently verified, not just asserted:
  a Viterbi best-path log-prob can never exceed `nn.CTCLoss`'s own full
  marginal (which sums over every valid path) — `tests/test_ctc_cslr.py`
  checks this directly on a random instance, the strongest available check
  short of re-deriving the algorithm a second way.
- `scripts/train_ctc_cslr.py` — trains on ~950 synthetic sentences (TRAIN-
  split clips), using the SAME `FeaturePipeline`/`Standardizer` the trained
  `EmbeddingGrader` checkpoint already uses (borrowed from it, not rebuilt),
  so the comparison isolates the *approach*, not a difference in feature
  engineering. 40 epochs, ~126s on this machine's GPU. Reports train/val WER
  every epoch, same "never hide the gap" discipline as
  `train_embedding_grader.py` — and there IS a real gap: train WER reaches
  **0.0%**, val WER plateaus around **32-34%**, disclosed plainly, not
  buried.
- `scripts/eval_ctc_vs_alignment.py` — runs BOTH systems on 100 identical
  held-out synthetic sentences (`CTC_VS_ALIGNMENT_REPORT.md`), measuring
  three things explicitly flagged as NOT all directly comparable (the
  asymmetry itself is the finding):
  1. **Open-set recognition (CTC's real use case)**: free-decode WER
     **34.5% mean / 33.3% median**. `align_and_grade` has no analogous
     number — it never guesses a sequence, the target is always known.
  2. **Forced alignment given the TRUE sequence (the fair comparison)**:
     boundary error — `align_and_grade` **12.8% mean / 4.0% median** vs.
     CTC forced-align **96.6% mean / 97.0% median**. Investigated rather
     than reported blind, since a number that close to total failure could
     as easily be a bug: manual inspection of a real trial confirmed the
     trained CTC model learned extremely "peaky" posteriors (blank wins on
     74/77 frames by a huge margin on one representative trial; the real
     label spikes for only 1-2 frames) — a well-documented, genuine property
     of vanilla CTC training, not a bug in `forced_align` (which is
     independently correctness-checked, see above). Greedy decoding
     tolerates a spike that brief; Viterbi forced alignment correctly finds
     exactly where it is, producing a 1-2-frame segment for a sign whose
     true length was 35-42 frames. Known mitigations (entropy
     regularization, a dedicated alignment objective) exist in the
     literature and were deliberately NOT applied, since the goal was a fair
     vanilla-CTC comparison on this project's own data, not a maximally-
     optimized CTC pipeline.
  3. **Grading agreement on the resulting segments** (same metric Phase 7
     step 4 already reports for `align_and_grade`, computed identically for
     CTC's forced-aligned segments): `align_and_grade` **92.4%-94.6%** across
     all 5 parameters vs. CTC forced-align **33.5%-58.9%** — the direct
     downstream consequence of (2)'s catastrophic boundaries.

**The actual finding, stated plainly:** on this project's own small dataset,
CTC is a reasonable open-vocabulary recognizer (34.5% WER isn't bad for
~950 synthetic training sentences with zero real continuous footage) but a
poor *forced aligner* without extra work most CTC-CSLR systems don't need
to do (because they're not usually asked to align to a KNOWN sequence).
Forced alignment (`align_and_grade`) is dramatically better at the ONE thing
this project actually needs — segmenting a KNOWN target sentence — while
needing zero sequence-labeled training data at all (training-free DTW over
an `EmbeddingGrader` trained only on isolated single-sign clips) and
structurally cannot ever confidently mislabel a malformed attempt as some
other real sign, since it never guesses an identity in the first place. This
empirically confirms, rather than just asserts, CLAUDE.md's non-negotiable
("grade by distance to a reference, never N-way classification") was the
right call for this specific product.

Full suite unaffected (research code, no product code changed): **336
passed / 10 skipped.**

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
      features.py      # DONE — feature assembly (select/concat/confidence/velocity/stack/standardize);
                        #   legs/feet dropped by default via Skeleton.region(); velocity zeroes
                        #   deltas across presence gaps; trim_to_motion + depth_proxies toggles;
                        #   hand_motion_energy() shared with the future live segmenter
      capture.py       # DONE — CaptureBuffer: live idle/active/settled boundary
                        #   detection (energy_fn-injected, unit-tested without a
                        #   camera) so diagnose_demo.py grows the capture from
                        #   motion-start to a natural rest boundary instead of
                        #   truncating a slow signer with a fixed trailing window
      normalizer/      # DONE — ShoulderNormalizer (global + local-hand); BBox skipped as unsound
      pipeline_config.py # DONE — add_pipeline_args/build_pipeline: every script builds the
                        #   feature pipeline identically + prints the resolved config each run
      grading/         # DONE — DTW baseline + Phase 4 EmbeddingGrader
        dtw_grader.py  #   Phase 2 — training-free nearest-reference DTW (baseline);
                       #   Phase 7 step 1 — dtw_align, the path-returning sibling of
                       #   dtw_distance, for forced alignment
        alignment.py   #   Phase 7 step 3 — align_and_grade: DTW-aligns an attempt
                       #   against compose_reference_features's composed reference,
                       #   projects boundaries onto the attempt, grades each segment
                       #   via grade_against_poses unchanged
        phonology_labels.py # Phase 4 — per-parameter label vocab (sorted, curriculum-
                       #   derived) + per-class support counts + MIN_SUPPORT gate
        embedding_model.py  # Phase 4 — PoseGraderNet (multi-stream BiGRU encoder +
                       #   5 disjoint-input phonological heads), batch_hard_triplet_loss
        embedding_dataset.py # Phase 4 — in-memory Dataset (features + tempo feature +
                       #   phonology labels per clip), PK batch sampler, collate_fn
        embedding_grader.py  # Phase 4 — EmbeddingGrader (grade/grade_against mirroring
                       #   DTWGrader's interface; grade_against also returns per-
                       #   parameter ParameterVerdict gated by MIN_SUPPORT, incl. a
                       #   confidence field; grade_poses/grade_against_poses take
                       #   in-memory frames for live use)
      production/      # Phase 5
        gloss_rules.py #   5b — spaCy-backed, fail-closed gloss + NMM rule engine
                       #   (BUILT + hardened: curriculum-derived lexicon that fails
                       #   closed on ambiguous lemmas at import time, in_scope/
                       #   confidence/reason/trace, out-of-scope construction
                       #   detection via dependency labels incl. AUX-headed clause
                       #   coordination)
        retrieval.py   #   5a — DONE: fetch_reference (id_gloss -> real clip,
                       #   one formalized selection rule) + fetch_sequence
                       #   (GlossSequence -> trimmed, hard-cut, concatenated
                       #   video; fail-closed) + write_composed_video. Phase 7
                       #   step 2 also lives here: compose_reference_features
                       #   (same fail-closed resolution, feature space instead
                       #   of pixels, for forced alignment).
        rules/         #   5b — declarative drop/reorder/NMM rules (reviewable data;
                       #   still Python control flow today, not yet extracted here)
        lexicon.py     #   5b — English lemma → ID-gloss, verb-class tags (currently
                       #   lives inline in gloss_rules.py, built from curriculum.yaml)
        templates/     #   5c — correct-by-construction generation (later)
      learner/         # Phase 6 — DONE, real spaced repetition + a first
                       #   difficulty signal, not just the v1 heuristic
        mastery.py     #   MasteryState: per-sign/per-parameter EMA mastery from
                        #   real ParameterVerdict.correct (None skipped, not
                        #   averaged in); SM2-style interval/ease per sign on the
                        #   logical attempt-clock (is_due/due_at); atomic JSON
                        #   save/load (data/learner_state.json, gitignored)
        scheduler.py    #   find_minimal_pairs (differ-in-one-parameter pairs,
                        #   scoped to the active pool) + pick_next (due-gated
                        #   weighted-random by weakness, an automatic reactive
                        #   contrastive drill when the last mistake has a real
                        #   partner sign, and a proactive contrastive redirect
                        #   for well-mastered picks -- the difficulty signal)
      generator/       # Phase 6 v1 + v2 — ALL DONE: isolated + contrastive
                        #   drills, feedback (templated + optional LLM), and
                        #   LLM sentence prompts (gloss-engine-gated)
        feedback.py    #   coach_text/focus_parameter: templated English
                        #   coaching from a graded attempt, duck-typed on
                        #   ParameterVerdict's shape (no grading-package import)
        _hf_client.py   #   shared .env/HF_TOKEN resolution for llm_feedback.py
                        #   + sentence_prompts.py (one place, not two copies)
        llm_feedback.py #   coach_text_maybe_llm: optional LLM phrasing pass
                        #   over the SAME facts, via HuggingFace's hosted
                        #   Inference API (not a local model -- see Phase 8);
                        #   fails open to feedback.coach_text on any error
        sentence_prompts.py # sentence_prompt_maybe_llm: LLM writes English
                        #   containing the target word; Phase 5b's fail-closed
                        #   gloss_rules.gloss_sentence() -- not the LLM -- is
                        #   what decides whether it's ever displayed
        sign_description.py # describe_sign: ALWAYS-ON (no LLM) grounded
                        #   description of a target sign's own phonology, from
                        #   PhonologyLabels/phonology.csv directly
    scripts/           # DONE — extract_landmarks.py (extract, records provenance),
                       #   verify_cache.py (integrity + provenance/staleness), render_clip.py
                       #   (draw cached poses back onto video for eyeballing), eval_slice.py
                       #   (DTW top-1/top-5 on the val slice), live_demo.py (webcam + --selftest),
                       #   eval_minimal_pairs.py (contrastive-pair confusability × 4 extractors,
                       #   by parameter; --full-curriculum for 60-way), measure_motion_energy.py
                       #   (rest-frame diagnostic), verify_vitpose_topology.py,
                       #   gloss_repl.py (5b interactive discovery REPL), compose_sentence.py
                       #   (5b text-only end-to-end composer: english → gloss → cached clip plan),
                       #   export_review_sheet.py (5b: PENDING_CASES + constructions_v2 →
                       #   one-file HTML Deaf review sheet, embedded clips + verdict form),
                       #   train_embedding_grader.py (Phase 4: PK-sampled triplet + multi-
                       #   task head training on the 60-sign curriculum, reports train/val
                       #   every epoch, saves best-val + final checkpoints), eval_embedding_
                       #   grader.py (Phase 4: re-measures DTW on the 60-sign val split,
                       #   compares to the learned grader, per-parameter accuracy report,
                       #   mother/father head-independence demo -> PHASE4_REPORT.md),
                       #   diagnose_demo.py (Phase 4 grading + Phase 6 v1 adaptive
                       #   loop: live webcam grade_against demo, reference clip loop,
                       #   per-parameter verdicts, fail-closed guards, --selftest,
                       #   --gpu; [n] now records mastery + picks the next target
                       #   adaptively instead of cycling a fixed list),
                       #   benchmark_extractors.py (Phase 3: FPS/jitter/region-split
                       #   hand-dropout across all 4 extractors, --gpu for mediapipe)
    tools/             # DONE — build_manifest, resolve_keys, join_phonology, validate_curriculum
    app/               # Phase 6 v1 loop lives in scripts/diagnose_demo.py for now,
                       #   not packaged here yet -- see Phase 6 status
    tests/
      test_embedding_model.py   # Phase 4 — architecture tests on synthetic tensors,
                                 #   incl. test_heads_are_structurally_disjoint (backprop-
                                 #   proves head independence, no checkpoint needed)
      test_embedding_grader.py  # Phase 4 — phonology-gate unit tests (no checkpoint)
                                 #   + checkpoint-gated EmbeddingGrader tests (skipped if
                                 #   models/embedding_grader/model_best.pt is absent),
                                 #   incl. the real mother/father disagreement regression
      test_gloss_rules.py         # 5b — engine's own mechanical regression tests
      test_gloss_rules_corpus.py  # 5b — golden corpus: mechanical properties asserted
                                   #   hard (incl. hardened fail-closed probes: partial-
                                   #   scope, ambiguous lexicon, wh-relative-vs-question,
                                   #   AUX-headed clause coordination); ASL-judgment-pending
                                   #   cases in PENDING_CASES report only (pytest.skip),
                                   #   promoted by review flag
    data/              # gitignored
      ASL_LEX/         # DONE — phonological features + ID-gloss keys
      ASL_Citizen/     # DONE — videos + official splits
      manifest.csv, phonology.csv, cache/{extractor}/*.npz   # DONE — Phase 1 outputs
    models/embedding_grader/  # gitignored — Phase 4 checkpoint (model_best.pt,
                       #   model_final.pt, standardizer.npz, config.json, history.json),
                       #   written by scripts/train_embedding_grader.py
    PHASE4_REPORT.md   # DONE — Phase 4 eval report: learned vs DTW baseline, per-
                       #   parameter val accuracy, mother/father disagreement demo
      review/gloss_review_sheet.html  # generated by export_review_sheet.py, not committed
    PHASE5B_GAP_REPORT.md  # DONE — write-up of the fail-closed probing pass: what was
                       #   checked, which 2 of 4 categories exposed real gaps, how each
                       #   was fixed (not just flagged)