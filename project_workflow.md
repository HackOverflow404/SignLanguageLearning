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
>   So the 66-vs-32 gap now *conflates backend quality with running mode* — see the
>   confound note in Phase 3.
> - **Depth proxies (apparent hand size / arm foreshortening) did NOT help** the one
>   pair they targeted (me/you, ~45% confusion either way): the raw signal doesn't
>   separate that pair, and me/you are indexical/deictic signs that are out of scope
>   anyway. True z is now a *motivated* Phase 3 ablation rather than a hunch, but
>   likely not worth re-extraction for one out-of-scope pair.
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
> Phase 3's core question — which extractor — is effectively answered (MediaPipe), so
> it stays narrowed to a couple of specific, cheap ablations noted in its section
> rather than a full grid (see "Phase 3 — status" below). With Phase 4 also done, the
> recognition track (Phases 1-4) has nothing left blocking it; what's open is Phase 5a
> (reference retrieval, not started), actual fluent Deaf review of Phase 5b's engine
> output, and Phase 6 (the adaptive loop).

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
below and `PHASE4_REPORT.md`. Phase 3's central question (which extractor) is
effectively settled — MediaPipe — so Phase 3 remains shrunk to two specific cheap
ablations rather than a full grid (see Phase 3 status); nothing on the recognition
track blocks moving to Phase 5a (reference retrieval, not started) or Phase 6.

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
   remains a possible Phase 3 ablation but is likely not worth re-extraction for one
   out-of-scope pair.

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

### Phase 3 — Benchmarking — status: mostly answered, narrowed to two ablations

**The central Phase-3 question — which extractor — is effectively answered: MediaPipe**
(see Phase 2 findings #1). So this is no longer a full grid to run; it's two specific
loose ends, then move to Phase 4. Cache-integrity is resolved (rtmlib only ran IMAGE
mode; clean across all 1,874 clips × 4 extractors).

**Loose end A — the running-mode confound (do before canonizing "MediaPipe wins by
34pts").** The 66-vs-32 gap conflates backend quality with running mode: rest-frame
trimming (+17pp) only helps MediaPipe because it runs VIDEO mode with temporal
smoothing, while the rtmlib backends run IMAGE mode. Two honest options: (a) re-run
the rtmlib backends in a smoothed/tracked mode and re-measure, or (b) simply *state*
that MediaPipe's default running mode is part of why it wins and stop there. The
minimal-pair advantage (finding #1) is independent of trim and already favors
MediaPipe, so this doesn't change the decision — it changes how the margin is
reported. Cheap; worth doing before the number goes in a paper/README.

**Loose end B — true z as a targeted ablation (optional, low priority).** Depth
proxies didn't help me/you (finding #3). If revisited, the hypothesis is specific:
"MediaPipe *with* its z channel reduces me/you confusion without hurting overall
top-1." But me/you is one out-of-scope indexical pair, and z requires re-extracting
MediaPipe (it currently discards the third coordinate at write time) and would break
apples-to-apples with the 2D rtmlib backends — so likely not worth it. Recorded so the
decision is deliberate, not forgotten.

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

### Phase 5 — Production track (parallel to Phases 1–4)

**5a Reference retrieval:** given an ID-gloss, fetch the real Deaf-signer clip(s)
and cached pose sequence. Serves both "show correct form" and "compose grading
target." Retrieval, never generation. *(Blocked on the ASL Citizen download.)*

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
  verdict form. **Not yet done:** actual video concatenation (still Phase 5a,
  blocked on nothing but hasn't been started), and — the part that actually
  gates calling this phase complete — no fluent Deaf reviewer has actually
  opened the review sheet and signed off on word order/NMM placement yet;
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
      features.py      # DONE — feature assembly (select/concat/confidence/velocity/stack/standardize);
                        #   legs/feet dropped by default via Skeleton.region(); velocity zeroes
                        #   deltas across presence gaps; trim_to_motion + depth_proxies toggles;
                        #   hand_motion_energy() shared with the future live segmenter
      normalizer/      # DONE — ShoulderNormalizer (global + local-hand); BBox skipped as unsound
      pipeline_config.py # DONE — add_pipeline_args/build_pipeline: every script builds the
                        #   feature pipeline identically + prints the resolved config each run
      grading/         # DONE — DTW baseline + Phase 4 EmbeddingGrader
        dtw_grader.py  #   Phase 2 — training-free nearest-reference DTW (baseline)
        phonology_labels.py # Phase 4 — per-parameter label vocab (sorted, curriculum-
                       #   derived) + per-class support counts + MIN_SUPPORT gate
        embedding_model.py  # Phase 4 — PoseGraderNet (multi-stream BiGRU encoder +
                       #   5 disjoint-input phonological heads), batch_hard_triplet_loss
        embedding_dataset.py # Phase 4 — in-memory Dataset (features + tempo feature +
                       #   phonology labels per clip), PK batch sampler, collate_fn
        embedding_grader.py  # Phase 4 — EmbeddingGrader (grade/grade_against mirroring
                       #   DTWGrader's interface; grade_against also returns per-
                       #   parameter ParameterVerdict gated by MIN_SUPPORT)
      production/      # Phase 5
        gloss_rules.py #   5b — spaCy-backed, fail-closed gloss + NMM rule engine
                       #   (BUILT + hardened: curriculum-derived lexicon that fails
                       #   closed on ambiguous lemmas at import time, in_scope/
                       #   confidence/reason/trace, out-of-scope construction
                       #   detection via dependency labels incl. AUX-headed clause
                       #   coordination)
        retrieval.py   #   5a — id_gloss → reference clip + pose sequence (later)
        rules/         #   5b — declarative drop/reorder/NMM rules (reviewable data;
                       #   still Python control flow today, not yet extracted here)
        lexicon.py     #   5b — English lemma → ID-gloss, verb-class tags (currently
                       #   lives inline in gloss_rules.py, built from curriculum.yaml)
        templates/     #   5c — correct-by-construction generation (later)
      learner/         # Phase 6 — learner model + adaptive engine
      generator/       # Phase 6 — drills, contrastive pairs, sentence prompts
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
                       #   mother/father head-independence demo -> PHASE4_REPORT.md)
    tools/             # DONE — build_manifest, resolve_keys, join_phonology, validate_curriculum
    app/               # Phase 6 — the loop, feedback presenter (LLM English only)
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