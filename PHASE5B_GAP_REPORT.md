# Phase 5b gloss rule engine -- fail-closed probe report

Scope: harder probing of the ONE property that matters most for this engine --
fail-closed refusal, per CLAUDE.md's non-negotiable "refuse out-of-scope input
rather than emit a wrong target." Four categories were probed beyond the
existing corpus suite's OOV/construction cases (all now in
`tests/test_gloss_rules_corpus.py`'s "6b" section). Two real gaps were found;
both are fixed in `gloss_rules.py`, not just flagged.

## 1. Partial-scope sentences (some words in-curriculum, some not)

**Probed:** unknown word as sentence-final object (`"I live in America."`),
as subject (`"Chicago is good."`), as the first of two coordinated objects
(`"I want a pizza and coffee."`), and as a mid-sentence appositive
(`"My friend Bob is happy."`).

**Result: PASS, no gap.** All four refuse the whole sentence and name the
unknown word, regardless of its position or POS tag (all PROPN here). This
was already covered in principle by one existing test
(`test_mixed_in_scope_and_out_of_scope_words_names_only_the_unknown_ones`);
the new `test_partial_scope_refuses_whole_sentence_not_truncated` broadens
that from one case to four, varying position and POS, since "the unknown
word happens to be the object" is a narrower property than "an unknown word
anywhere refuses the whole sentence."

## 2. Ambiguous lexicon hits (one English word -> multiple curriculum signs)

**Probed:** does `curriculum.yaml` currently contain a lemma claimed by two
different signs? Checked exhaustively -- **no, it does not.** So there was no
live case to run through `GlossRuleEngine.gloss()`.

**Result: real gap, in the code, not the data.** `_build_lexicon()` built the
lemma -> gloss map with a plain `dict[lemma] = gloss` assignment in a loop --
if curriculum.yaml ever *did* introduce a genuine collision (or a
`_SUPPLEMENTARY_LEXICON` synonym silently collided with a real curriculum
mapping), whichever sign was processed last would win, silently, with no
error and no test failure. That's exactly the "silent pick" the fail-closed
principle exists to prevent -- it was just latent, because no live data
happens to trigger it today.

**Fixed:** `_build_lexicon()` now raises `ValueError` if it ever sees a lemma
already mapped to a *different* gloss, from either source (two curriculum
signs, or a supplementary entry vs. a curriculum mapping). This fails at
**module import time** -- the whole engine refuses to load rather than run
with an ambiguous lexicon. Pinned with two tests that construct a synthetic
collision directly against `_build_lexicon(signs=...)` (there being no real
one to exercise this through `.gloss()`), plus one test that asserts the real
curriculum has zero collisions today, so a future curriculum edit that
introduces one fails loudly in CI rather than only at whatever moment someone
happens to type the ambiguous word into the REPL.

## 3. Wh-word as relative pronoun vs. wh-question

**Probed:** `"My friend who is happy is tired."` (relative clause) against
`"Who is your friend?"` / `"Who is happy?"` (genuine questions), all built
from curriculum-only vocabulary so the refusal (if any) couldn't be
coincidentally explained by an unrelated OOV word.

**Result: PASS, no gap -- the engine already gets this right.** spaCy's
dependency parse marks the relative clause's verb with `dep_ == "relcl"`
regardless of which wh-word introduces it, and the existing scope check
already refuses on any `relcl` label -- so `"who"` as a relative pronoun was
*already* being refused, for the *right* reason (checked the actual refusal
reason names `relcl`, not a vocabulary gap). The genuine questions
classify as `wh_question` and reorder normally. This means the risk
flagged in the task -- "does it wrongly apply wh-question NMM to a relative
clause?" -- does not happen; the distinction is structural (dependency
label), not a surface-word heuristic that could get fooled. Pinned as a
regression (`test_wh_word_as_relative_pronoun_refuses_not_tagged_as_question`
+ `test_wh_word_as_genuine_question_still_works`) so this stays true if the
scope-check logic changes later.

## 4. Multi-clause / conjoined input

**Probed:** verb-headed coordination (already tested: `"I want coffee and you
want water."`, `"I go home and you go school."`), copula/AUX-headed
coordination (`"I am tired but you are happy."`, `"I am happy and you are
tired."`), and punctuation-only juxtaposition with no conjunction
(`"I want coffee; you want water."`, `"I want coffee, you want water."`).

**Result: real gap, found and fixed.** `"I am tired but you are happy."`
**did not refuse** -- it produced `['ME', 'TIRED', 'YOU', 'HAPPY']`, silently
gluing two independent clauses (each with its own subject) into one flat
gloss sequence with no coordination marker at all. Root cause: the
clause-coordination check was `tok.dep_ == "conj" and tok.pos_ == "VERB"`, but
a second clause built on a copula gets its head tagged `AUX` by spaCy (`"are"`
in `"...but you ARE happy"`), not `VERB` -- so this entire class of
coordinated clause slipped past the check. Confirmed the bug is general (not
`"but"`-specific): `"I am happy and you are tired."` reproduced it too.
Confirmed this is a genuinely different case from what already worked, not a
regression risk in disguise: `"I am happy and tired."` (single clause,
coordinated *predicate*, no second subject) has `conj` on the second `ADJ`,
never on an `AUX`/`VERB` -- so broadening the check to `{VERB, AUX}` doesn't
touch it.

**Fixed:** `_scope_problems` in `gloss_rules.py` now checks
`tok.pos_ in _CLAUSE_HEAD_POS` where `_CLAUSE_HEAD_POS = {"VERB", "AUX"}`,
instead of `== "VERB"`. Verified against all cases above: both AUX-coordinated
sentences now refuse; both already-working VERB-coordinated sentences still
refuse; both already-working same-clause coordination sentences (`"I am happy
and tired."`, `"I want coffee and water."`) are unaffected; a new 3-item
Oxford-comma list (`"I want coffee, water, and milk."`) is also confirmed
unaffected. The punctuation-only cases (semicolon/comma, no conjunction) were
already refusing correctly before this fix -- spaCy attaches the second
clause via `ccomp` in both, which the pre-existing `_OUT_OF_SCOPE_DEPS` check
already caught; pinned as a regression alongside the others rather than left
unverified.

## Summary table

| Probe | Outcome | Action |
|---|---|---|
| Partial-scope (unknown word, varied position/POS) | Pass | Broadened existing test from 1 to 4 cases |
| Ambiguous lexicon hits | Gap (latent, no live trigger) | Fixed: `_build_lexicon` fails closed at import time |
| Wh-word: relative pronoun vs. question | Pass | Pinned as regression (was already correct) |
| Multi-clause: AUX-headed coordination | **Gap (live, wrong output today)** | **Fixed**: `_CLAUSE_HEAD_POS` now `{VERB, AUX}` |
| Multi-clause: punctuation-only juxtaposition | Pass | Pinned as regression (was already correct) |

Nothing was left as an unfixed "known limitation" from this pass -- both real
gaps found had a clean, narrowly-targeted fix rather than a fundamental scope
question, so both are fixed rather than deferred.
