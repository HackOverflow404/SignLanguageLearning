"""Property + regression suite for the Phase 5b gloss rule engine.

CRITICAL FRAMING -- read before adding a case here. The author of this suite
does not know ASL. Nothing in this file asserts a hand-authored "correct ASL"
gloss sequence as ground truth. Every `assert` below checks a MECHANICAL
property -- true regardless of ASL fluency, because it follows directly from
the engine's own stated rules (drop this word class, front time signs, tag
this construction), not from knowing what real ASL looks like.

Anything that DOES require ASL judgment -- whether a whole gloss sequence's
word order is actually correct, natural ASL -- is never asserted pass/fail
here. It is collected in PENDING_CASES below and reported via `pytest -rs`
(skip reasons show the engine's actual output) or by running this file
directly (see __main__). A case is promoted from "pending report" to a real
regression assertion by setting `reviewed=True` and filling in the expected
output -- one flag flip, not a rewrite. See test_deaf_reviewed_word_order.

Runs under pytest OR as a plain script (`python tests/test_gloss_rules_corpus.py`).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aslcv.production.gloss_rules import GlossRuleEngine, GlossSequence
from aslcv.production import gloss_rules as gr

E = GlossRuleEngine()

_CURRICULUM = yaml.safe_load(open(REPO / "curriculum.yaml"))
_SIGNS = [s for u in _CURRICULUM["units"] for s in u["signs"]]
CURRICULUM_ASLLEX_CODES = {s["gloss"]: s["asllex_code"] for s in _SIGNS}
CONSTRUCTIONS_V2 = _CURRICULUM["constructions_v2"]


# ---------------------------------------------------------------------------
# 1. function-word dropping: articles/copula/listed prepositions absent;
#    content words (noun/verb/adjective/adverb/numeral/pronoun) retained.
# ---------------------------------------------------------------------------

# (sentence, English surface forms that MUST NOT appear as a gloss source)
FUNCTION_WORD_DROP_CASES = [
    ("The dog is hungry.", {"the", "is"}),
    ("A book is good.", {"a", "is"}),
    ("We are happy.", {"are"}),
    ("Tomorrow I go to school.", {"to"}),
    ("I want to eat.", {"to"}),
    ("The mother and the father are tired.", {"the", "and", "are"}),
]


@pytest.mark.parametrize("sentence,dropped", FUNCTION_WORD_DROP_CASES)
def test_function_words_dropped(sentence, dropped):
    seq = E.gloss(sentence)
    assert seq.in_scope, f"expected in-scope, got refused: {seq.reason}"
    sources = {g.source.lower() for g in seq.glosses}
    leaked = dropped & sources
    assert not leaked, f"{sentence!r}: function word(s) {leaked} leaked into the gloss"


# (sentence, English surface forms that MUST appear as a gloss source --
# content words the drop rule must never remove)
CONTENT_WORD_RETAIN_CASES = [
    ("The dog is hungry.", {"dog", "hungry"}),
    ("Tomorrow I go to school.", {"tomorrow", "go", "school"}),  # lowered: case-insensitive check below
    ("We are happy.", {"we", "happy"}),
    ("I want to eat.", {"want", "eat"}),
]


@pytest.mark.parametrize("sentence,retained", CONTENT_WORD_RETAIN_CASES)
def test_content_words_retained(sentence, retained):
    seq = E.gloss(sentence)
    assert seq.in_scope, f"expected in-scope, got refused: {seq.reason}"
    sources = {g.source.lower() for g in seq.glosses}
    missing = retained - sources
    assert not missing, f"{sentence!r}: content word(s) {missing} were dropped"


# ---------------------------------------------------------------------------
# 2. time-fronting: a time sign appears clause-initial
# ---------------------------------------------------------------------------

TIME_FRONTING_CASES = [
    "I work today.",           # time already first in English -- must STAY first
    "I am tired now.",         # time last in English -- must MOVE to first
    "I go to school tomorrow.",
    "The time is now.",        # "time" itself is a TIME-set gloss
]


@pytest.mark.parametrize("sentence", TIME_FRONTING_CASES)
def test_time_sign_is_clause_initial(sentence):
    seq = E.gloss(sentence)
    assert seq.in_scope, f"expected in-scope, got refused: {seq.reason}"
    has_time = any(g.asllex_id in gr._TIME for g in seq.glosses)
    assert has_time, f"{sentence!r}: no TIME-set gloss in output (test setup is wrong, not the engine)"
    assert seq.glosses[0].asllex_id in gr._TIME, (
        f"{sentence!r}: time sign not clause-initial: {[g.text for g in seq.glosses]}")


# ---------------------------------------------------------------------------
# 3. wh-questions: wh-sign clause-final AND brow_furrow over the clause
# ---------------------------------------------------------------------------

WH_QUESTION_CASES = [
    "What is your name?",
    "Where is my mother?",
    "Who is your friend?",
    "Why are you sad?",
    "How are you?",
]


@pytest.mark.parametrize("sentence", WH_QUESTION_CASES)
def test_wh_question_final_and_furrowed(sentence):
    seq = E.gloss(sentence)
    assert seq.in_scope, f"expected in-scope, got refused: {seq.reason}"
    assert seq.sentence_type == "wh_question", f"{sentence!r}: not classified wh_question"
    assert seq.glosses[-1].asllex_id in gr._WH, (
        f"{sentence!r}: wh-sign not clause-final: {[g.text for g in seq.glosses]}")
    furrows = [n for n in seq.nmm_tags if n.marker == "brow_furrow"]
    assert len(furrows) == 1, f"{sentence!r}: expected exactly one brow_furrow tag"
    assert (furrows[0].start, furrows[0].end) == (0, len(seq.glosses)), (
        f"{sentence!r}: brow_furrow does not span the whole clause")


# ---------------------------------------------------------------------------
# 4. yes/no questions: brow_raise over the whole clause
# ---------------------------------------------------------------------------

YES_NO_QUESTION_CASES = [
    "Are you tired?",
    "Is your mother happy?",
    "Do you want coffee?",
    "Is the dog hungry?",
]


@pytest.mark.parametrize("sentence", YES_NO_QUESTION_CASES)
def test_yes_no_question_brow_raised(sentence):
    seq = E.gloss(sentence)
    assert seq.in_scope, f"expected in-scope, got refused: {seq.reason}"
    assert seq.sentence_type == "yes_no_question", f"{sentence!r}: not classified yes_no_question"
    raises = [n for n in seq.nmm_tags if n.marker == "brow_raise"]
    assert len(raises) == 1, f"{sentence!r}: expected exactly one brow_raise tag"
    assert (raises[0].start, raises[0].end) == (0, len(seq.glosses)), (
        f"{sentence!r}: brow_raise does not span the whole clause")


# ---------------------------------------------------------------------------
# 5. negation: headshake on the negated predicate
# ---------------------------------------------------------------------------

NEGATION_CASES = [
    "I don't want coffee.",
    "I do not want coffee.",
    "She does not work.",
    "We are not tired.",
]


@pytest.mark.parametrize("sentence", NEGATION_CASES)
def test_negation_headshake_on_predicate(sentence):
    seq = E.gloss(sentence)
    assert seq.in_scope, f"expected in-scope, got refused: {seq.reason}"
    assert seq.negated is True, f"{sentence!r}: negation flag not set"
    shakes = [n for n in seq.nmm_tags if n.marker == "headshake"]
    assert len(shakes) == 1, f"{sentence!r}: expected exactly one headshake tag"
    # the span must start at or after the first verb-sourced gloss (the
    # predicate), never at the very front over the subject alone
    verb_idx = next((i for i, g in enumerate(seq.glosses) if g.pos == "VERB"), 0)
    assert shakes[0].start >= verb_idx, f"{sentence!r}: headshake starts before the predicate"
    assert shakes[0].end == len(seq.glosses), f"{sentence!r}: headshake does not reach the clause end"


# ---------------------------------------------------------------------------
# 6. FAIL-CLOSED -- the safety property. Tested hardest.
# ---------------------------------------------------------------------------

UNKNOWN_WORD_CASES = [
    "I want pizza.",
    "I want to eat pizza now.",
    "My sofa is red.",
    "I have a computer.",
]


@pytest.mark.parametrize("sentence", UNKNOWN_WORD_CASES)
def test_unknown_word_refuses_not_a_confident_wrong_gloss(sentence):
    seq = E.gloss(sentence)
    assert seq.in_scope is False, f"{sentence!r}: expected refusal, engine produced a gloss instead"
    assert seq.glosses == [], f"{sentence!r}: refused sequence must carry NO glosses"
    assert seq.nmm_tags == [], f"{sentence!r}: refused sequence must carry NO nmm_tags"
    assert seq.confidence == 0.0, f"{sentence!r}: a refusal must not carry positive confidence"
    assert seq.reason, f"{sentence!r}: refusal reason must be populated"


UNSUPPORTED_CONSTRUCTION_CASES = [
    "The book that I read is good.",     # relative clause
    "I eat because I am hungry.",        # subordinate/adverbial clause
    "I know that you are tired.",        # clausal complement
    "The book was read by me.",          # passive voice
    "I want coffee and you want water.", # clause coordination
]


@pytest.mark.parametrize("sentence", UNSUPPORTED_CONSTRUCTION_CASES)
def test_unsupported_construction_refuses(sentence):
    seq = E.gloss(sentence)
    assert seq.in_scope is False, f"{sentence!r}: expected refusal (out-of-scope construction)"
    assert seq.glosses == [] and seq.nmm_tags == []
    assert seq.confidence == 0.0
    assert seq.reason and "construction" in seq.reason, f"{sentence!r}: reason should name the construction"


def test_want_to_verb_pattern_is_NOT_flagged_as_unsupported_embedding():
    """The one xcomp (open clausal complement) pattern the engine DOES support
    -- 'want to VERB' -- must not be caught by the complex-embedding refusal."""
    seq = E.gloss("I want to eat.")
    assert seq.in_scope, f"expected in-scope, got refused: {seq.reason}"


def test_noun_and_adjective_coordination_are_NOT_flagged():
    """'rice and beans' / 'happy and tired' are simple lists, not clause
    coordination -- conj is only a problem when it coordinates a second verb."""
    for sentence in ["I want coffee and water.", "I am happy and tired."]:
        seq = E.gloss(sentence)
        assert seq.in_scope, f"{sentence!r}: expected in-scope, got refused: {seq.reason}"


# ---------------------------------------------------------------------------
# 7. determinism
# ---------------------------------------------------------------------------

DETERMINISM_CASES = [
    "I work today.",
    "Are you tired?",
    "I want pizza.",              # a refusal must ALSO be deterministic
    "The book that I read is good.",
]


@pytest.mark.parametrize("sentence", DETERMINISM_CASES)
def test_determinism_same_input_same_output(sentence):
    results = [E.gloss(sentence) for _ in range(5)]
    first = results[0].to_dict()
    for r in results[1:]:
        assert r.to_dict() == first, f"{sentence!r}: non-deterministic output across repeated calls"


# ---------------------------------------------------------------------------
# 8. lexicon resolution: every gloss in an in_scope output resolves to a
#    real asllex_code in curriculum.yaml
# ---------------------------------------------------------------------------

LEXICON_RESOLUTION_SENTENCES = [
    "I work today.", "Are you tired?", "What is your name?",
    "I don't want coffee.", "The dog is hungry.", "Tomorrow I go to school.",
    "Where is my mother?", "We love you.", "My father is happy.",
    "Your sister is tired.",
    "I read a book at home.",
]


@pytest.mark.parametrize("sentence", LEXICON_RESOLUTION_SENTENCES)
def test_every_in_scope_gloss_resolves_to_curriculum_asllex_code(sentence):
    seq = E.gloss(sentence)
    if not seq.in_scope:
        pytest.skip(f"{sentence!r} refused ({seq.reason}) -- nothing to resolve, not a failure")
    for g in seq.glosses:
        assert g.asllex_id in CURRICULUM_ASLLEX_CODES, (
            f"{sentence!r}: gloss {g.asllex_id!r} has no curriculum.yaml asllex_code -- "
            f"the composer could not retrieve a reference clip for it")


def test_entire_lexicon_resolves_to_curriculum():
    """Stronger, exhaustive version of the property above: every value the
    lexicon can EVER emit (not just in these sample sentences) is a real
    curriculum gloss. This is also asserted at import time in gloss_rules.py
    itself; pinned here as a regression so a future lexicon change that
    breaks it fails loudly in the test suite too."""
    assert set(gr._LEXICON.values()) <= CURRICULUM_ASLLEX_CODES.keys()


# ---------------------------------------------------------------------------
# 9. vocabulary ceiling: non-curriculum content words are refused, and that
#    is EXPECTED BEHAVIOR, not a bug in the engine.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "I want pizza.", "My sofa is red.", "I have a computer.", "I watch television.",
])
def test_vocabulary_ceiling_is_expected_not_a_failure(sentence):
    """Same mechanism as the fail-closed tests above, asserted under its own
    name so it reads as what it is: the engine has EXACTLY the 60-sign
    curriculum vocabulary and nothing more, by design -- refusing a
    non-curriculum word is the ceiling working, not the engine being broken."""
    seq = E.gloss(sentence)
    assert seq.in_scope is False
    assert "unknown word" in seq.reason


# ---------------------------------------------------------------------------
# constructions_v2 coverage -- every construction curriculum.yaml declares
# supported must be exercised, mechanically, at least once. (Word order
# beyond these mechanical checks is PENDING review -- see below.)
# ---------------------------------------------------------------------------

def test_every_constructions_v2_entry_is_exercised():
    """Cross-check: every construction id curriculum.yaml lists must appear
    somewhere in this file's mechanical tests or the pending-review set below
    -- if a new construction is added to curriculum.yaml, this fails until a
    corresponding case exists here."""
    covered = {
        "time_first", "topic_comment", "yes_no_question", "wh_question", "negation",
    }
    declared = {c["id"] for c in CONSTRUCTIONS_V2}
    assert declared <= covered, f"constructions_v2 has uncovered ids: {declared - covered}"


def test_topic_comment_fronts_topic_and_brow_raises():
    """topic_comment is opt-in (.topicalize()), not automatic -- mechanically
    checkable: the chosen gloss moves to position 0 and gets brow_raise[0:1]."""
    base = E.gloss("I want coffee.")
    assert base.in_scope
    topic = E.topicalize(base, base.gloss_ids.index("coffee"))
    assert topic.glosses[0].asllex_id == "coffee"
    raises = [n for n in topic.nmm_tags if n.marker == "brow_raise"]
    assert len(raises) == 1
    assert (raises[0].start, raises[0].end) == (0, 1)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_empty_string_refuses():
    seq = E.gloss("")
    assert seq.in_scope is False
    assert seq.reason == "empty input"


def test_whitespace_only_refuses():
    seq = E.gloss("    ")
    assert seq.in_scope is False


def test_single_content_word():
    seq = E.gloss("Dog.")
    assert seq.in_scope
    assert [g.asllex_id for g in seq.glosses] == ["dog"]


def test_all_function_words_refuses_as_no_content():
    seq = E.gloss("To be.")
    assert seq.in_scope is False
    assert seq.reason == "no content words -- nothing to sign"


def test_mixed_in_scope_and_out_of_scope_words_names_only_the_unknown_ones():
    seq = E.gloss("I want coffee and pizza.")
    assert seq.in_scope is False
    assert "pizza" in seq.reason
    assert "coffee" not in seq.reason, "a resolvable word must not be reported as unknown"


# ---------------------------------------------------------------------------
# PENDING DEAF REVIEW -- word-order correctness beyond the mechanical checks
# above. NEVER asserted pass/fail while reviewed=False; report only. Promote
# a case by setting reviewed=True and filling in expected_glosses (and
# expected_nmm if relevant) -- that is the entire promotion process.
# ---------------------------------------------------------------------------


@dataclass
class PendingCase:
    english: str
    reviewed: bool = False
    expected_glosses: list[str] | None = None
    expected_nmm: list[tuple] | None = None
    note: str = ""


PENDING_CASES: list[PendingCase] = [
    # one per constructions_v2 example, using curriculum.yaml's own examples
    PendingCase("I work today.", note="time_first"),
    PendingCase("I want coffee.", note="topic_comment (via .topicalize)"),
    PendingCase("Are you tired?", note="yes_no_question"),
    PendingCase("What is your name?", note="wh_question"),
    PendingCase("I don't want coffee.", note="negation"),
    # a few varied-vocabulary sentences beyond the constructions_v2 examples
    PendingCase("Where is my mother?"),
    PendingCase("We love you."),
    PendingCase("My father is happy."),
    PendingCase("Why don't you go?", note="stacked wh_question + negation"),
    PendingCase("I read a book at home."),
]


@pytest.mark.parametrize("case", PENDING_CASES, ids=lambda c: c.english)
def test_deaf_reviewed_word_order(case: PendingCase):
    seq = E.gloss(case.english)
    if not case.reviewed:
        actual = seq.render() if seq.in_scope else f"REFUSED: {seq.reason}"
        pytest.skip(f"PENDING DEAF REVIEW -- engine produced: {actual}")
    assert case.expected_glosses is not None, (
        f"{case.english!r} marked reviewed=True but has no expected_glosses recorded")
    assert seq.in_scope, f"{case.english!r}: reviewed case is now refused ({seq.reason}) -- re-review"
    assert [g.text for g in seq.glosses] == case.expected_glosses
    if case.expected_nmm is not None:
        assert [(n.marker, n.start, n.end) for n in seq.nmm_tags] == case.expected_nmm


def print_pending_review_report() -> None:
    print("=" * 78)
    print("PENDING DEAF REVIEW REPORT")
    print("Nothing below is asserted pass/fail. A fluent signer must confirm")
    print("word order + NMM placement before any case here becomes a regression")
    print("test (see PendingCase.reviewed in tests/test_gloss_rules_corpus.py).")
    print("=" * 78)
    for case in PENDING_CASES:
        status = "REVIEWED" if case.reviewed else "pending"
        seq = E.gloss(case.english)
        print(f"\n[{status}] EN: {case.english}" + (f"   ({case.note})" if case.note else ""))
        if seq.in_scope:
            print(f"    -> {seq.render().replace(chr(10), chr(10) + '       ')}")
        else:
            print(f"    -> REFUSED: {seq.reason}")


if __name__ == "__main__":
    passed = failed = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except TypeError:
                pass  # parametrized function, not runnable bare in script mode
            except AssertionError as e:
                print(f"  FAIL {name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed (parametrized tests need pytest -- run "
          f"`.venv/bin/python -m pytest tests/test_gloss_rules_corpus.py`)")
    print()
    print_pending_review_report()
