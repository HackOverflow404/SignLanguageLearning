"""Regression tests for the Phase 5b gloss rule engine.

Runs under pytest OR as a plain script (`python tests/test_gloss_rules.py`).
These pin the engine's OWN behavior (the mechanical rules); tests/
test_gloss_rules_corpus.py is the broader property/regression suite that also
covers fail-closed edge cases and separates ASL-judgment-pending output from
mechanically-checkable assertions.
"""

from aslcv.production.gloss_rules import GlossRuleEngine, gloss_sentence
from aslcv.production import gloss_rules as gr

E = GlossRuleEngine()


def _ids(sentence):
    return E.gloss(sentence).gloss_ids


def _text(sentence):
    return " ".join(g.text for g in E.gloss(sentence).glosses)


# -- lexicon is curriculum-derived, by construction -------------------------

def test_lexicon_is_curriculum_derived():
    # every emittable gloss must be a real curriculum sign (self-check already
    # runs as an assertion at import time; this pins that it's non-trivial)
    assert len(gr._LEXICON) > 50
    assert set(gr._LEXICON.values()) <= gr._CURRICULUM_GLOSSES


def test_role_sets_are_internally_consistent():
    emittable = set(gr._LEXICON.values())
    assert gr._WH <= emittable
    assert gr._TIME <= emittable


# -- the five curriculum v2 constructions -----------------------------------

def test_time_first():
    assert _text("I work today.") == "TODAY ME WORK"


def test_yes_no_question_brow_raise():
    g = E.gloss("Are you tired?")
    assert g.in_scope
    assert g.sentence_type == "yes_no_question"
    assert _text("Are you tired?") == "YOU TIRED"
    assert [(n.marker, n.start, n.end) for n in g.nmm_tags] == [("brow_raise", 0, 2)]


def test_wh_question_final_and_brow_furrow():
    g = E.gloss("What is your name?")
    assert g.in_scope
    assert g.sentence_type == "wh_question"
    assert [x.text for x in g.glosses] == ["YOUR", "NAME", "WHAT"]  # wh moved final
    assert [(n.marker, n.start, n.end) for n in g.nmm_tags] == [("brow_furrow", 0, 3)]


def test_negation_headshake_over_predicate():
    g = E.gloss("I don't want coffee.")
    assert g.in_scope
    assert g.negated is True
    assert _text("I don't want coffee.") == "ME WANT COFFEE"
    # headshake spans the predicate (verb -> end), not the subject
    assert [(n.marker, n.start, n.end) for n in g.nmm_tags] == [("headshake", 1, 3)]


def test_stacked_construction_lowers_confidence():
    """A wh-question that's ALSO negated stacks two individually-tested
    features -- confidence should reflect that, and BOTH tags should fire."""
    g = E.gloss("Why don't you go?")
    assert g.in_scope
    assert g.sentence_type == "wh_question" and g.negated
    markers = {(n.marker, n.start, n.end) for n in g.nmm_tags}
    assert ("brow_furrow", 0, 3) in markers
    assert ("headshake", 1, 3) in markers
    assert g.confidence < 1.0


# -- drop rules -------------------------------------------------------------

def test_drops_articles_and_copula():
    assert _text("The dog is hungry.") == "DOG HUNGRY"


def test_drops_infinitive_to():
    assert _text("Tomorrow I go to school.") == "TOMORROW ME GO SCHOOL"


# -- lexical mapping --------------------------------------------------------

def test_pronoun_and_possessive_collapse():
    assert _ids("I love you.") == ["me", "love", "you"]
    assert _ids("she works") == ["he", "work"]           # she -> HE gloss
    assert _ids("my book") == ["my", "book"]


def test_inflections_map_to_base_gloss():
    assert _ids("he ate") == ["he", "eat_1"]
    assert _ids("he is reading") == ["he", "read"]       # copula dropped, reading -> read


def test_multiword_phrase_merges_to_one_gloss():
    """Regression: _merged_tokens collapses "thank"+"you" into the combined
    lemma "thank_you" BEFORE lexicon lookup -- that combined string is already
    the target gloss id, not an English lemma, so it must resolve to itself
    or lexicalize() silently treats it as an unknown word. Previously refused
    "Thank you." as `unknown word(s): Thank` because the lexicon (built from
    curriculum.yaml's english_lemmas) never had a "thank_you" key."""
    g = E.gloss("Thank you.")
    assert g.in_scope, f"expected in-scope, got refused: {g.reason}"
    assert _ids("Thank you.") == ["thank_you"]
    assert _text("Thank you.") == "THANK-YOU"


# -- fail-closed --------------------------------------------------------

def test_unknown_content_word_refuses_the_whole_sentence():
    """Vocabulary gaps used to flag-and-keep an OOV placeholder; they must now
    refuse the WHOLE sentence -- never a partial, confidently-wrong gloss."""
    g = E.gloss("I want to eat pizza now.")
    assert g.in_scope is False
    assert g.glosses == [] and g.nmm_tags == []
    assert "pizza" in g.reason


def test_unsupported_construction_refuses():
    g = E.gloss("The book that I read is good.")  # relative clause
    assert g.in_scope is False
    assert "construction" in g.reason


# -- topicalization (opt-in) ------------------------------------------------

def test_topicalize_fronts_and_brow_raises():
    base = E.gloss("I want coffee.")          # ME WANT COFFEE
    topic = E.topicalize(base, 2)             # front COFFEE
    assert [x.text for x in topic.glosses] == ["COFFEE", "ME", "WANT"]
    assert topic.nmm_tags[0].marker == "brow_raise"
    assert (topic.nmm_tags[0].start, topic.nmm_tags[0].end) == (0, 1)


def test_topicalize_refuses_on_out_of_scope_input():
    base = E.gloss("I want to eat pizza now.")
    try:
        E.topicalize(base, 0)
        assert False, "expected ValueError topicalizing an out-of-scope sequence"
    except ValueError:
        pass


# -- convenience ------------------------------------------------------------

def test_module_convenience_matches_engine():
    assert gloss_sentence("I work today.").gloss_ids == _ids("I work today.")


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
