"""Regression tests for the Phase 5b gloss rule engine.

Runs under pytest OR as a plain script (`python tests/test_gloss_rules.py`) so it
needs no extra dependency. The engine is meant to be a stable, self-contained
component, so these pin its behavior.
"""

from aslcv.production.gloss_rules import GlossRuleEngine, gloss_sentence
from aslcv.production import gloss_rules as gr

E = GlossRuleEngine()


def _ids(sentence):
    return E.gloss(sentence).gloss_ids


def _text(sentence):
    return " ".join(g.text for g in E.gloss(sentence).glosses)


# -- self-containment: no external, mutable dependency ----------------------

def test_no_curriculum_or_yaml_dependency():
    # the module must not have pulled in yaml or a curriculum path
    assert not hasattr(gr, "yaml")
    assert not hasattr(gr, "_DEFAULT_CURRICULUM")
    # and it must construct + run with no arguments and no file present
    assert GlossRuleEngine().gloss("I work today.").glosses


def test_role_sets_are_internally_consistent():
    emittable = set(gr._LEXICON.values())
    assert gr._WH <= emittable
    assert gr._TIME <= emittable
    assert gr._VERBS <= emittable


# -- the five curriculum v2 constructions -----------------------------------

def test_time_first():
    assert _text("I work today.") == "TODAY ME WORK"


def test_yes_no_question_brow_raise():
    g = E.gloss("Are you tired?")
    assert g.sentence_type == "yes_no_question"
    assert _text("Are you tired?") == "YOU TIRED"
    assert [(n.marker, n.start, n.end) for n in g.non_manuals] == [("brow_raise", 0, 2)]


def test_wh_question_final_and_brow_furrow():
    g = E.gloss("What is your name?")
    assert g.sentence_type == "wh_question"
    assert [x.text for x in g.glosses] == ["YOUR", "NAME", "WHAT"]  # wh moved final
    assert [(n.marker, n.start, n.end) for n in g.non_manuals] == [("brow_furrow", 0, 3)]


def test_negation_headshake_over_predicate():
    g = E.gloss("I don't want coffee.")
    assert g.negated is True
    assert _text("I don't want coffee.") == "ME WANT COFFEE"
    # headshake spans the predicate (verb -> end), not the subject
    assert [(n.marker, n.start, n.end) for n in g.non_manuals] == [("headshake", 1, 3)]


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


# -- out of vocabulary ------------------------------------------------------

def test_oov_flagged_not_dropped():
    g = E.gloss("I want to eat pizza now.")
    assert g.oov == ["pizza"]
    assert _text("I want to eat pizza now.") == "NOW ME WANT EAT PIZZA"


# -- topicalization (opt-in) ------------------------------------------------

def test_topicalize_fronts_and_brow_raises():
    base = E.gloss("I want coffee.")          # ME WANT COFFEE
    topic = E.topicalize(base, 2)             # front COFFEE
    assert [x.text for x in topic.glosses] == ["COFFEE", "ME", "WANT"]
    assert topic.non_manuals[0].marker == "brow_raise"
    assert (topic.non_manuals[0].start, topic.non_manuals[0].end) == (0, 1)


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
