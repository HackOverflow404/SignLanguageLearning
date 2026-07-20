"""Phase 5b — English -> ordered ASL gloss + non-manual tags.

A written, inspectable ruleset (grammar is a rule engine, not a trained model).

SELF-CONTAINED BY DESIGN. This module depends on nothing external that can
change out from under it — no curriculum file, no dataset, no model. Its two
pieces of data are its own:
  * the grammar rules (drop / reorder / non-manual tagging), and
  * an embedded English->gloss lexicon keyed to ASL-LEX gloss ids (a fixed
    published standard, not the project's evolving curriculum).
Which of the emitted glosses happen to be in the current teaching vocabulary,
and whether a reference clip exists for one, are separate concerns handled by
the caller (task generator / retrieval), NOT here.

The pipeline, in order:

  1. normalize   : lowercase, join known phrases, strip punctuation, note a
                   trailing '?'.
  2. classify    : statement / yes-no question / wh-question, plus a negation flag.
  3. lexicalize  : map each English token to an ASL gloss via the embedded
                   lexicon; drop articles / copula / auxiliary "do" / infinitive
                   "to"; negators trigger the negation flag and are dropped.
  4. reorder     : time-first (time signs move to the front); wh-final (wh-sign
                   moves to clause end). Topic-comment is available but NOT applied
                   automatically (topic choice is context-dependent) -- see
                   GlossRuleEngine.topicalize.
  5. tag         : non-manual markers as spans over the gloss list --
                   yes/no -> brow_raise over the clause; wh -> brow_furrow over the
                   clause; negation -> headshake over the predicate.

Scope (matches the plan): reorder / drop / non-manual tagging only. Does NOT
handle classifiers, spatial agreement, or productive use of space.

The output `GlossedSentence` carries the ordered glosses (each keyed to an
ASL-LEX gloss id), so Phase 5a can later concatenate the matching reference pose
sequences to build a grading target.

NOTE: output is an approximation of a living language, gated on Deaf review
before anything built from it is shown to a learner as authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# The lexicon: English token -> ASL gloss id (ASL-LEX EntryID convention).
#
# Closed-class words and common inflections are enumerated here. This is the
# engine's own data; ASL-LEX ids are a fixed external standard.
# ---------------------------------------------------------------------------

# Multi-word phrases collapsed to a single token before tokenizing.
_PHRASES = {
    "thank you": "thankyou",
}

_LEXICON: dict[str, str] = {
    # pronouns (subject/object/possessive collapse to one gloss each)
    "i": "me", "me": "me",
    "my": "my", "mine": "my",
    "you": "you",
    "your": "your", "yours": "your",
    "he": "he", "she": "he", "him": "he", "her": "he", "his": "he",
    "we": "we", "us": "we", "our": "we", "ours": "we",
    # question words
    "what": "what_1",
    "who": "who", "whom": "who",
    "where": "where",
    "why": "why",
    "how": "how",
    # greetings / social
    "hello": "hello", "hi": "hello",
    "thankyou": "thank_you", "thanks": "thank_you", "thank": "thank_you",
    "please": "please",
    "sorry": "sorry",
    "fine": "fine_1",
    "name": "name",
    "yes": "yes", "yeah": "yes", "yep": "yes",
    "no": "no",
    "ok": "ok", "okay": "ok",
    # wants
    "want": "want_2", "wants": "want_2", "wanted": "want_2",
    # family / people
    "mother": "mother", "mom": "mother", "mommy": "mother",
    "father": "father", "dad": "father", "daddy": "father",
    "sister": "sister",
    "brother": "brother",
    "family": "family",
    "baby": "baby",
    "friend": "friend",
    # verbs (+ common inflections)
    "eat": "eat_1", "eats": "eat_1", "ate": "eat_1", "eating": "eat_1",
    "drink": "drink", "drinks": "drink", "drank": "drink", "drinking": "drink",
    "sleep": "sleep", "sleeps": "sleep", "slept": "sleep", "sleeping": "sleep",
    "work": "work", "works": "work", "worked": "work", "working": "work",
    "go": "go", "goes": "go", "went": "go", "going": "go",
    "come": "come", "comes": "come", "came": "come", "coming": "come",
    "learn": "learn", "learns": "learn", "learned": "learn", "learning": "learn",
    "read": "read", "reads": "read", "reading": "read",
    "love": "love", "loves": "love", "loved": "love",
    # food / drink
    "water": "water",
    "milk": "milk",
    "coffee": "coffee",
    # feelings / descriptions
    "happy": "happy",
    "sad": "sad",
    "tired": "tired",
    "hungry": "hungry",
    "good": "good",
    # time
    "today": "today",
    "tomorrow": "tomorrow",
    "now": "now",
    "time": "time",
    # colors
    "red": "red", "blue": "blue", "green": "green",
    "yellow": "yellow", "black": "black", "white": "white",
    # objects / places
    "book": "book", "books": "book",
    "car": "car", "cars": "car",
    "home": "home",
    "school": "school",
    "dog": "dog", "dogs": "dog",
}

# Function words removed from the gloss (articles, copula, auxiliary "do",
# infinitive marker).
_DROP = {
    "a", "an", "the",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "to",
}

# Tokens that trigger the negation flag (and are themselves dropped). Apostrophes
# are stripped before matching, so "don't" arrives as "dont".
_NEGATORS = {
    "not", "never", "dont", "doesnt", "didnt",
    "cant", "cannot", "wont", "isnt", "arent", "wasnt", "werent",
}

# Gloss ids grouped by grammatical role. WH drives wh-final reordering; TIME drives
# time-first reordering; VERBS anchors the negation (headshake) span.
_WH = {"what_1", "who", "where", "why", "how"}
_TIME = {"today", "tomorrow", "now", "time"}
_VERBS = {
    "want_2", "eat_1", "drink", "sleep", "work",
    "go", "come", "learn", "read", "love",
}

# Internal consistency (no external source of truth): every role gloss must be
# something the lexicon can actually emit.
_EMITTABLE = set(_LEXICON.values())
assert _WH <= _EMITTABLE, sorted(_WH - _EMITTABLE)
assert _TIME <= _EMITTABLE, sorted(_TIME - _EMITTABLE)
assert _VERBS <= _EMITTABLE, sorted(_VERBS - _EMITTABLE)


def _display_gloss(asllex_id: str) -> str:
    """Render an asllex_id as a conventional gloss: strip the `_<n>` sense
    suffix, turn remaining underscores into hyphens, uppercase.
    e.g. want_2 -> WANT, thank_you -> THANK-YOU, what_1 -> WHAT.
    """
    stem = re.sub(r"_\d+$", "", asllex_id)
    return stem.replace("_", "-").upper()


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gloss:
    """One sign in the ordered output."""

    text: str              # display gloss, e.g. "WANT"
    asllex_id: str | None  # ASL-LEX gloss id; None if the word is out of vocabulary
    source: str            # the English token it came from
    oov: bool = False      # True when no lexicon entry matched


@dataclass(frozen=True)
class NonManual:
    """A non-manual marker spanning a half-open range of gloss indices."""

    marker: str        # "brow_raise" | "brow_furrow" | "headshake"
    start: int         # inclusive
    end: int           # exclusive


@dataclass
class GlossedSentence:
    english: str
    sentence_type: str  # "statement" | "yes_no_question" | "wh_question"
    negated: bool
    glosses: list[Gloss] = field(default_factory=list)
    non_manuals: list[NonManual] = field(default_factory=list)

    @property
    def gloss_ids(self) -> list[str | None]:
        """Ordered asllex_ids — what Phase 5a concatenates into a pose target."""
        return [g.asllex_id for g in self.glosses]

    @property
    def oov(self) -> list[str]:
        """English tokens with no gloss (a diagnostic, not a failure)."""
        return [g.source for g in self.glosses if g.oov]

    def render(self) -> str:
        """Human-readable gloss line plus one line per non-manual span."""
        line = " ".join(g.text for g in self.glosses) or "(empty)"
        out = [line]
        for nm in self.non_manuals:
            covered = " ".join(g.text for g in self.glosses[nm.start:nm.end])
            out.append(f"    {nm.marker:<11} over: {covered}")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "english": self.english,
            "sentence_type": self.sentence_type,
            "negated": self.negated,
            "glosses": [
                {"text": g.text, "asllex_id": g.asllex_id, "source": g.source, "oov": g.oov}
                for g in self.glosses
            ],
            "non_manuals": [
                {"marker": nm.marker, "start": nm.start, "end": nm.end}
                for nm in self.non_manuals
            ],
        }

    def __str__(self) -> str:
        return self.render()


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class GlossRuleEngine:
    """English -> GlossedSentence. Fully self-contained; the optional overrides
    default to the module's embedded lexicon / verb set."""

    def __init__(
        self,
        lexicon: dict[str, str] | None = None,
        verbs: set[str] | None = None,
    ):
        self.lexicon = _LEXICON if lexicon is None else lexicon
        self.verbs = _VERBS if verbs is None else verbs

    # -- step 1: normalize --------------------------------------------------

    @staticmethod
    def _normalize(sentence: str) -> tuple[list[str], bool]:
        text = sentence.strip().lower()
        is_question = text.endswith("?")
        for phrase, token in _PHRASES.items():
            text = text.replace(phrase, token)
        text = text.replace("'", "")           # don't -> dont, name's -> names
        text = re.sub(r"[^\w\s]", " ", text)    # drop remaining punctuation
        return text.split(), is_question

    # -- step 3: lexicalize (also flags negation) ---------------------------

    def _lexicalize(self, tokens: list[str]) -> tuple[list[Gloss], bool]:
        glosses: list[Gloss] = []
        negated = False
        for tok in tokens:
            if tok in _NEGATORS:
                negated = True
                continue
            if tok in _DROP:
                continue
            asllex_id = self.lexicon.get(tok)
            if asllex_id is None:
                # content word with no gloss: keep it, flagged, so the gap is
                # visible rather than silently dropped.
                glosses.append(Gloss(text=tok.upper(), asllex_id=None, source=tok, oov=True))
            else:
                glosses.append(Gloss(text=_display_gloss(asllex_id), asllex_id=asllex_id, source=tok))
        return glosses, negated

    # -- step 4: reorder ----------------------------------------------------

    @staticmethod
    def _reorder(glosses: list[Gloss], sentence_type: str) -> list[Gloss]:
        # time-first: stable-partition time signs to the front
        time = [g for g in glosses if g.asllex_id in _TIME]
        rest = [g for g in glosses if g.asllex_id not in _TIME]
        ordered = time + rest
        # wh-final: move wh signs to the end (stable)
        if sentence_type == "wh_question":
            non_wh = [g for g in ordered if g.asllex_id not in _WH]
            wh = [g for g in ordered if g.asllex_id in _WH]
            ordered = non_wh + wh
        return ordered

    def topicalize(self, sent: GlossedSentence, index: int) -> GlossedSentence:
        """Optional topic-comment transform: front the gloss at `index` as the
        topic and mark it with brow_raise. NOT applied automatically because
        choosing the topic is context-dependent; call it explicitly when the
        task generator wants a topicalized target."""
        if not (0 <= index < len(sent.glosses)):
            raise IndexError(index)
        g = sent.glosses[index]
        reordered = [g] + sent.glosses[:index] + sent.glosses[index + 1:]
        # topic occupies position 0; question/negation tags recomputed on the
        # new order, with the topic itself carrying brow_raise.
        tags = [NonManual("brow_raise", 0, 1)]
        tags += self._tag(reordered, sent.sentence_type, sent.negated, skip_question=True)
        return GlossedSentence(
            english=sent.english,
            sentence_type=sent.sentence_type,
            negated=sent.negated,
            glosses=reordered,
            non_manuals=tags,
        )

    # -- step 5: non-manual tagging ----------------------------------------

    def _first_verb(self, glosses: list[Gloss]) -> int | None:
        for i, g in enumerate(glosses):
            if g.asllex_id in self.verbs:
                return i
        return None

    def _tag(self, glosses: list[Gloss], sentence_type: str, negated: bool,
             skip_question: bool = False) -> list[NonManual]:
        tags: list[NonManual] = []
        n = len(glosses)
        if n == 0:
            return tags
        if not skip_question:
            if sentence_type == "yes_no_question":
                tags.append(NonManual("brow_raise", 0, n))
            elif sentence_type == "wh_question":
                tags.append(NonManual("brow_furrow", 0, n))
        if negated:
            # headshake over the predicate: first verb to end, else whole clause
            v = self._first_verb(glosses)
            start = v if v is not None else 0
            tags.append(NonManual("headshake", start, n))
        return tags

    # -- orchestration ------------------------------------------------------

    def gloss(self, sentence: str) -> GlossedSentence:
        tokens, is_question = self._normalize(sentence)
        has_wh = any(self.lexicon.get(t) in _WH for t in tokens)
        if has_wh:
            sentence_type = "wh_question"
        elif is_question:
            sentence_type = "yes_no_question"
        else:
            sentence_type = "statement"

        glosses, negated = self._lexicalize(tokens)
        glosses = self._reorder(glosses, sentence_type)
        non_manuals = self._tag(glosses, sentence_type, negated)
        return GlossedSentence(
            english=sentence,
            sentence_type=sentence_type,
            negated=negated,
            glosses=glosses,
            non_manuals=non_manuals,
        )


# module-level convenience (builds a default engine once)
_default_engine: GlossRuleEngine | None = None


def gloss_sentence(sentence: str) -> GlossedSentence:
    """Gloss one English sentence with the default (embedded-lexicon) engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = GlossRuleEngine()
    return _default_engine.gloss(sentence)


if __name__ == "__main__":
    engine = GlossRuleEngine()
    examples = [
        "I work today.",
        "Are you tired?",
        "What is your name?",
        "I don't want coffee.",
        "The dog is hungry.",        # article + copula dropped
        "Tomorrow I go to school.",  # time-first + "to" dropped
        "Where is my mother?",
        "We love you.",
        "I want to eat pizza now.",  # 'pizza' is out of vocabulary -> flagged
    ]
    for s in examples:
        g = engine.gloss(s)
        print(f"\nEN: {s}")
        print(f"   type={g.sentence_type} negated={g.negated}"
              + (f" oov={g.oov}" if g.oov else ""))
        print("   " + g.render().replace("\n", "\n   "))
