"""Phase 5b — English -> ordered ASL gloss + non-manual tags.

A written, inspectable ruleset (grammar is a rule engine, not a trained model),
implementing the settled interface from project_workflow.md 5b:

    english -> GlossSequence(glosses, nmm_tags, in_scope, confidence)

THREE LAYERS, kept separate on purpose (project_workflow.md: "treat it as
non-negotiable"):
  1. English analysis  — spaCy (POS, lemma, dependency parse). Standard,
     maintained, replaceable; this module never re-derives what spaCy already
     tells it (e.g. negation comes from spaCy's `neg` dependency label, not a
     hand-maintained word list).
  2. The ruleset        — drop / reorder / non-manual tagging, expressed as the
     small role-sets and functions below. Still Python, not yet the declarative
     data format project_workflow.md eventually wants (a Deaf reviewer editing
     rules without touching code) -- that split is real future work, not done
     here.
  3. Lexicon mapping     — English lemma -> ASL-LEX EntryID, auto-derived from
     curriculum.yaml's `english_lemmas` field (see `_build_lexicon`) plus a
     small supplementary set of colloquial synonyms curriculum.yaml doesn't
     list. Every value is therefore, by construction, one of the 60 curriculum
     signs -- see `_curriculum_glosses()` and its self-check at import time.

FAIL CLOSED — the reliability principle that matters most here. The engine
refuses (`in_scope=False`, `reason` populated, empty `glosses`) rather than
emit a confident wrong gloss, on either of two independent triggers:
  * a CONTENT word (noun/verb/adjective/adverb/numeral/pronoun by spaCy POS)
    has no lexicon entry -- there is no vocabulary to compose it from.
  * the sentence's dependency parse contains a construction outside the
    supported set (relative clause, subordinate clause, clausal complement,
    passive voice, or clause coordination) -- these are "complex embedding",
    explicitly out of scope per project_workflow.md.
An empty sentence, or one that reduces to zero content words after dropping
function words, is ALSO refused: there is nothing to compose a target from,
and a silently-empty GlossSequence would look like a valid (if boring) result.

The pipeline, in order:

  1. parse       : spaCy tokenizes/lemmatizes/POS-tags/dependency-parses the
                   RAW sentence once. Everything downstream reads this parse;
                   nothing re-tokenizes or re-analyzes English independently.
  2. scope check : dependency labels (relcl/advcl/ccomp/(nsub|aux)pass, or a
                   VERB-headed `conj`) -> refuse with a reason naming which one.
  3. lexicalize  : each surviving token's LEMMA looked up in the curriculum
                   lexicon; function words (articles/copula/auxiliary "do"/
                   infinitive "to") dropped; negation read off spaCy's `neg`
                   dependency label (drops the negator, sets a flag); any
                   unmapped CONTENT word -> refuse, naming the word(s).
  4. classify    : statement / yes-no question / wh-question, from a trailing
                   "?" and whether a WH gloss survived lexicalization.
  5. reorder     : time-first (time signs move to the front); wh-final (wh-sign
                   moves to clause end). Topic-comment is available but NOT
                   applied automatically (topic choice is context-dependent)
                   -- see GlossRuleEngine.topicalize.
  6. tag         : non-manual markers as spans over the gloss list --
                   yes/no -> brow_raise over the clause; wh -> brow_furrow over
                   the clause; negation -> headshake over the predicate (first
                   VERB-sourced gloss to the end).
  7. confidence  : a heuristic, NOT a calibrated probability (documented on
                   `_confidence` below) -- flags outputs that stack more
                   simultaneous constructions than curriculum.yaml's own
                   single-feature examples individually validate.

Scope (matches the plan): reorder / drop / non-manual tagging only. Does NOT
handle classifiers, spatial agreement, role shift, aspectual movement
modulation, or productive use of space.

The output `GlossSequence` carries the ordered glosses (each keyed to an
ASL-LEX gloss id / curriculum asllex_code), so Phase 5a can later concatenate
the matching reference pose sequences to build a grading target; see
scripts/compose_sentence.py for the retrieval-side proof that this resolves.

NOTE: output is an approximation of a living language, gated on Deaf review
before anything built from it is shown to a learner as authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import spacy
import yaml

REPO = Path(__file__).resolve().parents[3]
CURRICULUM_PATH = REPO / "curriculum.yaml"

# ---------------------------------------------------------------------------
# spaCy: loaded lazily (a ~3s cost) and cached -- importing this module (e.g.
# for the Gloss/GlossSequence types alone) should not pay it until a caller
# actually glosses a sentence. NER is excluded: unused here, and dropping it
# skips a real chunk of the pipeline's inference cost.
# ---------------------------------------------------------------------------

_NLP_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _nlp():
    return spacy.load(_NLP_MODEL, exclude=["ner"])


# ---------------------------------------------------------------------------
# The lexicon: English lemma -> ASL-LEX gloss id, auto-derived from
# curriculum.yaml so it can never drift ahead of what's actually composable
# (every value is a real asllex_code -- see the self-check below).
# ---------------------------------------------------------------------------

# Multi-word phrases matched as CONSECUTIVE LEMMAS after spaCy parses the raw
# sentence (spaCy's own tokenizer never merges these; English "thank" + "you"
# stay two tokens, but ASL THANK-YOU is one sign).
_PHRASES: dict[tuple[str, ...], str] = {
    ("thank", "you"): "thank_you",
}

# Colloquial synonyms curriculum.yaml's own `english_lemmas` doesn't list.
# Every value must still be a real curriculum gloss (asserted below) -- this
# widens the SURFACE forms the engine accepts, never the sign vocabulary.
_SUPPLEMENTARY_LEXICON: dict[str, str] = {
    "his": "he", "her": "he",  # spaCy does not lemmatize possessive pronouns
    "our": "we", "ours": "we",
    "daddy": "father", "mommy": "mother",
    "yeah": "yes", "yep": "yes",
}

# Function words dropped from the gloss (articles, copula, auxiliary "do",
# infinitive marker). Cross-checked against spaCy POS at lexicalize time, not
# just this fixed list -- see _lexicalize.
_DROP = {
    "a", "an", "the",
    "be", "do",  # lemma form covers am/is/are/was/were/been/being/does/did
    "to",
}

# POS tags counted as CONTENT words: an unmapped token with one of these tags
# is what triggers fail-closed refusal (project_workflow.md: "keep possessive/
# personal pronouns plus nouns, verbs, adjectives, adverbs, numerals").
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "PRON"}

# Dependency labels marking a construction outside this engine's scope
# ("complex embedding" in project_workflow.md's out-of-scope list). `conj` is
# only a problem when it coordinates a second CLAUSE -- noun/adjective
# coordination ("rice and beans", "happy and tired") stays in scope. A second
# clause's head is VERB ("I go home and you go school") OR AUX when the
# second clause is copula-based ("I am tired but you are happy" -- "are" is
# tagged AUX, not VERB, since spaCy treats copulas as auxiliaries). Checking
# only VERB let this second class of coordinated clause silently through --
# found via scripts/export_review_sheet.py's fail-closed probing, see
# tests/test_gloss_rules_corpus.py's test_aux_headed_clause_coordination_refuses.
_OUT_OF_SCOPE_DEPS = {"relcl", "advcl", "ccomp", "auxpass", "nsubjpass"}
_CLAUSE_HEAD_POS = {"VERB", "AUX"}


def _curriculum_signs() -> list[dict]:
    doc = yaml.safe_load(open(CURRICULUM_PATH))
    return [s for unit in doc["units"] for s in unit["signs"]]


def _build_lexicon(signs: list[dict] | None = None) -> dict[str, str]:
    """lemma -> curriculum gloss, derived from curriculum.yaml's own
    `english_lemmas` per sign, plus _SUPPLEMENTARY_LEXICON. NOT hand-typed:
    editing curriculum.yaml is what changes the vocabulary this engine
    accepts, so the two can never silently drift apart.

    FAILS CLOSED on an ambiguous lemma -- two DIFFERENT curriculum signs (or
    a curriculum sign and a _SUPPLEMENTARY_LEXICON entry) claiming the same
    English lemma. There is no tie-break rule; a plain dict-merge would let
    whichever one is processed last silently win, so a future curriculum.yaml
    edit could quietly change what an existing sentence glosses to. Raising
    here means that has to be resolved by a human (rename one entry's
    english_lemmas, or drop the supplementary synonym) before the module
    loads at all -- an ambiguity can never reach a learner silently.

    `signs` defaults to the real curriculum; overridable so a test can feed a
    synthetic conflicting pair without needing a second curriculum.yaml."""
    if signs is None:
        signs = _curriculum_signs()
    lexicon: dict[str, str] = {}
    for sign in signs:
        for lemma in sign.get("english_lemmas", []):
            if lemma in lexicon and lexicon[lemma] != sign["gloss"]:
                raise ValueError(
                    f"ambiguous lexicon entry: english lemma {lemma!r} maps to both "
                    f"{lexicon[lemma]!r} and {sign['gloss']!r} in curriculum.yaml -- "
                    f"no silent tie-break exists; rename one entry's english_lemmas")
            lexicon[lemma] = sign["gloss"]
    for lemma, gloss in _SUPPLEMENTARY_LEXICON.items():
        if lemma in lexicon and lexicon[lemma] != gloss:
            raise ValueError(
                f"_SUPPLEMENTARY_LEXICON entry {lemma!r} -> {gloss!r} conflicts with "
                f"curriculum.yaml's own mapping {lemma!r} -> {lexicon[lemma]!r}")
        lexicon[lemma] = gloss
    # _merged_tokens replaces a matched _PHRASES key with its COMBINED value
    # (e.g. "thank"+"you" -> "thank_you") before lexicalize() ever looks it
    # up -- that combined value is already the target gloss id, not an
    # English lemma, so it must resolve to itself here or lexicalize() would
    # never find it (the bug this line fixes: "Thank you." was refusing as
    # an unknown word).
    for combined in _PHRASES.values():
        lexicon[combined] = combined
    return lexicon


_LEXICON: dict[str, str] = _build_lexicon()

# Gloss ids grouped by grammatical role. WH drives wh-final reordering + the
# wh-question brow_furrow tag; TIME drives time-first reordering. Small,
# explicit sets (not POS-derived): "is this specifically an ASL wh-sign" isn't
# a clean POS category in English, and these are exactly the gloss ids that
# matter for the transforms below, nothing more.
_WH = {"what_1", "who", "where", "why", "how"}
_TIME = {"today", "tomorrow", "now", "time"}

# Internal consistency (no external source of truth beyond curriculum.yaml):
# every role gloss must be something the lexicon can actually emit.
_EMITTABLE = set(_LEXICON.values())
_CURRICULUM_GLOSSES = {s["gloss"] for s in _curriculum_signs()}
assert _WH <= _EMITTABLE, sorted(_WH - _EMITTABLE)
assert _TIME <= _EMITTABLE, sorted(_TIME - _EMITTABLE)
assert _EMITTABLE <= _CURRICULUM_GLOSSES, sorted(_EMITTABLE - _CURRICULUM_GLOSSES)


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

    text: str          # display gloss, e.g. "WANT"
    asllex_id: str      # ASL-LEX gloss id / curriculum asllex_code
    source: str         # the English token it came from (surface form)
    pos: str             # the source token's spaCy POS tag (drives _first_verb)


@dataclass(frozen=True)
class NonManual:
    """A non-manual marker spanning a half-open range of gloss indices."""

    marker: str        # "brow_raise" | "brow_furrow" | "headshake"
    start: int         # inclusive
    end: int            # exclusive


@dataclass
class GlossSequence:
    """english -> GlossSequence(glosses, nmm_tags, in_scope, confidence) --
    the settled Phase 5b interface (project_workflow.md). `reason` is
    populated iff `in_scope` is False; `glosses`/`nmm_tags` are empty in that
    case (fail-closed: never a partial, confidently-wrong output). `trace` is
    a step-by-step log of which pipeline stage did what -- what
    scripts/gloss_repl.py's `:why` prints.
    """

    english: str
    in_scope: bool
    confidence: float
    reason: str | None
    sentence_type: str | None  # "statement" | "yes_no_question" | "wh_question"
    negated: bool
    glosses: list[Gloss] = field(default_factory=list)
    nmm_tags: list[NonManual] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    @property
    def gloss_ids(self) -> list[str]:
        """Ordered asllex_ids — what Phase 5a concatenates into a pose target."""
        return [g.asllex_id for g in self.glosses]

    def render(self) -> str:
        """Human-readable gloss line plus one line per non-manual span, or the
        refusal reason if not in scope."""
        if not self.in_scope:
            return f"REFUSED: {self.reason}"
        line = " ".join(g.text for g in self.glosses) or "(empty)"
        out = [line]
        for nm in self.nmm_tags:
            covered = " ".join(g.text for g in self.glosses[nm.start:nm.end])
            out.append(f"    {nm.marker:<11} over: {covered}")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "english": self.english,
            "in_scope": self.in_scope,
            "confidence": self.confidence,
            "reason": self.reason,
            "sentence_type": self.sentence_type,
            "negated": self.negated,
            "glosses": [
                {"text": g.text, "asllex_id": g.asllex_id, "source": g.source, "pos": g.pos}
                for g in self.glosses
            ],
            "nmm_tags": [
                {"marker": nm.marker, "start": nm.start, "end": nm.end}
                for nm in self.nmm_tags
            ],
        }

    def __str__(self) -> str:
        return self.render()


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class GlossRuleEngine:
    """English -> GlossSequence. `lexicon` defaults to the curriculum-derived
    module lexicon; pass an override only for testing a hypothetical
    vocabulary -- production use should always take the default, since that is
    what guarantees every emitted gloss is retrievable (see
    scripts/compose_sentence.py)."""

    def __init__(self, lexicon: dict[str, str] | None = None):
        self.lexicon = _LEXICON if lexicon is None else lexicon

    # -- scope check ----------------------------------------------------

    @staticmethod
    def _scope_problems(doc) -> list[str]:
        problems = []
        for tok in doc:
            if tok.dep_ in _OUT_OF_SCOPE_DEPS:
                problems.append(f"{tok.dep_} ('{tok.text}')")
            elif tok.dep_ == "conj" and tok.pos_ in _CLAUSE_HEAD_POS:
                problems.append(f"clause coordination ('{tok.text}')")
        return problems

    # -- phrase merge + lexicalize ---------------------------------------

    def _merged_tokens(self, doc) -> list:
        """spaCy tokens with _PHRASES collapsed: consecutive tokens whose
        lemmas match a phrase key become ONE entry (the first token, tagged
        with the phrase's combined lemma) so lexicalize sees "thank you" as
        the single unit ASL treats it as. Scope-checking already ran on the
        UNMERGED doc, so this never hides a real dependency-parse signal."""
        tokens = list(doc)
        lemmas = [t.lemma_.lower() for t in tokens]
        merged = []
        i = 0
        while i < len(tokens):
            hit = None
            for phrase, combined in _PHRASES.items():
                n = len(phrase)
                if tuple(lemmas[i:i + n]) == phrase:
                    hit = (combined, n)
                    break
            if hit:
                combined, n = hit
                merged.append((combined, tokens[i]))
                i += n
            else:
                merged.append((lemmas[i], tokens[i]))
                i += 1
        return merged

    def _lexicalize(self, merged) -> tuple[list[Gloss], bool, list[str]]:
        glosses: list[Gloss] = []
        negated = False
        unknown: list[str] = []
        for lemma, tok in merged:
            if tok.dep_ == "neg":
                negated = True
                continue
            if lemma in _DROP or tok.pos_ in ("DET", "AUX", "PART", "PUNCT", "SPACE"):
                continue
            asllex_id = self.lexicon.get(lemma)
            if asllex_id is not None:
                glosses.append(Gloss(
                    text=_display_gloss(asllex_id), asllex_id=asllex_id,
                    source=tok.text, pos=tok.pos_))
            elif tok.pos_ in _CONTENT_POS:
                unknown.append(tok.text)
            # else: a content-adjacent token spaCy tagged something else
            # (e.g. INTJ, CCONJ) with no lexicon entry -- silently dropped,
            # not a vocabulary gap worth refusing over.
        return glosses, negated, unknown

    # -- reorder ----------------------------------------------------------

    @staticmethod
    def _reorder(glosses: list[Gloss], sentence_type: str) -> list[Gloss]:
        time = [g for g in glosses if g.asllex_id in _TIME]
        rest = [g for g in glosses if g.asllex_id not in _TIME]
        ordered = time + rest
        if sentence_type == "wh_question":
            non_wh = [g for g in ordered if g.asllex_id not in _WH]
            wh = [g for g in ordered if g.asllex_id in _WH]
            ordered = non_wh + wh
        return ordered

    def topicalize(self, seq: GlossSequence, index: int) -> GlossSequence:
        """Optional topic-comment transform: front the gloss at `index` as the
        topic and mark it with brow_raise. NOT applied automatically because
        choosing the topic is context-dependent; call it explicitly when the
        task generator wants a topicalized target. Refuses to touch an
        out-of-scope sequence (nothing to topicalize)."""
        if not seq.in_scope:
            raise ValueError(f"cannot topicalize an out-of-scope sequence: {seq.reason}")
        if not (0 <= index < len(seq.glosses)):
            raise IndexError(index)
        g = seq.glosses[index]
        reordered = [g] + seq.glosses[:index] + seq.glosses[index + 1:]
        tags = [NonManual("brow_raise", 0, 1)]
        tags += self._tag(reordered, seq.sentence_type, seq.negated, skip_question=True)
        return GlossSequence(
            english=seq.english, in_scope=True, confidence=seq.confidence, reason=None,
            sentence_type=seq.sentence_type, negated=seq.negated,
            glosses=reordered, nmm_tags=tags,
            trace=seq.trace + [f"topicalize({index}): fronted {g.text!r}, brow_raise[0:1]"],
        )

    # -- non-manual tagging -------------------------------------------------

    @staticmethod
    def _first_verb(glosses: list[Gloss]) -> int | None:
        for i, g in enumerate(glosses):
            if g.pos == "VERB":
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
            v = self._first_verb(glosses)
            start = v if v is not None else 0
            tags.append(NonManual("headshake", start, n))
        return tags

    # -- confidence -----------------------------------------------------

    @staticmethod
    def _confidence(sentence_type: str, negated: bool, time_fronted: bool) -> float:
        """A HEURISTIC, not a calibrated probability. curriculum.yaml's
        constructions_v2 examples each validate exactly ONE feature in
        isolation (a plain time-first sentence, OR a plain yes/no question,
        OR a plain negation, ...). This counts how many of those individually-
        tested features are stacked in THIS output and discounts each one
        beyond the first -- flagging "this specific combination hasn't itself
        been checked against curriculum.yaml's examples," not "this is wrong."
        """
        features = int(sentence_type != "statement") + int(negated) + int(time_fronted)
        if features <= 1:
            return 1.0
        return max(0.4, 1.0 - 0.15 * (features - 1))

    # -- orchestration ------------------------------------------------------

    def gloss(self, sentence: str) -> GlossSequence:
        trace = [f"input: {sentence!r}"]
        if not sentence.strip():
            trace.append("refused: empty input")
            return GlossSequence(sentence, False, 0.0, "empty input", None, False, trace=trace)

        doc = _nlp()(sentence)
        trace.append(f"spacy: {len(doc)} tokens -- "
                      + ", ".join(f"{t.text}/{t.pos_}/{t.dep_}" for t in doc))

        problems = self._scope_problems(doc)
        if problems:
            reason = "unsupported construction: " + "; ".join(problems)
            trace.append(f"scope check: refused -- {reason}")
            return GlossSequence(sentence, False, 0.0, reason, None, False, trace=trace)
        trace.append("scope check: no unsupported dependency labels")

        merged = self._merged_tokens(doc)
        glosses, negated, unknown = self._lexicalize(merged)
        if unknown:
            reason = "unknown word(s): " + ", ".join(unknown)
            trace.append(f"lexicalize: refused -- {reason}")
            return GlossSequence(sentence, False, 0.0, reason, None, negated, trace=trace)
        trace.append(f"lexicalize: {[g.text for g in glosses]} (negated={negated})")

        if not glosses:
            trace.append("refused: no content words survived (nothing to sign)")
            return GlossSequence(sentence, False, 0.0, "no content words -- nothing to sign",
                                  None, negated, trace=trace)

        is_question = sentence.strip().endswith("?")
        has_wh = any(g.asllex_id in _WH for g in glosses)
        sentence_type = "wh_question" if has_wh else ("yes_no_question" if is_question else "statement")
        trace.append(f"classify: sentence_type={sentence_type}")

        time_fronted = any(g.asllex_id in _TIME for g in glosses)  # _reorder always fronts it
        ordered = self._reorder(glosses, sentence_type)
        trace.append(f"reorder: {[g.text for g in ordered]}")

        nmm_tags = self._tag(ordered, sentence_type, negated)
        trace.append(f"tag: {[(n.marker, n.start, n.end) for n in nmm_tags]}")

        confidence = self._confidence(sentence_type, negated, time_fronted)
        trace.append(f"confidence={confidence}")

        return GlossSequence(
            english=sentence, in_scope=True, confidence=confidence, reason=None,
            sentence_type=sentence_type, negated=negated,
            glosses=ordered, nmm_tags=nmm_tags, trace=trace,
        )


# module-level convenience (builds a default engine once)
_default_engine: GlossRuleEngine | None = None


def gloss_sentence(sentence: str) -> GlossSequence:
    """Gloss one English sentence with the default (curriculum-lexicon) engine."""
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
        "The dog is hungry.",          # article + copula dropped
        "Tomorrow I go to school.",    # time-first + "to" dropped
        "Where is my mother?",
        "We love you.",
        "I want to eat pizza now.",    # 'pizza' out of vocabulary -> refused
        "The book that I read is good.",  # relative clause -> refused
    ]
    for s in examples:
        g = engine.gloss(s)
        print(f"\nEN: {s}")
        if g.in_scope:
            print(f"   type={g.sentence_type} negated={g.negated} confidence={g.confidence}")
        print("   " + g.render().replace("\n", "\n   "))
