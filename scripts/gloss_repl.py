#!/usr/bin/env python3
"""gloss_repl.py -- interactive exploration of the Phase 5b gloss rule engine.

Type an English sentence; see EXACTLY what the engine did with it: the spaCy
parse driving the transforms, the resulting GlossSequence (ordered glosses,
NMM tags with their scope, in_scope, confidence, and -- if refused -- the
REASON), and on request the full rule-by-rule trace. This is a discovery
tool, not a demo -- everything is printed, nothing is terse for its own sake.

Commands:
    :quit / :q      exit
    :why            full step-by-step trace for the LAST sentence
    :help / :h      show commands

Anything else is treated as a sentence to gloss.

    .venv/bin/python scripts/gloss_repl.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aslcv.production.gloss_rules import GlossRuleEngine, GlossSequence, _nlp  # noqa: E402

BANNER = """\
Phase 5b gloss rule engine -- interactive REPL
Type an English sentence to gloss it. :help for commands, :quit to exit.
"""

HELP = """\
  :quit  / :q     exit
  :why            full rule-by-rule trace for the LAST sentence
  :help  / :h     this message
  (anything else) glossed as an English sentence
"""


def _print_spacy_parse(sentence: str) -> None:
    doc = _nlp()(sentence)
    print("spaCy parse (this is what drives the transforms):")
    if len(doc) == 0:
        print("  (no tokens)")
        return
    w_text = max(4, max(len(t.text) for t in doc))
    w_lemma = max(5, max(len(t.lemma_) for t in doc))
    w_pos = max(3, max(len(t.pos_) for t in doc))
    w_dep = max(3, max(len(t.dep_) for t in doc))
    header = (f"  {'TOKEN':<{w_text}}  {'LEMMA':<{w_lemma}}  {'POS':<{w_pos}}  "
              f"{'DEP':<{w_dep}}  HEAD")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in doc:
        head = t.head.text if t.head is not t else "(root)"
        print(f"  {t.text:<{w_text}}  {t.lemma_:<{w_lemma}}  {t.pos_:<{w_pos}}  "
              f"{t.dep_:<{w_dep}}  {head}")


def _print_result(seq: GlossSequence) -> None:
    print()
    print(f"in_scope   : {seq.in_scope}")
    print(f"confidence : {seq.confidence}")
    if not seq.in_scope:
        print(f"REASON     : {seq.reason}")
        print("           (fail-closed: no partial gloss is emitted for a refused sentence)")
        return
    print(f"sentence_type : {seq.sentence_type}")
    print(f"negated       : {seq.negated}")
    print()
    print("glosses (ordered):")
    for i, g in enumerate(seq.glosses):
        print(f"  [{i}] {g.text:<12} asllex_id={g.asllex_id:<12} pos={g.pos:<6} from {g.source!r}")
    print()
    if seq.nmm_tags:
        print("non-manual markers (scope shown against the gloss line above):")
        gloss_line = " ".join(g.text for g in seq.glosses)
        print(f"  {gloss_line}")
        for nm in seq.nmm_tags:
            covered = " ".join(g.text for g in seq.glosses[nm.start:nm.end])
            print(f"    {nm.marker:<11} [{nm.start}:{nm.end})  over: {covered}")
    else:
        print("non-manual markers: none")


def _print_trace(seq: GlossSequence | None) -> None:
    if seq is None:
        print("(nothing glossed yet)")
        return
    print(f"trace for: {seq.english!r}")
    for i, step in enumerate(seq.trace, 1):
        print(f"  {i}. {step}")


def main() -> None:
    print(BANNER)
    print("Loading spaCy model (en_core_web_sm)...", flush=True)
    engine = GlossRuleEngine()
    _nlp()  # force the lazy load now, not on the first sentence
    print("Ready.\n")

    last: GlossSequence | None = None
    while True:
        try:
            line = input("gloss> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q"):
            break
        if line in (":help", ":h"):
            print(HELP)
            continue
        if line == ":why":
            _print_trace(last)
            continue
        if line.startswith(":"):
            print(f"unknown command {line!r} -- :help for the list")
            continue

        print()
        _print_spacy_parse(line)
        last = engine.gloss(line)
        _print_result(last)
        print()


if __name__ == "__main__":
    main()
