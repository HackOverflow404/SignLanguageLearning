"""Production track (Phase 5).

5b — the gloss rule engine (English -> ordered ASL gloss + non-manual tags) is
implemented here in `gloss_rules`. It is a written, inspectable ruleset, NOT a
trained model: ASL's reorder/drop/non-manual rules are enumerable, so they are
encoded directly.

5a — reference retrieval and pose-sequence composition depend on the Phase 1
cached sequences and are deferred until ASL Citizen is ingested. The rule engine
already emits the ordered gloss list those steps will consume.

NOTE: output is an approximation of a living language, gated on Deaf review
before anything built from it is shown to a learner as authoritative.
"""

from .gloss_rules import (
    Gloss,
    GlossSequence,
    GlossRuleEngine,
    NonManual,
    gloss_sentence,
)

__all__ = [
    "Gloss",
    "NonManual",
    "GlossSequence",
    "GlossRuleEngine",
    "gloss_sentence",
]
