"""Production track (Phase 5).

5b — the gloss rule engine (English -> ordered ASL gloss + non-manual tags) is
implemented here in `gloss_rules`. It is a written, inspectable ruleset, NOT a
trained model: ASL's reorder/drop/non-manual rules are enumerable, so they are
encoded directly.

5a — `retrieval` fetches the real reference clip(s) a GlossSequence resolves
to and concatenates them (video only -- see its module docstring for why a
concatenated pose-sequence grading target is deliberately not built yet).
Retrieval, never generation.

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
from .retrieval import (
    ComposedReference,
    ComposedReferenceFeatures,
    ReferenceClip,
    compose_reference_features,
    fetch_reference,
    fetch_sequence,
    write_composed_video,
)

__all__ = [
    "Gloss",
    "NonManual",
    "GlossSequence",
    "GlossRuleEngine",
    "gloss_sentence",
    "ReferenceClip",
    "ComposedReference",
    "ComposedReferenceFeatures",
    "fetch_reference",
    "fetch_sequence",
    "compose_reference_features",
    "write_composed_video",
]
