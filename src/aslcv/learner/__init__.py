"""Phase 6 -- the adaptive loop's memory of the learner.

`mastery` tracks per-sign, per-parameter mastery from graded attempts (never
from a classifier's guess -- every update traces back to a real
EmbeddingGrader.grade_against verdict). `scheduler` picks what to drill next
from that state: gap-targeting (weak parameters first) + a light recency bias
(spaced-repetition-flavored, not a full interval scheduler for v1) + automatic
contrastive minimal-pair drills when a specific parameter was just missed.
"""
from .mastery import MasteryState
from .scheduler import find_minimal_pairs, pick_next

__all__ = ["MasteryState", "find_minimal_pairs", "pick_next"]
