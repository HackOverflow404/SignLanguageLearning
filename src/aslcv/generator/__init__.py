"""Phase 6 -- learner-facing content generation.

`feedback` phrases an already-graded attempt's verdicts as short English
coaching text -- templated, the v1 default (project_workflow.md's Phase 6
scoping decision). `llm_feedback` is the optional upgrade: an LLM phrases the
SAME pre-computed facts more naturally, never adds ASL knowledge of its own,
and fails open to `feedback.coach_text` on any error/missing key -- CLAUDE.md's
"an LLM only ever touches English" rule, enforced structurally, not just by
convention.
"""
from .feedback import coach_text, focus_parameter
from .llm_feedback import coach_text_maybe_llm, llm_coach_text

__all__ = ["coach_text", "focus_parameter", "coach_text_maybe_llm", "llm_coach_text"]
