"""Phase 6 -- learner-facing content generation.

`feedback` phrases an already-graded attempt's verdicts as short English
coaching text -- templated, the v1 default (project_workflow.md's Phase 6
scoping decision). `llm_feedback` is the optional upgrade: an LLM phrases the
SAME pre-computed facts more naturally, never adds ASL knowledge of its own,
and fails open to `feedback.coach_text` on any error/missing key -- CLAUDE.md's
"an LLM only ever touches English" rule, enforced structurally, not just by
convention. `sentence_prompts` is Phase 6 v2's sentence-prompt generator: an
LLM writes an English sentence containing the target word, and Phase 5b's
fail-closed gloss rule engine -- not the LLM -- decides whether it ever
becomes a displayed prompt.
"""
from .feedback import coach_text, focus_parameter
from .llm_feedback import coach_text_maybe_llm, llm_coach_text
from .sentence_prompts import sentence_prompt_maybe_llm

__all__ = [
    "coach_text", "focus_parameter", "coach_text_maybe_llm", "llm_coach_text",
    "sentence_prompt_maybe_llm",
]
