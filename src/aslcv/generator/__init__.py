"""Phase 6 -- learner-facing content generation.

`feedback` phrases an already-graded attempt's verdicts as short English
coaching text. Templated for v1 (project_workflow.md's Phase 6 scoping
decision), not an LLM call -- CLAUDE.md's "an LLM only ever touches English"
rule means an LLM pass here would be a phrasing upgrade over this module's
output, never a replacement for the grading it summarizes.
"""
from .feedback import coach_text

__all__ = ["coach_text"]
