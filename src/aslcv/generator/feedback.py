"""Templated English coaching text from a graded attempt's per-parameter verdicts.

Duck-typed on purpose: takes anything with `.parameter`/`.correct`/`.confidence`/
`.predicted`/`.target` attributes (real usage passes `GradeResult.parameters.values()`,
i.e. `ParameterVerdict` objects) without importing that dataclass -- this module has
no reason to depend on the grading package, only on the shape of its output.

`.predicted` (what the model classified the attempt's own value as) and
`.target` (the TARGET sign's true grounded ASL-LEX/curriculum value) are both
already computed by the grader -- this module's whole job is surfacing them,
never inventing anything: "you signed X, the target is Y" is strictly a
readout of two pre-computed, grounded facts, not ASL knowledge this module
originates.
"""
from __future__ import annotations

PARAM_NAME = {
    "handshape": "handshape",
    "major_location": "general location",
    "minor_location": "exact location",
    "movement": "movement path",
    "repeated_movement": "movement repetition",
}

PARAM_TIP = {
    "handshape": "check your finger position and shape",
    "major_location": "check the broad area you're signing in (e.g. face vs. body vs. neutral space)",
    "minor_location": "you're in the right general area, but not quite the exact spot",
    "movement": "check the direction and shape of the motion",
    "repeated_movement": "check whether this sign repeats its movement or not",
}

_PRAISE = "Nice -- every parameter matched the reference."
_NO_VERDICT = "Not enough was resolved to give a diagnosis -- try signing more clearly, closer to the camera."


def coach_text(parameters) -> str:
    """One short line: praise if everything matched, otherwise the single
    parameter the grader was MOST confident is wrong (the strongest signal to
    act on -- an instructor gives one correction at a time, not five at once),
    with any other misses named briefly rather than dropped silently."""
    verdicts = list(parameters.values()) if hasattr(parameters, "values") else list(parameters)
    judged = [v for v in verdicts if v.correct is not None]
    if not judged:
        return _NO_VERDICT
    wrong = [v for v in judged if v.correct is False]
    if not wrong:
        return _PRAISE

    focus = max(wrong, key=lambda v: v.confidence)
    msg = (f"Focus on your {PARAM_NAME[focus.parameter]}: you signed "
           f"'{focus.predicted}', the target is '{focus.target}' -- {PARAM_TIP[focus.parameter]}.")
    others = [v.parameter for v in wrong if v is not focus]
    if others:
        msg += " Also off: " + ", ".join(PARAM_NAME[p] for p in others) + "."
    return msg


def focus_parameter(parameters) -> "str | None":
    """The single parameter coach_text would tell the learner to focus on, or
    None if nothing was wrong (or nothing was judged) -- this is exactly the
    signal Phase 6's scheduler needs to fire a contrastive minimal-pair drill,
    factored out so the scheduler doesn't have to re-derive it from scratch or
    re-parse coach_text's prose."""
    verdicts = list(parameters.values()) if hasattr(parameters, "values") else list(parameters)
    wrong = [v for v in verdicts if v.correct is False]
    if not wrong:
        return None
    return max(wrong, key=lambda v: v.confidence).parameter
