"""What to drill next: gap-targeting via MasteryState's SM2-style due-gate, plus
automatic contrastive minimal-pair drills -- both reactive (last attempt's wrong
parameter) and proactive (a well-mastered sign gets proactively stress-tested
against its minimal-pair partner, this scheduler's only notion of "difficulty").
"""
from __future__ import annotations

import random

from ..grading.phonology_labels import ALL_PARAMETERS, PhonologyLabels

WEAKNESS_FLOOR = 0.05  # a fully-mastered sign still keeps a small chance of being picked
HIGH_MASTERY_THRESHOLD = 0.8    # "well-mastered enough to stress-test with a minimal pair"
PROACTIVE_CONTRASTIVE_PROB = 0.5  # chance a well-mastered pick gets redirected to its partner


def find_minimal_pairs(phon_labels: PhonologyLabels, signs) -> dict[str, list[tuple[str, str]]]:
    """parameter -> [(signA, signB), ...] for every pair within `signs` that
    differs in EXACTLY that one parameter and matches on all four others --
    the same definition eval_minimal_pairs.py uses for the accuracy screen,
    re-derived here rather than imported (that's a script, this is library
    code the live session imports; scripts import library code, not the
    other way around)."""
    signs = list(signs)
    pairs: dict[str, list[tuple[str, str]]] = {p: [] for p in ALL_PARAMETERS}
    for i, a in enumerate(signs):
        for b in signs[i + 1:]:
            diffs = [p for p in ALL_PARAMETERS if phon_labels.label_for(a, p) != phon_labels.label_for(b, p)]
            if len(diffs) == 1:
                pairs[diffs[0]].append((a, b))
    return pairs


def _contrastive_partner(minimal_pairs, sign, parameter, candidates):
    if not minimal_pairs or parameter not in minimal_pairs:
        return None
    for a, b in minimal_pairs[parameter]:
        if a == sign and b in candidates:
            return b
        if b == sign and a in candidates:
            return a
    return None


def _any_contrastive_partner(minimal_pairs, sign, candidates):
    """Like `_contrastive_partner` but not limited to one parameter -- used for
    the PROACTIVE stress-test path, which doesn't have a "wrong parameter" to
    key off since nothing was graded wrong; any minimal-pair partner works."""
    if not minimal_pairs:
        return None
    for pairs in minimal_pairs.values():
        for a, b in pairs:
            if a == sign and b in candidates:
                return b
            if b == sign and a in candidates:
                return a
    return None


def pick_next(mastery, signs, last_sign=None, last_wrong_parameter=None,
              minimal_pairs=None, rng=None):
    """Next sign to drill from `signs` (the pre-validated, has-a-reference-clip
    pool -- see diagnose_demo.py's resolve_targets).

    Fires a contrastive drill FIRST and unconditionally when the last attempt's
    focus mistake (`last_wrong_parameter`) has a minimal-pair partner for
    `last_sign` in the pool ("fire automatically when two signs differing in
    one parameter get confused", per project_workflow.md's Phase 6 design).

    Otherwise: restricts the candidate pool to signs MasteryState considers
    "due" (its SM2-style interval since last seen has elapsed) -- a sign just
    drilled successfully is excluded until enough other attempts have passed,
    same as SM2 skipping a well-known card. If nothing in the pool is due yet
    (e.g. everything was just drilled), falls through to the full pool rather
    than stalling -- a session should never run out of things to present.
    Within that pool, weighted-random by weakness alone (WEAKNESS_FLOOR keeps
    even a mastered sign reachable, since skill can regress).

    Difficulty control: a pick that lands on an already well-mastered sign
    (>= HIGH_MASTERY_THRESHOLD) has a chance of being redirected to its
    minimal-pair partner instead -- proactively stress-testing a parameter the
    learner hasn't been recently tripped up on, rather than only reacting
    after a wrong verdict."""
    candidates = list(signs)
    if not candidates:
        raise ValueError("pick_next() needs at least one candidate sign")

    partner = _contrastive_partner(minimal_pairs, last_sign, last_wrong_parameter, candidates)
    if partner is not None:
        return partner

    rng = rng or random

    due = [sign for sign in candidates if mastery.is_due(sign)]
    pool = due if due else candidates

    weights = [max(WEAKNESS_FLOOR, 1.05 - mastery.sign_mastery(sign)) for sign in pool]
    chosen = rng.choices(pool, weights=weights, k=1)[0]

    if mastery.sign_mastery(chosen) >= HIGH_MASTERY_THRESHOLD and rng.random() < PROACTIVE_CONTRASTIVE_PROB:
        stress_partner = _any_contrastive_partner(minimal_pairs, chosen, candidates)
        if stress_partner is not None:
            return stress_partner

    return chosen
