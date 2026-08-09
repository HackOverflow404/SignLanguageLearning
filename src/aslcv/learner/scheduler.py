"""What to drill next: gap-targeting + a light recency bias, plus automatic
contrastive minimal-pair drills. A simple weighted-random heuristic, not a real
spaced-repetition interval scheduler (SM2/Leitner) -- deliberately, for v1: see
project_workflow.md's Phase 6 section for the scoping decision.
"""
from __future__ import annotations

import random

from ..grading.phonology_labels import ALL_PARAMETERS, PhonologyLabels

RECENCY_CAP = 20     # attempts since last seen at which the recency boost saturates
WEAKNESS_FLOOR = 0.05  # a fully-mastered sign still keeps a small chance of being picked


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


def pick_next(mastery, signs, last_sign=None, last_wrong_parameter=None,
              minimal_pairs=None, rng=None):
    """Next sign to drill from `signs` (the pre-validated, has-a-reference-clip
    pool -- see diagnose_demo.py's resolve_targets).

    Fires a contrastive drill FIRST and unconditionally when the last attempt's
    focus mistake (`last_wrong_parameter`) has a minimal-pair partner for
    `last_sign` in the pool ("fire automatically when two signs differing in
    one parameter get confused", per project_workflow.md's Phase 6 design) --
    otherwise falls back to weighted-random gap-targeting: weight is
    (weakness) x (recency boost), so a weak, long-unseen sign is likeliest,
    a strong, just-drilled one is rarest but never impossible (WEAKNESS_FLOOR
    keeps even a mastered sign reachable, since real-world skill can regress
    and a session shouldn't ever fully starve a sign)."""
    candidates = list(signs)
    if not candidates:
        raise ValueError("pick_next() needs at least one candidate sign")

    partner = _contrastive_partner(minimal_pairs, last_sign, last_wrong_parameter, candidates)
    if partner is not None:
        return partner

    rng = rng or random
    weights = []
    for sign in candidates:
        weakness = max(WEAKNESS_FLOOR, 1.05 - mastery.sign_mastery(sign))
        gap = mastery.clock - mastery.last_seen.get(sign, -RECENCY_CAP)
        recency = 1.0 + min(max(gap, 0), RECENCY_CAP) / RECENCY_CAP  # in [1, 2]
        weights.append(weakness * recency)
    return rng.choices(candidates, weights=weights, k=1)[0]
