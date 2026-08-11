"""Grounded, plain-English descriptions of ASL-LEX handshape codes -- built
from the exact per-sign sub-features ASL-LEX itself recorded (`data/phonology.csv`,
joined from `data/ASL_LEX/Data Files/signdata.csv`'s `SelectedFingers.2.0`/
`Flexion.2.0`/`Spread.2.0`/`ThumbPosition.2.0` columns), never guessed.

Deliberately PARTIAL, not exhaustive. Only 4 of Brentari's 9 Flexion categories
have an explicit verbal definition in this project's locally available ASL-LEX
documentation (`data/ASL_LEX/ASL-LEX Manuscript.pdf`, ~lines 631-648): `FullyOpen`
= "fully extended" (degree 1 of 9); `FullyClosed` = the opposite endpoint;
`Stacked` = "as in the fingerspelled letter K"; `Crossed` = "as in the
fingerspelled letter R". The three intermediate categories (`Bent`, `Curved`,
`Flat`) are real, standard ASL phonology terms, but neither the manuscript nor
the ASL-LEX 2.0 coding supplement spells out their precise definition in this
project's local copies -- rather than write a plausible-sounding definition
from general knowledge and risk it being wrong (exactly what CLAUDE.md's
Deaf-review gate exists to prevent), `describe_handshape` returns None for
codes whose majority-recorded Flexion is one of these three; callers fall
back to `feedback.readable_value`'s formatting-only treatment.

`SelectedFingers` letters (i/m/r/p/t) are grounded via the manuscript's Fig. 3
caption ("m = middle finger, r = ring finger, p = pinky, i = index finger");
thumb-only ('t') is the one case the manuscript explicitly names ("[t]he
thumb was never coded as a selected finger unless it was the only selected
finger in the sign"). `Spread` (0/1) and `ThumbPosition` (Open/Closed) are
used via their own already-English value names, not a translated definition
-- "together"/"spread apart" and "tucked in"/"out to the side" restate the
data's own terms, the same formatting-only principle as
`feedback.readable_value`, not a new claim about what the code means.

A description covers only the SELECTED fingers ASL-LEX records (the ones
that move / are foregrounded) -- it says nothing about the other fingers'
position, since that isn't part of what this data records. This makes some
descriptions necessarily partial (e.g. 'y', where only the pinky is
"selected" even though the thumb is also extended in the real handshape) --
honestly incomplete rather than filled in with an unsourced guess.

Two handshape codes used in the curriculum ('s', 'y') have a genuine minority
variant recorded for one sign each (real ASL-LEX data, not an error -- see
`_build_descriptions`); only the majority variant is described.

NOT YET DEAF-REVIEWED, same as every other live-diagnosis surface in this
app -- callers are responsible for showing it alongside the same disclaimer.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PHONOLOGY_CSV = REPO / "data" / "phonology.csv"

_SELECTED_FINGER_NAMES = {"i": "index", "m": "middle", "r": "ring", "p": "pinky"}

# Only these four of Brentari's nine Flexion categories have an explicit
# verbal definition in this project's local ASL-LEX documentation -- see
# module docstring. Any other Flexion value (Bent/Curved/Flat/NA on a
# multi-finger group) means describe_handshape() returns None for that code.
_GROUNDED_FLEXION = {
    "FullyOpen": "straight",
    "FullyClosed": "curled into a fist",
    "Stacked": "stacked at different heights (like the fingerspelled K)",
    "Crossed": "crossed over each other (like the fingerspelled R)",
}

_SPREAD_PHRASE = {"0": "together", "1": "spread apart"}
_THUMB_PHRASE = {"Open": "thumb out to the side", "Closed": "thumb tucked in"}
_THUMB_ALONE_PHRASE = {"Open": "held out to the side", "Closed": "tucked in"}


def _fingers_phrase(selected_fingers: str) -> str:
    names = [_SELECTED_FINGER_NAMES[c] for c in selected_fingers]
    if len(names) == 1:
        return f"the {names[0]} finger"
    return ", ".join(names[:-1]) + f" and {names[-1]} fingers"


def _describe_row(fingers: str, flexion: str, spread: str, thumb: str) -> "str | None":
    if fingers == "t":  # thumb is the only selected finger -- no Flexion category applies
        state = _THUMB_ALONE_PHRASE.get(thumb)
        return f"the thumb, {state}" if state else "the thumb"
    if flexion not in _GROUNDED_FLEXION:
        return None
    parts = [f"{_fingers_phrase(fingers)} {_GROUNDED_FLEXION[flexion]}"]
    spread_phrase = _SPREAD_PHRASE.get(spread)
    if spread_phrase:
        parts.append(spread_phrase)
    thumb_phrase = _THUMB_PHRASE.get(thumb)
    if thumb_phrase:
        parts.append(thumb_phrase)
    return ", ".join(parts)


def _build_descriptions() -> dict:
    """One description per curriculum handshape code, built from the
    MAJORITY sub-feature combination recorded across curriculum signs using
    that code. Two codes have a real minority variant for one sign each
    (both genuine ASL-LEX-recorded realizations, not a data error): 's' is
    thumb-Closed for 4 of 5 curriculum signs using it, thumb-Open for the
    fifth; 'y' has SelectedFingers=p (pinky) for 3 of 4 signs,
    SelectedFingers=imr for the fourth. Only the majority variant is
    described here."""
    if not PHONOLOGY_CSV.exists():
        return {}
    with open(PHONOLOGY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    by_code = {}
    for r in rows:
        code = r["handshape"]
        key = (r["selected_fingers"], r["flexion"], r["spread"], r["thumb_position"])
        by_code.setdefault(code, Counter())[key] += 1

    descriptions = {}
    for code, counts in by_code.items():
        fingers, flexion, spread, thumb = counts.most_common(1)[0][0]
        description = _describe_row(fingers, flexion, spread, thumb)
        if description is not None:
            descriptions[code] = description
    return descriptions


_DESCRIPTIONS = _build_descriptions()


def describe_handshape(code: str) -> "str | None":
    """A grounded plain-English description of handshape `code`, or None if
    this code's (majority-recorded) Flexion category isn't one this project
    has a locally sourced definition for (Bent/Curved/Flat -- see module
    docstring), or the code isn't in the curriculum at all."""
    return _DESCRIPTIONS.get(code)
