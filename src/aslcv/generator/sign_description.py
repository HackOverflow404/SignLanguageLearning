"""A grounded, plain-English description of a target sign's phonology --
shown persistently alongside its reference video, not just when an attempt
gets something wrong. Built entirely from `data/phonology.csv` via
`PhonologyLabels` (the exact same grounded ASL-LEX/curriculum source every
`ParameterVerdict.target` already comes from) -- no LLM, no generation, just
formatting (`feedback.readable_value`) plus the grounded handshape
description (`handshape_descriptions.describe_handshape`) already used
elsewhere. Nothing here is invented; every clause traces to a specific
phonology.csv column for this specific sign.
"""
from __future__ import annotations

from .feedback import readable_value
from .handshape_descriptions import describe_handshape


def describe_sign(phon_labels, id_gloss: str) -> str:
    """One-paragraph description of `id_gloss`'s grounded phonology:
    handshape (with a grounded physical description when one exists -- see
    handshape_descriptions.py's module docstring for which codes qualify),
    location, movement, and whether it repeats."""
    handshape_code = phon_labels.label_for(id_gloss, "handshape")
    major = phon_labels.label_for(id_gloss, "major_location")
    minor = phon_labels.label_for(id_gloss, "minor_location")
    movement = phon_labels.label_for(id_gloss, "movement")
    repeats = phon_labels.repeated_bool(id_gloss)

    handshape_part = f"Handshape: {readable_value('handshape', handshape_code)}"
    detail = describe_handshape(handshape_code)
    if detail:
        handshape_part += f" ({detail})"

    location_part = f"Location: {readable_value('major_location', major)}"
    if minor and minor != "Neutral":
        location_part += f", {readable_value('minor_location', minor)}"

    movement_part = (f"Movement: {readable_value('movement', movement)}, "
                      f"{'repeated' if repeats else 'not repeated'}")

    return "  |  ".join([handshape_part, location_part, movement_part])
