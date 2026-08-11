"""A grounded, plain-English description of a target sign's phonology --
shown persistently alongside its reference video, not just when an attempt
gets something wrong. Built entirely from `data/phonology.csv` via
`PhonologyLabels` (the exact same grounded ASL-LEX/curriculum source every
`ParameterVerdict.target` already comes from) -- no LLM, no generation, just
formatting (`feedback.readable_value`) plus the grounded handshape
description (`handshape_descriptions.describe_handshape`) already used
elsewhere, composed into flowing sentences rather than a "Label: value"
fragment list. Nothing here is invented; every clause traces to a specific
phonology.csv column for this specific sign -- the sentence structure is
just prose, not a new claim about the sign.
"""
from __future__ import annotations

from .feedback import readable_value
from .handshape_descriptions import describe_handshape

_MOVEMENT_WORD = {"Straight": "straight", "Curved": "curved", "Circular": "circular",
                  "Other": "irregular"}


def _location_sentence(major: str, minor: str) -> str:
    if minor and minor != "Neutral":
        if minor.endswith("Away"):
            base = readable_value("minor_location", minor[:-len("Away")]).lower()
            return f"near, but not touching, the {base}"
        return f"at the {readable_value('minor_location', minor).lower()}"
    if major == "Neutral":
        return "in the neutral space in front of your body"
    return f"near the {readable_value('major_location', major).lower()}"


def _movement_sentence(movement: str, repeats: bool) -> str:
    if movement == "None":
        sentence = "It has little to no path movement -- the hand stays mostly in place."
        if repeats:
            sentence += " There's a small repeated motion (like a wiggle or twist)."
        return sentence
    path_word = _MOVEMENT_WORD.get(movement, readable_value("movement", movement).lower())
    rep_word = "repeated" if repeats else "single"
    return f"It moves with a {rep_word} {path_word} movement."


def describe_sign(phon_labels, id_gloss: str) -> str:
    """A short, natural-language description of `id_gloss`'s grounded
    phonology: handshape (with a grounded physical description when one
    exists -- see handshape_descriptions.py's module docstring for which
    codes qualify), location, and movement -- two or three plain sentences,
    not a labeled fragment list."""
    handshape_code = phon_labels.label_for(id_gloss, "handshape")
    major = phon_labels.label_for(id_gloss, "major_location")
    minor = phon_labels.label_for(id_gloss, "minor_location")
    movement = phon_labels.label_for(id_gloss, "movement")
    repeats = phon_labels.repeated_bool(id_gloss)

    handshape_label = readable_value("handshape", handshape_code)
    article = "an" if handshape_label[:1].upper() in "AEIOU" else "a"
    detail = describe_handshape(handshape_code)
    handshape_clause = (f"{article} {handshape_label} handshape ({detail})" if detail
                         else f"{article} {handshape_label} handshape")

    first = f"This sign uses {handshape_clause}, {_location_sentence(major, minor)}."
    second = _movement_sentence(movement, repeats)
    return f"{first} {second}"
