"""sign_description: describe_sign builds a grounded, always-available
(no LLM, no network) NATURAL-LANGUAGE description of a target sign's
phonology from the real PhonologyLabels table -- two or three plain
sentences, not a "Label: value" fragment list. Runs under pytest OR as a
plain script (`python tests/test_sign_description.py`).
"""
import sys

from aslcv.generator.handshape_descriptions import describe_handshape
from aslcv.generator.sign_description import describe_sign
from aslcv.grading.phonology_labels import PhonologyLabels

_phon_labels = PhonologyLabels()


def test_reads_as_prose_not_labeled_fragments():
    text = describe_sign(_phon_labels, "water")
    assert text.startswith("This sign uses")
    assert "Handshape:" not in text and "Location:" not in text and "Movement:" not in text
    assert text.endswith(".")


def test_uses_readable_not_raw_codes():
    text = describe_sign(_phon_labels, "mother")
    assert "_" not in text  # no raw snake_case ASL-LEX code leaking through


def test_repeated_movement_is_stated_in_the_movement_sentence():
    water_text = describe_sign(_phon_labels, "water")  # water repeats
    assert "repeated" in water_text


def test_single_non_repeated_movement_says_single_not_repeated():
    # 'you' does not repeat
    text = describe_sign(_phon_labels, "you")
    assert "single" in text
    assert "a repeated" not in text


def test_neutral_location_reads_naturally_not_as_a_raw_value():
    # 'you' has both major_location and minor_location = Neutral
    text = describe_sign(_phon_labels, "you")
    assert "neutral space" in text.lower()
    assert "Neutral, Neutral" not in text  # never a raw double-Neutral fragment


def test_vowel_starting_handshape_gets_an_not_a():
    # 'thank_you'/'good' use handshape 'open_b' -> readable "Open B", which
    # needs "an", not "a" ("a Open B handshape" is bad grammar)
    text = describe_sign(_phon_labels, "good")
    assert "an Open B handshape" in text
    assert "a Open B handshape" not in text


def test_handshape_description_appended_when_grounded():
    code = _phon_labels.label_for("water", "handshape")
    text = describe_sign(_phon_labels, "water")
    first_sentence = text.split(".")[0]
    if describe_handshape(code):
        assert "(" in first_sentence
    else:
        assert "(" not in first_sentence


def test_away_minor_location_is_phrased_as_near_not_touching():
    # find a real curriculum sign whose minor_location ends in "Away"
    away_sign = next(
        (row["id_gloss"] for row in _phon_labels.rows if row["minor_location"].endswith("Away")),
        None)
    assert away_sign is not None, "expected at least one curriculum sign with an Away minor location"
    text = describe_sign(_phon_labels, away_sign)
    assert "not touching" in text
    assert "Away" not in text  # raw code suffix must not leak into the sentence


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
