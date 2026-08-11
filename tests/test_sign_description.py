"""sign_description: describe_sign builds a grounded, always-available
(no LLM, no network) description of a target sign's phonology from the real
PhonologyLabels table. Runs under pytest OR as a plain script
(`python tests/test_sign_description.py`).
"""
import sys

from aslcv.generator.sign_description import describe_sign
from aslcv.grading.phonology_labels import PhonologyLabels

_phon_labels = PhonologyLabels()


def test_describes_all_three_parts_for_a_real_sign():
    text = describe_sign(_phon_labels, "water")
    assert text.startswith("Handshape:")
    assert "Location:" in text
    assert "Movement:" in text


def test_uses_readable_not_raw_codes():
    text = describe_sign(_phon_labels, "mother")
    # mother's minor_location is a PascalCase ASL-LEX code -- must be spaced/readable
    assert "_" not in text.split("Movement:")[0]  # no raw snake_case leaking through


def test_repeated_state_is_plain_english():
    water_text = describe_sign(_phon_labels, "water")  # water repeats
    assert "repeated" in water_text and "not repeated" not in water_text


def test_neutral_minor_location_is_dropped_major_location_is_not():
    # 'you' has major_location=Neutral (a real, informative value -- shown)
    # AND minor_location=Neutral (redundant once major already says so --
    # dropped, so "Location: Neutral" doesn't become "Location: Neutral, Neutral")
    text = describe_sign(_phon_labels, "you")
    location_part = text.split("Location:")[1].split("Movement:")[0]
    assert location_part.count("Neutral") == 1


def test_handshape_description_appended_when_grounded():
    # 'water' uses handshape 'w' -- check whichever real code it has renders
    # a parenthetical detail if and only if handshape_descriptions has one
    from aslcv.generator.handshape_descriptions import describe_handshape
    code = _phon_labels.label_for("water", "handshape")
    text = describe_sign(_phon_labels, "water")
    if describe_handshape(code):
        assert "(" in text.split("Location:")[0]
    else:
        assert "(" not in text.split("Location:")[0]


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
