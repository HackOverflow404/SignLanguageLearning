"""handshape_descriptions: grounded, per-code descriptions built from
data/phonology.csv's exact ASL-LEX sub-features -- only for codes whose
(majority-recorded) Flexion category has a locally sourced definition
(FullyOpen/FullyClosed/Stacked/Crossed); everything else returns None rather
than a guessed description (Bent/Curved/Flat -- see module docstring).

Runs under pytest OR as a plain script (`python tests/test_handshape_descriptions.py`).
"""
import sys

from aslcv.generator.handshape_descriptions import describe_handshape


def test_closed_b_is_fingers_together_thumb_tucked():
    text = describe_handshape("closed_b")
    assert text is not None
    assert "together" in text
    assert "tucked in" in text


def test_open_b_differs_from_closed_b_only_in_thumb():
    closed = describe_handshape("closed_b")
    open_ = describe_handshape("open_b")
    assert "tucked in" in closed
    assert "out to the side" in open_


def test_s_is_a_fist():
    text = describe_handshape("s")
    assert text is not None
    assert "fist" in text


def test_thumb_only_handshape_reads_naturally_not_doubled():
    text = describe_handshape("a")
    assert text is not None
    assert text.count("thumb") == 1


def test_ungrounded_flexion_returns_none():
    # 'c' is Curved-flexion in the curriculum -- not one of the 4 locally
    # sourced Flexion categories, so this must stay None, never a guess.
    assert describe_handshape("c") is None


def test_unknown_code_returns_none():
    assert describe_handshape("not_a_real_code") is None


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
