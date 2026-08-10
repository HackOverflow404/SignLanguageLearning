"""coach_text / focus_parameter: templated feedback from duck-typed verdicts.

No real ParameterVerdict/checkpoint needed -- a plain object with
parameter/correct/confidence attributes is all this module reads, verified by
using a bare stand-in here instead of the real (torch-backed) dataclass.

Runs under pytest OR as a plain script (`python tests/test_feedback.py`).
"""
from types import SimpleNamespace

from aslcv.generator.feedback import coach_text, focus_parameter


def verdict(parameter, correct, confidence=0.8, predicted="A", target="B"):
    return SimpleNamespace(parameter=parameter, correct=correct, confidence=confidence,
                            predicted=predicted, target=target)


def test_all_correct_is_praise():
    parameters = {
        "handshape": verdict("handshape", True),
        "movement": verdict("movement", True),
    }
    assert "Nice" in coach_text(parameters)
    assert focus_parameter(parameters) is None


def test_no_judged_parameters_is_the_no_verdict_message():
    parameters = {"handshape": verdict("handshape", None)}
    assert "Not enough" in coach_text(parameters)
    assert focus_parameter(parameters) is None


def test_single_wrong_parameter_is_named():
    parameters = {
        "handshape": verdict("handshape", False, confidence=0.9),
        "movement": verdict("movement", True),
    }
    text = coach_text(parameters)
    assert "handshape" in text
    assert "Also off" not in text
    assert focus_parameter(parameters) == "handshape"


def test_focus_is_the_most_confident_wrong_parameter():
    parameters = {
        "handshape": verdict("handshape", False, confidence=0.55),
        "movement": verdict("movement", False, confidence=0.95),
    }
    assert focus_parameter(parameters) == "movement"
    assert "movement" in coach_text(parameters)


def test_other_wrong_parameters_are_named_but_not_focused():
    parameters = {
        "handshape": verdict("handshape", False, confidence=0.95),
        "movement": verdict("movement", False, confidence=0.6),
    }
    text = coach_text(parameters)
    assert text.startswith("Focus on your handshape")
    assert "Also off: movement" in text


def test_accepts_a_plain_iterable_not_just_dict_values():
    parameters = [verdict("handshape", False, confidence=0.9)]
    assert focus_parameter(parameters) == "handshape"


def test_focus_message_states_what_was_signed_and_the_target_value():
    # this is the actionable content the templated line must carry -- not
    # just "focus on your handshape" but WHAT was wrong and WHAT it should
    # have been, both pre-computed grounded facts from the grader, never
    # invented here.
    parameters = {"handshape": verdict("handshape", False, confidence=0.9, predicted="1", target="5")}
    text = coach_text(parameters)
    assert "'1'" in text
    assert "'5'" in text


if __name__ == "__main__":
    import sys
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
