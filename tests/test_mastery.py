"""MasteryState: EMA updates, None-skipping, save/load round trip.

Runs under pytest OR as a plain script (`python tests/test_mastery.py`).
"""
import tempfile
from pathlib import Path

from aslcv.learner.mastery import INITIAL_MASTERY, LEARNING_RATE, MasteryState


def test_never_attempted_sign_reads_as_neutral_prior():
    m = MasteryState()
    assert m.sign_mastery("mother") == INITIAL_MASTERY
    assert m.parameter_mastery("mother", "handshape") == INITIAL_MASTERY
    assert m.attempts_for("mother") == 0


def test_correct_verdict_moves_mastery_up():
    m = MasteryState()
    m.update("mother", {"handshape": True})
    expected = INITIAL_MASTERY + LEARNING_RATE * (1.0 - INITIAL_MASTERY)
    assert m.parameter_mastery("mother", "handshape") == expected
    assert m.attempts_for("mother") == 1


def test_incorrect_verdict_moves_mastery_down():
    m = MasteryState()
    m.update("mother", {"handshape": False})
    expected = INITIAL_MASTERY + LEARNING_RATE * (0.0 - INITIAL_MASTERY)
    assert m.parameter_mastery("mother", "handshape") == expected


def test_none_verdict_is_skipped_not_averaged_as_medium():
    m = MasteryState()
    m.update("mother", {"handshape": True, "movement": None})
    assert m.parameter_mastery("mother", "handshape") != INITIAL_MASTERY
    # movement was never resolved -- still the neutral prior, and no attempt recorded
    assert m.parameter_mastery("mother", "movement") == INITIAL_MASTERY
    assert "movement" not in m.attempts.get("mother", {})


def test_repeated_correct_verdicts_converge_toward_one():
    m = MasteryState()
    for _ in range(30):
        m.update("mother", {"handshape": True})
    assert m.parameter_mastery("mother", "handshape") > 0.99


def test_sign_mastery_averages_only_attempted_parameters():
    m = MasteryState()
    m.update("mother", {"handshape": True})  # -> 0.65
    # a sign with only ONE attempted parameter has sign_mastery == that parameter's value
    assert m.sign_mastery("mother") == m.parameter_mastery("mother", "handshape")


def test_clock_and_last_seen_advance_per_update_call():
    m = MasteryState()
    m.update("mother", {"handshape": True})
    m.update("father", {"handshape": True})
    assert m.clock == 2
    assert m.last_seen["mother"] == 1
    assert m.last_seen["father"] == 2


def test_save_load_round_trip():
    m = MasteryState()
    m.update("mother", {"handshape": True, "movement": False})
    m.update("father", {"minor_location": True})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "learner_state.json"
        m.save(path)
        loaded = MasteryState.load(path)
    assert loaded.mastery == m.mastery
    assert loaded.attempts == m.attempts
    assert loaded.last_seen == m.last_seen
    assert loaded.clock == m.clock


def test_load_missing_file_returns_fresh_state():
    m = MasteryState.load("/tmp/definitely_does_not_exist_learner_state.json")
    assert m.clock == 0
    assert m.mastery == {}


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
