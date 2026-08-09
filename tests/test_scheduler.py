"""find_minimal_pairs + pick_next: real phonology data (same convention as
test_dataset.py; no data means nothing to schedule), deterministic weighted-random
picks via a seeded rng.

Runs under pytest OR as a plain script (`python tests/test_scheduler.py`).
"""
import random

from aslcv.grading.phonology_labels import PhonologyLabels
from aslcv.learner.mastery import MasteryState
from aslcv.learner.scheduler import find_minimal_pairs, pick_next

PHON = PhonologyLabels()


def test_mother_father_is_a_real_minor_location_minimal_pair():
    # the curriculum's built-in minimal pair (differ ONLY in minor_location) --
    # same pair diagnose_demo.py's --selftest exercises for head independence.
    pairs = find_minimal_pairs(PHON, ["mother", "father", "water"])
    assert ("father", "mother") in pairs["minor_location"] or ("mother", "father") in pairs["minor_location"]
    for param in ("handshape", "major_location", "movement", "repeated_movement"):
        assert ("mother", "father") not in pairs[param]
        assert ("father", "mother") not in pairs[param]


def test_minimal_pairs_restricted_to_the_given_pool():
    # a pair whose partner isn't in the pool must not appear at all
    pairs = find_minimal_pairs(PHON, ["mother", "water"])
    for param_pairs in pairs.values():
        for a, b in param_pairs:
            assert a in ("mother", "water") and b in ("mother", "water")


def test_contrastive_drill_fires_when_partner_available():
    m = MasteryState()
    pairs = find_minimal_pairs(PHON, ["mother", "father", "water"])
    picked = pick_next(m, ["mother", "father", "water"], last_sign="father",
                        last_wrong_parameter="minor_location", minimal_pairs=pairs)
    assert picked == "mother"


def test_contrastive_drill_does_not_fire_without_a_real_partner():
    m = MasteryState()
    pairs = find_minimal_pairs(PHON, ["mother", "father", "water"])
    # "water" has no minor_location minimal-pair partner in this pool -- must fall
    # through to weighted-random, not silently return None or crash
    rng = random.Random(0)
    picked = pick_next(m, ["mother", "father", "water"], last_sign="water",
                        last_wrong_parameter="minor_location", minimal_pairs=pairs, rng=rng)
    assert picked in ("mother", "father", "water")


def test_weighted_random_favors_the_weaker_sign():
    m = MasteryState()
    for _ in range(20):
        m.update("mother", {"handshape": True})   # -> mastered
        m.update("father", {"handshape": False})  # -> weak
    rng = random.Random(0)
    picks = [pick_next(m, ["mother", "father"], rng=rng) for _ in range(200)]
    assert picks.count("father") > picks.count("mother")


def test_never_seen_sign_is_reachable():
    m = MasteryState()
    m.update("mother", {"handshape": True})
    rng = random.Random(0)
    picks = {pick_next(m, ["mother", "father"], rng=rng) for _ in range(50)}
    assert "father" in picks  # never-seen sign must get picked at least once in 50 draws


def test_pick_next_empty_pool_raises():
    m = MasteryState()
    try:
        pick_next(m, [])
        assert False, "expected ValueError"
    except ValueError:
        pass


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
