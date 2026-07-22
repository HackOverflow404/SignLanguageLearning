"""DTW distance + nearest-reference grader sanity checks.

Runs under pytest OR as a plain script (`python tests/test_dtw_grader.py`).
Uses synthetic sequences and a hand-built grader, so no cache/models are needed.
"""
import numpy as np

from aslcv.grading.dtw_grader import DTWGrader, dtw_distance


def _seq(n, f=8, seed=0):
    return np.random.default_rng(seed).standard_normal((n, f)).astype(np.float32)


def test_dtw_zero_on_identical():
    a = _seq(12)
    assert dtw_distance(a, a) < 1e-6


def test_dtw_invariant_to_time_stretch():
    # duplicating every frame is a pure time warp -> DTW should align it at cost 0
    a = _seq(10, seed=1)
    stretched = np.repeat(a, 2, axis=0)
    assert dtw_distance(a, stretched) < 1e-5


def test_dtw_symmetric():
    a, b = _seq(9, seed=2), _seq(14, seed=3)
    assert abs(dtw_distance(a, b) - dtw_distance(b, a)) < 1e-5


def test_dtw_orders_by_similarity():
    a = _seq(10, seed=4)
    near = a + 0.01 * _seq(10, seed=5)   # small perturbation
    far = _seq(10, seed=6)               # unrelated
    assert dtw_distance(a, near) < dtw_distance(a, far)


def test_band_matches_full_when_wide():
    a, b = _seq(10, seed=7), _seq(12, seed=8)
    assert abs(dtw_distance(a, b, band=None) - dtw_distance(a, b, band=50)) < 1e-6


def _grader(agg="min"):
    refs = {
        "x": [_seq(10, seed=10), _seq(11, seed=11)],
        "y": [_seq(9, seed=20), _seq(13, seed=21)],
        "z": [_seq(12, seed=30)],
    }
    return DTWGrader(refs, standardizer=None, pipeline=None, agg=agg)


def test_grade_self_retrieval_and_ranking():
    g = _grader()
    attempt = g.references["y"][0]  # an actual reference of "y"
    ranked = g.grade(attempt)
    assert [s for s, _ in ranked][0] == "y"          # nearest sign is its own
    assert ranked[0][1] < 1e-6                         # min-agg distance to itself is ~0
    dists = [d for _, d in ranked]
    assert dists == sorted(dists)                      # returned sorted ascending


def test_grade_against_matches_ranked_entry():
    g = _grader()
    attempt = _seq(10, seed=99)
    ranked = dict(g.grade(attempt))
    for sign in g.signs:
        assert abs(g.grade_against(attempt, sign) - ranked[sign]) < 1e-9


def test_mean_vs_min_aggregate():
    attempt = _seq(10, seed=100)
    gmin = _grader("min").grade_against(attempt, "x")
    gmean = _grader("mean").grade_against(attempt, "x")
    assert gmin <= gmean  # min over a sign's clips is never above the mean


if __name__ == "__main__":
    passed = failed = 0
    for _name, fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {_name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
