"""DTW distance + nearest-reference grader sanity checks.

Runs under pytest OR as a plain script (`python tests/test_dtw_grader.py`).
Uses synthetic sequences and a hand-built grader, so no cache/models are needed.
"""
import numpy as np

from aslcv.grading.dtw_grader import DTWGrader, dtw_align, dtw_distance


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


# -- dtw_align: same recurrence as dtw_distance, but also returns the warp path -------

def test_align_distance_matches_dtw_distance():
    # dtw_align is a superset of dtw_distance, not a different algorithm --
    # the returned distance must always agree, band or no band.
    a, b = _seq(9, seed=2), _seq(14, seed=3)
    dist, _ = dtw_align(a, b)
    assert abs(dist - dtw_distance(a, b)) < 1e-6
    dist_banded, _ = dtw_align(a, b, band=5)
    assert abs(dist_banded - dtw_distance(a, b, band=5)) < 1e-6


def test_align_path_covers_both_endpoints_monotonically():
    a, b = _seq(9, seed=2), _seq(14, seed=3)
    _, path = dtw_align(a, b)
    assert path[0] == (0, 0)
    assert path[-1] == (8, 13)
    # a valid warp path never goes backwards in either sequence
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        assert i1 >= i0 and j1 >= j0
        assert (i1, j1) != (i0, j0)  # strictly advances at least one index


def test_align_path_is_identity_on_identical_sequences():
    a = _seq(10, seed=1)
    dist, path = dtw_align(a, a)
    assert dist < 1e-6
    assert path == [(i, i) for i in range(10)]


def test_align_empty_sequence_returns_no_path():
    a = _seq(5, seed=0)
    empty = np.zeros((0, 8), dtype=np.float32)
    dist, path = dtw_align(a, empty)
    assert dist == float("inf")
    assert path == []


def test_align_projects_known_segment_boundary():
    # forced-alignment's actual use case: two DISTINCT reference segments
    # concatenated (known boundary at frame 10), aligned against a noisy copy
    # of the same concatenation -- the warp path should place the boundary
    # frame (index 9 -> 10) at very close to the same reference index, so
    # projecting "which segment" onto the attempt via the path recovers the
    # true cut point.
    rng = np.random.default_rng(42)
    seg_a = rng.standard_normal((10, 8)).astype(np.float32) + 5.0   # segment "A" cluster
    seg_b = rng.standard_normal((10, 8)).astype(np.float32) - 5.0   # segment "B" cluster, far apart
    reference = np.concatenate([seg_a, seg_b], axis=0)
    attempt = reference + 0.01 * rng.standard_normal(reference.shape).astype(np.float32)

    _, path = dtw_align(attempt, reference)
    # for each reference index, the LAST attempt index aligned to it
    ref_to_attempt = {}
    for i, j in path:
        ref_to_attempt[j] = i
    boundary_attempt_frame = ref_to_attempt[9]  # last reference frame of segment A
    assert 7 <= boundary_attempt_frame <= 11  # close to the true boundary at attempt frame 9


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
