"""DTW nearest-reference grader for the Phase-2 sign slice (numpy only).

The grader holds a reference bank -- for every sign in the slice, the featurized
TRAIN clips kept as variable-length (T, F) sequences (never padded; DTW aligns them).
Grading an attempt runs Dynamic Time Warping against each reference and aggregates
per sign (nearest clip by default), so `grade` returns the signs ranked by distance.

`grade` is the open-set "which sign is this?" the Phase-2 live demo uses.
`grade_against` is the closed-set "how close to the KNOWN target?" the real system
(Phase 4, grading against a prompted target) uses.

Features are z-standardized with statistics fit on the TRAIN references only, so every
feature dimension contributes comparably to the Euclidean local cost and no val/test
information leaks in.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..features import FeaturePipeline, Standardizer


def _pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) Euclidean distance between every frame of `a` and every frame of `b`.

    Computed in float64 -- the a2 + b2 - 2ab identity cancels badly in float32 (an
    identical frame pair can leave ~1e-3 instead of 0, which then accumulates along a
    long warp path)."""
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a2 = np.einsum("ij,ij->i", a, a)[:, None]
    b2 = np.einsum("ij,ij->i", b, b)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)  # kill tiny negatives from float cancellation
    return np.sqrt(d2)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: "int | None" = None) -> float:
    """Length-normalized DTW distance between two (T, F) sequences.

    Local cost is per-frame Euclidean distance; the accumulated optimal-path cost is
    divided by (n + m) so sequences of different lengths compare fairly. `band`, if
    set, is a Sakoe-Chiba radius (in frames, scaled to the length ratio) that both
    speeds the DP and forbids pathological warps.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    cost = _pairwise_euclidean(a, b)  # (n, m)
    inf = float("inf")
    prev = np.full(m + 1, inf)
    prev[0] = 0.0
    ratio = m / n
    for i in range(1, n + 1):
        cur = np.full(m + 1, inf)
        row = cost[i - 1]
        if band is None:
            lo, hi = 1, m
        else:
            center = i * ratio
            lo = max(1, int(center - band))
            hi = min(m, int(center + band) + 1)
        left = inf
        for j in range(lo, hi + 1):
            best = prev[j]
            pj = prev[j - 1]
            if pj < best:
                best = pj
            if left < best:
                best = left
            left = row[j - 1] + best
            cur[j] = left
        prev = cur

    total = prev[m]
    return float(total) / (n + m) if np.isfinite(total) else inf


def dtw_align(a: np.ndarray, b: np.ndarray, band: "int | None" = None) -> "tuple[float, list[tuple[int, int]]]":
    """Like `dtw_distance`, but also returns the optimal warp path -- needed
    for FORCED ALIGNMENT (Phase 7: projecting a known reference's per-frame
    gloss labels onto a live attempt), which `dtw_distance` alone can't give
    since it only keeps the previous DP row and discards it as it goes.

    Shares the identical recurrence and cost function as `dtw_distance` (same
    `_pairwise_euclidean`, same length-normalization, same `band` semantics)
    so the returned distance always matches what `dtw_distance(a, b, band)`
    would return -- this is a superset, not a different algorithm. The only
    difference is keeping the FULL (n+1, m+1) DP table instead of a rolling
    row, so the path can be backtraced -- O(n*m) memory, fine at the
    single-attempt scale this is used at (never call this over a whole
    reference bank the way `dtw_distance` is inside `DTWGrader`; use
    `dtw_distance` there).

    Returns `(length_normalized_distance, path)` where `path` is a list of
    `(i, j)` 0-indexed frame pairs into `a` and `b`, in increasing order,
    covering the start and end of both sequences.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf"), []

    cost = _pairwise_euclidean(a, b)  # (n, m)
    inf = float("inf")
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    ratio = m / n
    for i in range(1, n + 1):
        if band is None:
            lo, hi = 1, m
        else:
            center = i * ratio
            lo = max(1, int(center - band))
            hi = min(m, int(center + band) + 1)
        row = cost[i - 1]
        dp_prev, dp_cur = dp[i - 1], dp[i]
        for j in range(lo, hi + 1):
            best = dp_prev[j]
            if dp_prev[j - 1] < best:
                best = dp_prev[j - 1]
            if dp_cur[j - 1] < best:
                best = dp_cur[j - 1]
            dp_cur[j] = row[j - 1] + best

    total = dp[n, m]
    if not np.isfinite(total):
        return inf, []

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diag, up, left = dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]
        best = min(diag, up, left)
        if diag <= best:
            i, j = i - 1, j - 1
        elif up <= best:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(total) / (n + m), path


class DTWGrader:
    """Nearest-reference DTW grader over a fixed set of signs."""

    def __init__(self, references, standardizer, pipeline, *, agg="min", band=None, missing=0):
        if agg not in ("min", "mean"):
            raise ValueError(f"agg must be 'min' or 'mean', got {agg!r}")
        self.references = references          # {sign: [ (T, F) standardized, ... ]}
        self.standardizer = standardizer     # Standardizer | None
        self.pipeline = pipeline             # FeaturePipeline (raw, no standardizer)
        self.agg = agg
        self.band = band
        self.missing = missing               # train clips skipped for a missing cache
        self.signs = list(references.keys())

    @classmethod
    def build(cls, pipeline: FeaturePipeline, cache_dir, signs, train_rows,
              *, agg="min", band=None, standardize=True):
        """Featurize the TRAIN clips of `signs` into the reference bank.

        `cache_dir` is data/cache/{extractor}; `train_rows` are manifest dict rows
        (needs `id_gloss`, `video_id`). Fits the standardizer on the TRAIN features.
        """
        cache_dir = Path(cache_dir)
        wanted = set(signs)
        refs: dict[str, list[np.ndarray]] = defaultdict(list)
        raw_all: list[np.ndarray] = []
        missing = 0
        for r in train_rows:
            if r["id_gloss"] not in wanted:
                continue
            npz = cache_dir / f"{r['video_id']}.npz"
            if not npz.exists():
                missing += 1
                continue
            feats = pipeline.assemble_npz(npz).features  # raw (T, F)
            refs[r["id_gloss"]].append(feats)
            raw_all.append(feats)
        if not raw_all:
            raise RuntimeError(f"no reference clips found under {cache_dir} for the slice")

        standardizer = Standardizer.fit(raw_all) if standardize else None
        if standardizer is not None:
            refs = {s: [standardizer.transform(f) for f in feats] for s, feats in refs.items()}
        else:
            refs = dict(refs)
        return cls(refs, standardizer, pipeline, agg=agg, band=band, missing=missing)

    # -- featurize (raw; grade standardizes) ---------------------------------

    def featurize_npz(self, npz_path) -> np.ndarray:
        return self.pipeline.assemble_npz(npz_path).features

    def _prep(self, attempt_features) -> np.ndarray:
        feats = np.asarray(attempt_features, dtype=np.float32)
        return self.standardizer.transform(feats) if self.standardizer is not None else feats

    def _aggregate(self, dists) -> float:
        if not dists:
            return float("inf")
        return float(np.min(dists)) if self.agg == "min" else float(np.mean(dists))

    # -- grading -------------------------------------------------------------

    def grade(self, attempt_features):
        """Rank every sign by aggregated DTW distance -> [(sign, distance), ...]."""
        att = self._prep(attempt_features)
        ranked = [
            (sign, self._aggregate([dtw_distance(att, ref, self.band) for ref in seqs]))
            for sign, seqs in self.references.items()
        ]
        ranked.sort(key=lambda sd: sd[1])
        return ranked

    def grade_against(self, attempt_features, target_sign) -> float:
        """Closed-set distance to a KNOWN target sign (Phase-4 grading)."""
        if target_sign not in self.references:
            raise KeyError(f"unknown target sign {target_sign!r}; have {self.signs}")
        att = self._prep(attempt_features)
        return self._aggregate([dtw_distance(att, ref, self.band) for ref in self.references[target_sign]])
