"""CTC-based continuous sign language recognition (CSLR) -- the industry-
standard approach to segmenting AND recognizing a continuous signing
sequence with no known target, built here ONLY as an isolated research
comparison against this project's own forced-alignment approach (see
`research/__init__.py`'s module docstring for why this is deliberately kept
out of the shipped product).

CTC (Connectionist Temporal Classification) trains a per-frame classifier
over (vocabulary + 1 blank) classes with a loss that marginalizes over every
possible frame-to-label alignment consistent with the target LABEL SEQUENCE
-- no frame boundaries needed at training time, exactly the property that
lets it solve open-vocabulary continuous recognition. This is structurally
the N-way classifier CLAUDE.md's non-negotiables rule out for the actual
product (a classifier must emit SOME label per frame, and will confidently
mislabel a malformed attempt), which is the whole reason this module is
isolated rather than integrated.

Two decoders are provided, matching the two distinct questions the
comparison in `scripts/eval_ctc_vs_alignment.py` asks:
  - `greedy_decode` -- CTC's actual normal use case: no known target, decode
    whatever the model thinks was signed. Reported as Word/Sign Error Rate.
  - `forced_align` -- given the model's per-frame log-probs AND the true
    label sequence (this project's actual situation, same as
    `grading.dtw_grader.dtw_align`), find the best-scoring frame-to-label
    alignment via the standard CTC Viterbi forced-alignment recurrence. This
    is the fair, apples-to-apples comparison point against `dtw_align`/
    `align_and_grade`: both get the true sequence, both produce frame
    boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

BLANK = 0  # CTC blank class index; real labels occupy 1..vocab_size


class CTCEncoder(nn.Module):
    """A single BiGRU over the full per-frame feature vector + a linear head
    to (vocab_size + 1) logits per frame. Deliberately NOT PoseGraderNet's
    multi-stream disjoint-head architecture -- that structure exists
    specifically to keep phonological parameters independently readable
    (CLAUDE.md's Phase 4 writeup), which is irrelevant here: CTC-CSLR
    predicts one flat gloss vocabulary, not five separate parameters. Sized
    to roughly match PoseGraderNet's combined stream capacity (hidden=128,
    vs. PoseGraderNet's own per-stream hidden), so this isn't a strawman
    comparison against a deliberately undersized model.
    """

    def __init__(self, in_dim: int, vocab_size: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=0.2 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden * 2, vocab_size + 1)  # +1 for blank

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """`features`: (B, T, F) padded batch. `lengths`: (B,) real lengths.
        Returns (B, T, vocab_size+1) log-probabilities (log_softmax already
        applied -- what both `nn.CTCLoss` and the decoders below expect)."""
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=features.shape[1])
        logits = self.head(out)
        return torch.log_softmax(logits, dim=-1)


def greedy_decode(log_probs: np.ndarray, blank: int = BLANK) -> list[int]:
    """CTC's actual normal use case -- no known target. Argmax per frame,
    then collapse consecutive repeats and drop blanks (the standard CTC
    greedy-decode collapse rule). `log_probs`: (T, vocab_size+1)."""
    frame_labels = log_probs.argmax(axis=-1)
    decoded = []
    prev = None
    for label in frame_labels:
        label = int(label)
        if label != blank and label != prev:
            decoded.append(label)
        prev = label
    return decoded


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance -- the standard basis for Word/Sign Error Rate."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev_diag + cost)
            prev_diag = tmp
    return dp[m]


def word_error_rate(predicted: list, true: list) -> float:
    """Edit distance normalized by the TRUE sequence length -- the standard
    CSLR metric (lower is better; can exceed 100% if predicted is much
    longer than true, same convention as speech-recognition WER)."""
    if not true:
        return 0.0 if not predicted else float("inf")
    return edit_distance(predicted, true) / len(true)


def _expand_target(target: list, blank: int = BLANK) -> list:
    """The standard CTC "extended" label sequence: blank, label, blank,
    label, ..., blank -- the state sequence a valid CTC alignment path must
    move through in order (each state either repeats itself or advances to
    the next one, one frame at a time)."""
    expanded = [blank]
    for label in target:
        expanded.append(label)
        expanded.append(blank)
    return expanded


def forced_align(log_probs: np.ndarray, target: list, blank: int = BLANK) -> list[int]:
    """The fair, apples-to-apples comparison point against `dtw_align`: given
    the model's per-frame log-probs AND the TRUE label sequence (this
    project's actual situation -- the target is always known), find the
    best-scoring frame-to-label alignment via the standard CTC Viterbi
    forced-alignment recurrence over the extended (blank-interleaved) target
    sequence.

    Returns a length-T list of indices into `target` (or -1 for a frame
    aligned to a blank) -- the per-frame gloss assignment, directly
    comparable to what `grading.alignment._segment_ranges` derives from
    `dtw_align`'s warp path.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T = log_probs.shape[0]
    expanded = _expand_target(target, blank)
    S = len(expanded)
    NEG_INF = -1e18

    # dp[t, s] = best log-prob of any valid path reaching extended-state s
    # after t frames. Standard CTC Viterbi transitions: stay at s, advance
    # from s-1, or (only for a non-blank s whose label differs from the
    # label two states back) skip the blank at s-1 and advance from s-2.
    dp = np.full((T, S), NEG_INF)
    back = np.zeros((T, S), dtype=np.int64)

    dp[0, 0] = log_probs[0, expanded[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, expanded[1]]

    for t in range(1, T):
        for s in range(S):
            best_prev, best_from = dp[t - 1, s], s
            # s-1/s-2 guarded explicitly -- numpy/Python negative indices would
            # otherwise silently wrap to the LAST column instead of meaning
            # "unreachable", corrupting the alignment near s==0/s==1.
            if s >= 1 and dp[t - 1, s - 1] > best_prev:
                best_prev, best_from = dp[t - 1, s - 1], s - 1
            if (s >= 2 and expanded[s] != blank and expanded[s] != expanded[s - 2]
                    and dp[t - 1, s - 2] > best_prev):
                best_prev, best_from = dp[t - 1, s - 2], s - 2
            dp[t, s] = best_prev + log_probs[t, expanded[s]] if best_prev > NEG_INF else NEG_INF
            back[t, s] = best_from

    end_state = S - 1 if (S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2]) else S - 2

    path_states = [0] * T
    s = end_state
    for t in range(T - 1, -1, -1):
        path_states[t] = s
        s = back[t, s]

    return [(-1 if expanded[s] == blank else (s - 1) // 2) for s in path_states]


@dataclass
class ForcedAlignedSegments:
    """Per-gloss [start, stop) frame ranges recovered from `forced_align`'s
    per-frame state path -- the CTC analogue of
    `grading.alignment.AlignedGrade.frame_range`."""

    ranges: "list[tuple[int, int]]"


def segments_from_forced_align(frame_labels: list, n_glosses: int) -> ForcedAlignedSegments:
    """Collapse `forced_align`'s per-frame gloss-index assignment (-1 for
    blank) into per-gloss [start, stop) ranges, same min/max-over-assigned-
    frames approach `grading.alignment._segment_ranges` uses for a DTW warp
    path -- blank frames are simply unassigned to any gloss, unlike DTW
    (which has no blank concept and assigns every frame somewhere); a gloss
    with zero non-blank frames falls back to an empty range rather than
    raising, since a poorly-trained CTC model failing to attend to a short
    word is a real, reportable failure mode here, not a bug in this
    function."""
    lo = [None] * n_glosses
    hi = [None] * n_glosses
    for t, g in enumerate(frame_labels):
        if g < 0:
            continue
        if lo[g] is None or t < lo[g]:
            lo[g] = t
        if hi[g] is None or t > hi[g]:
            hi[g] = t
    ranges = [(lo[g], hi[g] + 1) if lo[g] is not None else (0, 0) for g in range(n_glosses)]
    return ForcedAlignedSegments(ranges=ranges)
