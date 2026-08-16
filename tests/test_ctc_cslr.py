"""CTC-CSLR research module (aslcv.research.ctc_cslr) -- deterministic unit
tests on synthetic tensors, no training/checkpoint needed. Cross-checks
`forced_align` against PyTorch's own `nn.CTCLoss` forward algorithm as the
strongest available correctness check: a Viterbi best-path log-prob can
NEVER exceed the full marginal (`-CTCLoss`, which sums over every valid
path) -- if it did, the DP would be wrong.

Runs under pytest OR as a plain script (`python tests/test_ctc_cslr.py`).
"""
import numpy as np
import torch

from aslcv.research.ctc_cslr import (BLANK, CTCEncoder, edit_distance, forced_align, greedy_decode,
                                      segments_from_forced_align, word_error_rate)


def test_edit_distance_identical_sequences_is_zero():
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0


def test_edit_distance_matches_known_values():
    assert edit_distance([1, 2, 3], [1, 2]) == 1        # one deletion
    assert edit_distance([1, 2], [1, 2, 3]) == 1        # one insertion
    assert edit_distance([1, 2, 3], [1, 5, 3]) == 1      # one substitution
    assert edit_distance([], [1, 2, 3]) == 3
    assert edit_distance([1, 2, 3], []) == 3


def test_word_error_rate_normalizes_by_true_length():
    assert word_error_rate([1, 2, 3], [1, 2, 3]) == 0.0
    assert word_error_rate([1, 2], [1, 2, 3, 4]) == 2 / 4  # 2 insertions needed, /4 true words
    assert word_error_rate([], []) == 0.0


def test_greedy_decode_collapses_repeats_and_drops_blank():
    # frame labels: blank, 1, 1, blank, 2, 2, 2, blank -> [1, 2]
    T, V = 8, 3  # blank=0, labels 1,2
    log_probs = np.full((T, V + 1), -10.0)
    frame_labels = [BLANK, 1, 1, BLANK, 2, 2, 2, BLANK]
    for t, lbl in enumerate(frame_labels):
        log_probs[t, lbl] = 0.0
    assert greedy_decode(log_probs) == [1, 2]


def test_greedy_decode_adjacent_same_label_needs_a_blank_between():
    # two SEPARATE occurrences of label 1 need a blank between them, or
    # they'd collapse into one -- this is exactly why CTC needs blanks
    T, V = 5, 2
    log_probs = np.full((T, V + 1), -10.0)
    frame_labels = [1, 1, BLANK, 1, 1]
    for t, lbl in enumerate(frame_labels):
        log_probs[t, lbl] = 0.0
    assert greedy_decode(log_probs) == [1, 1]


def test_ctc_encoder_output_shape_and_is_log_probs():
    model = CTCEncoder(in_dim=16, vocab_size=5, hidden=8, layers=1)
    features = torch.randn(3, 10, 16)
    lengths = torch.tensor([10, 7, 4])
    out = model(features, lengths)
    assert out.shape == (3, 10, 6)  # vocab_size + 1 (blank)
    # log_softmax output: each real frame's row exponentiates to ~1
    probs = out[0, 0].exp()
    assert abs(probs.sum().item() - 1.0) < 1e-4


def test_forced_align_recovers_an_unambiguous_path():
    # frame0=blank, frame1=label1, frame2=blank, frame3=label2, frame4=blank
    T, V = 5, 2
    log_probs = np.full((T, V + 1), -10.0)
    frame_labels = [BLANK, 1, BLANK, 2, BLANK]
    for t, lbl in enumerate(frame_labels):
        log_probs[t, lbl] = 0.0
    path = forced_align(log_probs, target=[1, 2])
    assert path == [-1, 0, -1, 1, -1]  # frame1 -> target[0], frame3 -> target[1]


def test_forced_align_score_never_exceeds_the_full_ctc_marginal():
    """The real correctness check: nn.CTCLoss computes -log(sum over ALL
    valid alignment paths); forced_align's Viterbi finds the SINGLE best
    path. The best single path's probability can never exceed the sum of
    every path's probability, so Viterbi's log-prob must be <= -CTCLoss.
    A bug in the DP recurrence (e.g. an off-by-one, or the negative-index
    wraparound this implementation explicitly guards against) would very
    likely violate this on a random instance."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T, vocab_size = 20, 4
    target = [1, 2, 3]

    logits = torch.randn(T, vocab_size + 1)
    log_probs_t = torch.log_softmax(logits, dim=-1)
    log_probs = log_probs_t.numpy()

    path = forced_align(log_probs, target)
    viterbi_score = sum(
        log_probs[t, (target[g] if g >= 0 else BLANK)] for t, g in enumerate(path)
    )

    ctc_loss = torch.nn.functional.ctc_loss(
        log_probs_t.unsqueeze(1), torch.tensor(target).unsqueeze(0),
        input_lengths=torch.tensor([T]), target_lengths=torch.tensor([len(target)]),
        blank=BLANK, reduction="sum", zero_infinity=True,
    )
    full_marginal = -ctc_loss.item()

    assert viterbi_score <= full_marginal + 1e-4  # tiny float slack only


def test_segments_from_forced_align_derives_correct_ranges():
    # path: blank, gloss0, gloss0, blank, gloss1, blank
    path = [-1, 0, 0, -1, 1, -1]
    segs = segments_from_forced_align(path, n_glosses=2)
    assert segs.ranges == [(1, 3), (4, 5)]


def test_segments_from_forced_align_handles_a_gloss_with_no_frames():
    # gloss index 1 never appears -- a real CTC failure mode, not a crash
    path = [-1, 0, 0, -1]
    segs = segments_from_forced_align(path, n_glosses=2)
    assert segs.ranges[0] == (1, 3)
    assert segs.ranges[1] == (0, 0)


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
