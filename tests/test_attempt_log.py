"""AttemptRecorder save/load round trip and replay_capture's faithfulness to
the real CaptureBuffer. Poses carry their synthetic per-frame energy in
keypoints[0, 0], so `energy_fn` is a pure function of the poses it's handed
-- the same contract the real hand_motion_energy has.

Runs under pytest OR as a plain script (`python tests/test_attempt_log.py`).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from aslcv.attempt_log import (AttemptRecorder, load_attempt, longest_internal_pause,
                               replay_capture)
from aslcv.capture import CaptureBuffer
from aslcv.extractor.base import Pose


def _poses(energies, k=4):
    out = []
    for e in energies:
        kp = np.zeros((k, 2), np.float32)
        kp[0, 0] = e
        out.append(Pose(kp, np.ones(k, np.float32)))
    return out


def _energy_fn(poses):
    return [float(p.keypoints[0, 0]) for p in poses]


# a 3-sign sentence: rest, three bursts with short pauses, rest, then the
# learner keeps going (a second, later burst) -- as after a premature settle
SENTENCE = [0.0] * 20 + ([1.0] * 15 + [0.0] * 6) * 3 + [0.0] * 40 + [1.0] * 10 + [0.0] * 20


def _live_run(energies, settle_frames, preroll=5, max_frames=400):
    """What run_live_sentence does per frame: append to the buffer, record
    the pose with the buffer's resulting state."""
    poses = _poses(energies)
    buf = CaptureBuffer(_energy_fn, 0.5, settle_frames=settle_frames, preroll=preroll, max_frames=max_frames)
    rec = AttemptRecorder()
    n_captured = 0
    for p in poses:
        buf.append(p)
        rec.record(p, buf.state)
        if buf.state == "settled" and not n_captured:
            n_captured = len(buf.frames)
    return poses, buf, rec, n_captured


def test_replay_reproduces_live_settle_point():
    poses, buf, rec, n_captured = _live_run(SENTENCE, settle_frames=10)
    states = list(rec._states)
    live_idx = states.index(2)
    r = replay_capture(poses, _energy_fn, 0.5, settle_frames=10, preroll=5, max_frames=400)
    assert r.settle_index == live_idx
    assert r.span == rec.captured_span(n_captured)
    assert r.span[1] - r.span[0] == len(buf.frames)


def test_larger_settle_window_rides_through_inter_sign_pauses():
    short = replay_capture(_poses(SENTENCE), _energy_fn, 0.5, settle_frames=4, preroll=5, max_frames=400)
    long_ = replay_capture(_poses(SENTENCE), _energy_fn, 0.5, settle_frames=10, preroll=5, max_frames=400)
    first_pause_end = 20 + 15 + 6
    assert short.settle_index < first_pause_end  # settles in the first 6-frame pause
    assert long_.settle_index > 20 + 21 * 3 - 6  # only after the third sign


def test_recorder_keeps_frames_after_settle():
    _, buf, rec, _ = _live_run(SENTENCE, settle_frames=10)
    assert buf.state == "settled"
    assert len(rec) == len(SENTENCE)  # recording continued past the freeze


def test_save_load_round_trip():
    poses, _, rec, n_captured = _live_run(SENTENCE, settle_frames=10)
    with tempfile.TemporaryDirectory() as d:
        path = rec.save(d, {"sentence": "I want water.", "captured_span": list(rec.captured_span(n_captured))})
        att = load_attempt(path)
    assert len(att.poses) == len(poses)
    assert np.allclose(_energy_fn(att.poses), SENTENCE)
    assert list(att.states) == list(rec._states)
    assert att.meta["sentence"] == "I want water."
    assert att.meta["n_frames"] == len(poses)
    assert att.meta["dropped_frames"] == 0


def test_recorder_bounds_memory_and_counts_dropped():
    rec = AttemptRecorder(max_frames=10)
    for p in _poses([0.0] * 25):
        rec.record(p, "idle", t=0.0)
    assert len(rec) == 10
    assert rec.dropped == 15
    assert not rec.ever_active


def test_longest_internal_pause_ignores_leading_and_trailing_rest():
    e = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], float)
    assert longest_internal_pause(e, 0.5, 0, len(e)) == 3
    assert longest_internal_pause(np.array([0, 1, 1, 0], float), 0.5, 0, 4) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    sys.exit(0)
