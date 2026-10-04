"""Live-attempt logging + offline CaptureBuffer replay -- the data side of
tuning sentence-mode capture against a REAL camera instead of synthetic
energy patterns (Phase 7 step 5's open gap).

`AttemptRecorder` keeps every raw pose the live loop feeds `CaptureBuffer`
since the last reset -- INCLUDING frames after the buffer settles, which
CaptureBuffer itself discards. Those post-settle frames are the evidence for
the failure that matters most here: a capture that settled mid-sentence
while the learner kept signing. Keypoints only, never video.

`replay_capture` re-runs a recorded stream through the real `CaptureBuffer`
class with any parameters, so a settle/max/threshold change is measured on
real recordings before it ships, not reasoned about. It feeds the buffer
frame INDICES and maps them back to poses inside `energy_fn`, so the state
machine sees exactly what it would have seen live (energy is computed over
the buffer's current contents, which idle-trimming changes) without any
reimplementation of its logic.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .capture import CaptureBuffer
from .extractor.base import Pose

STATE_CODES = {"idle": 0, "active": 1, "settled": 2}
STATE_NAMES = {v: k for k, v in STATE_CODES.items()}


class AttemptRecorder:
    """Rolling record of (pose, capture state, timestamp) since the last
    `reset()`. Bounded by `max_frames` (oldest dropped first) so an idle
    session left open can't grow without limit; the dropped prefix is idle
    rest, which replay only needs `preroll` frames of anyway."""

    def __init__(self, max_frames: int = 1800):
        self.max_frames = max_frames
        self.reset()

    def reset(self) -> None:
        self._poses: deque = deque(maxlen=self.max_frames)
        self._states: deque = deque(maxlen=self.max_frames)
        self._times: deque = deque(maxlen=self.max_frames)
        self.dropped = 0  # frames evicted from the front since reset

    def record(self, pose: Pose, capture_state: str, t: "float | None" = None) -> None:
        if len(self._poses) == self.max_frames:
            self.dropped += 1
        self._poses.append(pose)
        self._states.append(STATE_CODES[capture_state])
        self._times.append(time.time() if t is None else t)

    def __len__(self) -> int:
        return len(self._poses)

    @property
    def ever_active(self) -> bool:
        return any(s != STATE_CODES["idle"] for s in self._states)

    def captured_span(self, n_captured: int) -> "tuple[int, int] | None":
        """[start, stop) of the frames CaptureBuffer held when it settled, as
        indices into this recording. The settling frame is itself part of the
        capture (append adds it, then flips the state), so the span ends one
        past the first `settled` frame."""
        states = list(self._states)
        if STATE_CODES["settled"] not in states:
            return None
        stop = states.index(STATE_CODES["settled"]) + 1
        return max(0, stop - n_captured), stop

    def save(self, directory: "str | Path", meta: dict) -> Path:
        """Write `<stamp>.npz` (keypoints/scores/states/timestamps) plus a
        `<stamp>.json` sidecar with `meta`. Atomic, same temp-file +
        os.replace convention as extract_landmarks.write_npz."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        npz_path = directory / f"{stamp}.npz"

        poses = list(self._poses)
        keypoints = np.stack([p.keypoints for p in poses]).astype(np.float32)
        scores = np.stack([p.scores for p in poses]).astype(np.float32)
        tmp = Path(str(npz_path) + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, keypoints=keypoints, scores=scores,
                                states=np.asarray(self._states, np.int8),
                                timestamps=np.asarray(self._times, np.float64))
        os.replace(tmp, npz_path)

        times = np.asarray(self._times)
        meta = dict(meta, n_frames=len(poses), dropped_frames=self.dropped,
                    stream_fps=_fps(times))
        json_path = npz_path.with_suffix(".json")
        tmp = Path(str(json_path) + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, json_path)
        return npz_path


def _fps(times: np.ndarray) -> "float | None":
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return float((len(times) - 1) / (times[-1] - times[0]))


@dataclass
class LoggedAttempt:
    path: Path
    poses: list
    states: np.ndarray  # per-frame CaptureBuffer state code, as recorded live
    timestamps: np.ndarray
    meta: dict

    @property
    def fps(self) -> "float | None":
        return _fps(self.timestamps)


def load_attempt(npz_path: "str | Path") -> LoggedAttempt:
    npz_path = Path(npz_path)
    with np.load(npz_path) as d:
        kp, sc = d["keypoints"], d["scores"]
        states, times = d["states"], d["timestamps"]
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    poses = [Pose(kp[t], sc[t]) for t in range(len(kp))]
    return LoggedAttempt(npz_path, poses, states, times, meta)


@dataclass
class ReplayResult:
    state: str                          # final CaptureBuffer state
    span: "tuple[int, int] | None"      # captured [start, stop) into the stream, if settled
    settle_index: "int | None"          # stream index of the frame that settled it


def replay_capture(poses: list, energy_fn: Callable[[list], "object"], motion_threshold: float,
                   settle_frames: int, preroll: int, max_frames: int) -> ReplayResult:
    """Feed `poses` through a real CaptureBuffer frame by frame; stop at the
    first settle, exactly as the live loop freezes there."""
    buf = CaptureBuffer(lambda idxs: energy_fn([poses[i] for i in idxs]), motion_threshold,
                        settle_frames=settle_frames, preroll=preroll, max_frames=max_frames)
    for i in range(len(poses)):
        buf.append(i)
        if buf.state == "settled":
            return ReplayResult("settled", (buf.frames[0], buf.frames[-1] + 1), i)
    return ReplayResult(buf.state, None, None)


def longest_internal_pause(energy: np.ndarray, threshold: float, start: int, stop: int) -> int:
    """Longest run of consecutive at-or-below-threshold frames strictly
    BETWEEN the first and last above-threshold frames of energy[start:stop]
    -- the inter-word pauses a sentence-mode settle window has to ride
    through. 0 if there is at most one motion burst."""
    active = np.nonzero(np.asarray(energy[start:stop]) > threshold)[0]
    if len(active) < 2:
        return 0
    gaps = np.diff(active) - 1
    return int(gaps.max()) if len(gaps) else 0
