"""Motion-aware live capture boundary detection -- replaces a fixed-size
trailing window with a buffer that grows from the start of motion to a
natural rest boundary, the same signal `features.py`'s `trim_to_motion`
already uses for cached clips (see `hand_motion_energy`'s own docstring:
"it is meant to be the SAME signal Phase 6's live boundary detector uses
online" -- this is that detector, not previously built).

Why this matters: a fixed `deque(maxlen=N)` trailing window silently evicts
its OLDEST frames as new ones arrive. A signer slower than N frames gets
only the TAIL of their attempt graded -- exactly the kind of corruption that
would hit `repeated_movement` hardest (it needs the full cyclic pattern to
judge periodicity) and can catch `handshape` mid-transition. `CaptureBuffer`
instead tracks three states:

  idle    -- no motion; buffer trimmed to a short PREROLL so a sign that
             starts abruptly still gets a natural rest lead-in (mirrors how
             ASL Citizen clips are framed: rest -> sign -> rest), without
             growing forever while nothing is happening.
  active  -- motion detected; keeps accumulating (up to `max_frames`, a
             safety cap so a stuck/very slow session doesn't grow unbounded).
  settled -- motion has dropped back below threshold for `settle_frames`
             consecutive frames (or `max_frames` was hit) -- a complete
             attempt, frozen until the caller calls `reset()`.

`energy_fn` is injected (a `list[Pose] -> np.ndarray` callable) rather than
importing `hand_motion_energy` directly, so this state machine is fully
unit-testable with a synthetic energy signal -- no real pose/MediaPipe
fixtures needed to test the logic itself.
"""
from __future__ import annotations

from typing import Callable, List


class CaptureBuffer:
    def __init__(self, energy_fn: Callable[[list], "object"], motion_threshold: float,
                 settle_frames: int = 10, preroll: int = 15, max_frames: int = 150):
        self.energy_fn = energy_fn
        self.motion_threshold = motion_threshold
        self.settle_frames = settle_frames
        self.preroll = preroll
        self.max_frames = max_frames
        self.frames: List = []
        self.state = "idle"  # "idle" | "active" | "settled"
        self._energy_ok = True  # False once energy_fn has raised -- degrade, don't crash the loop

    def reset(self) -> None:
        self.frames = []
        self.state = "idle"

    def append(self, pose) -> None:
        if self.state == "settled":
            return  # frozen -- caller must reset() (the demo's [c]/[n] keys do)

        self.frames.append(pose)

        if not self._energy_ok:
            # Degraded mode: energy_fn is unusable (e.g. a normalizer without
            # hand blocks) -- fall back to a plain trailing window so the
            # live loop still works, just without the settle detection.
            self.frames = self.frames[-self.max_frames:]
            self.state = "active" if len(self.frames) >= self.preroll else "idle"
            return

        try:
            energy = self.energy_fn(self.frames)
        except Exception:  # noqa: BLE001 -- any energy-computation failure degrades, never crashes
            self._energy_ok = False
            return

        if len(energy) < 2:
            return  # not enough frames for a velocity-based signal yet

        if self.state == "idle":
            recent = energy[-3:]
            if len(recent) and max(recent) > self.motion_threshold:
                self.state = "active"
            else:
                self.frames = self.frames[-self.preroll:]
            return

        # state == "active"
        if len(self.frames) >= self.max_frames:
            self.state = "settled"
            return
        tail = energy[-self.settle_frames:]
        if len(tail) >= self.settle_frames and max(tail) <= self.motion_threshold:
            self.state = "settled"
