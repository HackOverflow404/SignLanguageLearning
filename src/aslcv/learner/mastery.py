"""Per-sign, per-parameter mastery -- the learner model Phase 6's scheduler reads.

Deliberately decoupled from `grading.embedding_grader.ParameterVerdict`: `update()`
takes a plain `dict[parameter -> bool | None]` (the caller pulls `.correct` off each
real verdict), not the dataclass itself. Mastery is a downstream *consumer* of a
grading verdict, never a second opinion on it -- it has no way to produce a
correct/incorrect judgment on its own, only to remember ones EmbeddingGrader already
made. Kept dependency-light on purpose: this module (and its tests) never need to
import torch or load a checkpoint.

A `None` verdict -- MIN_SUPPORT insufficient data, see phonology_labels.py -- is
skipped entirely, never averaged in as a fixed 0.5: an unresolved question isn't
evidence of medium mastery, it's an absence of a verdict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

INITIAL_MASTERY = 0.5  # neutral prior for "never attempted" -- neither known-weak nor known-strong
LEARNING_RATE = 0.3    # EMA step; how fast one verdict moves the estimate


class MasteryState:
    """sign -> parameter -> mastery in [0, 1], plus enough bookkeeping (attempt
    counts, a logical clock for recency) for the scheduler's gap-targeting +
    recency bias. Never wall-clock timestamps -- a plain incrementing counter is
    deterministic and trivially testable, and "how many attempts ago" is all the
    scheduler actually needs, not real elapsed time."""

    def __init__(self):
        self.mastery: dict[str, dict[str, float]] = {}
        self.attempts: dict[str, dict[str, int]] = {}
        self.last_seen: dict[str, int] = {}
        self.clock: int = 0

    def sign_mastery(self, sign: str) -> float:
        """Mean mastery over this sign's ATTEMPTED parameters only -- a sign with
        zero attempts reads as the neutral prior, not as "known weak," so it
        competes fairly with genuinely-drilled-and-still-weak signs."""
        params = self.mastery.get(sign)
        if not params:
            return INITIAL_MASTERY
        return sum(params.values()) / len(params)

    def parameter_mastery(self, sign: str, parameter: str) -> float:
        return self.mastery.get(sign, {}).get(parameter, INITIAL_MASTERY)

    def attempts_for(self, sign: str) -> int:
        return sum(self.attempts.get(sign, {}).values())

    def update(self, sign: str, correct_by_parameter: dict[str, "bool | None"]) -> None:
        """One graded attempt's worth of per-parameter verdicts. EMA toward 1.0
        on correct, 0.0 on incorrect; a `None` (insufficient support) entry is
        skipped -- see module docstring. Advances the clock once per call and
        records this sign as seen at the new tick, regardless of whether any
        parameter had a usable verdict (the attempt still happened)."""
        self.clock += 1
        self.last_seen[sign] = self.clock
        sign_mastery = self.mastery.setdefault(sign, {})
        sign_attempts = self.attempts.setdefault(sign, {})
        for parameter, correct in correct_by_parameter.items():
            if correct is None:
                continue
            target = 1.0 if correct else 0.0
            old = sign_mastery.get(parameter, INITIAL_MASTERY)
            sign_mastery[parameter] = old + LEARNING_RATE * (target - old)
            sign_attempts[parameter] = sign_attempts.get(parameter, 0) + 1

    def to_dict(self) -> dict:
        return dict(mastery=self.mastery, attempts=self.attempts,
                    last_seen=self.last_seen, clock=self.clock)

    @classmethod
    def from_dict(cls, data: dict) -> "MasteryState":
        state = cls()
        state.mastery = {s: dict(p) for s, p in data.get("mastery", {}).items()}
        state.attempts = {s: dict(p) for s, p in data.get("attempts", {}).items()}
        state.last_seen = dict(data.get("last_seen", {}))
        state.clock = int(data.get("clock", 0))
        return state

    def save(self, path: "str | Path") -> None:
        """Atomic write (temp file + os.replace), same convention as
        extract_landmarks.py's write_npz -- a session killed mid-write must
        never leave a corrupt state file that looks complete."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: "str | Path") -> "MasteryState":
        """A missing file is a fresh learner, not an error -- the very first
        session has no prior state to load."""
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))
