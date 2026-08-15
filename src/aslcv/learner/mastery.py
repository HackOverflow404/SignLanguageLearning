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

# SM2-style spaced repetition, adapted to the logical attempt-clock this project
# already uses instead of SM2's usual wall-clock days: `interval` is measured in
# ATTEMPTS (of any sign) since this sign was last seen, not days. A sign becomes
# "due" once that many attempts have passed. Same shape as classic SM2 (ease
# factor grows/shrinks the interval, a miss resets it to the start) but with the
# magnitudes rescaled for a scale of tens-to-hundreds of attempts in one session
# rather than weeks of real time.
INITIAL_EASE = 2.5   # classic SM2 starting ease factor
MIN_EASE = 1.3        # classic SM2 floor -- an ease factor can't collapse to zero growth
MAX_EASE = 3.0        # not part of classic SM2; caps how fast intervals can balloon in one session
EASE_DELTA_GOOD = 0.05
EASE_DELTA_BAD = 0.2
SECOND_REP_INTERVAL = 3  # ticks -- scaled-down analogue of SM2's 6-day second interval
MAX_INTERVAL = 40         # ticks -- keeps a mastered sign reachable within one session instead of
                           # letting the interval balloon past a realistic session length


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
        self.interval: dict[str, int] = {}
        self.ease: dict[str, float] = {}
        self.repetitions: dict[str, int] = {}

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

    def _sm2_advance(self, sign: str, good: bool) -> None:
        """One pass/fail signal for `sign` as a whole (see `update()` for how
        "good" is derived from per-parameter verdicts). Mirrors classic SM2's
        ease-factor/interval update, binary quality instead of 0-5 since that's
        all a pass/fail verdict gives us."""
        ease = self.ease.get(sign, INITIAL_EASE)
        reps = self.repetitions.get(sign, 0)
        if good:
            reps += 1
            if reps == 1:
                interval = 1
            elif reps == 2:
                interval = SECOND_REP_INTERVAL
            else:
                interval = min(MAX_INTERVAL, round(self.interval.get(sign, 1) * ease))
            ease = min(MAX_EASE, ease + EASE_DELTA_GOOD)
        else:
            reps = 0
            interval = 1
            ease = max(MIN_EASE, ease - EASE_DELTA_BAD)
        self.repetitions[sign] = reps
        self.interval[sign] = interval
        self.ease[sign] = ease

    def due_at(self, sign: str) -> "int | None":
        """The tick at which `sign` next becomes due, or None if it has never
        been attempted (always due)."""
        last = self.last_seen.get(sign)
        if last is None:
            return None
        return last + self.interval.get(sign, 1)

    def is_due(self, sign: str) -> bool:
        due = self.due_at(sign)
        return due is None or self.clock >= due

    def update(self, sign: str, correct_by_parameter: dict[str, "bool | None"]) -> None:
        """One graded attempt's worth of per-parameter verdicts. EMA toward 1.0
        on correct, 0.0 on incorrect; a `None` (insufficient support) entry is
        skipped -- see module docstring. Advances the clock once per call and
        records this sign as seen at the new tick, regardless of whether any
        parameter had a usable verdict (the attempt still happened).

        Also advances the sign's SM2 interval: "good" means every JUDGED
        parameter matched (a real full pass, not an average) -- one wrong
        parameter is a miss for scheduling purposes, same standard `focus_parameter`
        already applies when deciding whether to coach. A `None`-verdict-only
        attempt (`judged` empty) carries no pass/fail evidence, so the interval
        is left untouched rather than guessed."""
        self.clock += 1
        self.last_seen[sign] = self.clock
        sign_mastery = self.mastery.setdefault(sign, {})
        sign_attempts = self.attempts.setdefault(sign, {})
        judged = []
        for parameter, correct in correct_by_parameter.items():
            if correct is None:
                continue
            judged.append(correct)
            target = 1.0 if correct else 0.0
            old = sign_mastery.get(parameter, INITIAL_MASTERY)
            sign_mastery[parameter] = old + LEARNING_RATE * (target - old)
            sign_attempts[parameter] = sign_attempts.get(parameter, 0) + 1
        if judged:
            self._sm2_advance(sign, good=all(judged))

    def to_dict(self) -> dict:
        return dict(mastery=self.mastery, attempts=self.attempts,
                    last_seen=self.last_seen, clock=self.clock,
                    interval=self.interval, ease=self.ease,
                    repetitions=self.repetitions)

    @classmethod
    def from_dict(cls, data: dict) -> "MasteryState":
        state = cls()
        state.mastery = {s: dict(p) for s, p in data.get("mastery", {}).items()}
        state.attempts = {s: dict(p) for s, p in data.get("attempts", {}).items()}
        state.last_seen = dict(data.get("last_seen", {}))
        state.clock = int(data.get("clock", 0))
        state.interval = dict(data.get("interval", {}))
        state.ease = dict(data.get("ease", {}))
        state.repetitions = dict(data.get("repetitions", {}))
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
