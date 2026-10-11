"""Phase 8: the adaptive loop's grounded text, plus golden fixtures for its Dart port.

Writes models/mobile/text.json -- the only English the phone shows that comes
from ASL-LEX data rather than formatting:
  signs:      sign -> describe_sign() (the always-on "what the correct sign
              looks like" paragraph shown beside the reference clip)
  handshapes: handshape code -> describe_handshape(), only for codes that HAVE
              a grounded description (see handshape_descriptions.py)
Python stays the only author of this text; the phone looks it up, never
re-derives it from phonology.csv.

Also writes models/mobile/fixtures/learner.json for app/aslcv_core's parity
tests: readable_value() on every class value, coach_text()/focus_parameter() on
synthetic verdict sets, find_minimal_pairs() over the bundle's signs, and a
MasteryState after a fixed sequence of updates.

    .venv/bin/python scripts/export_learner_text.py
"""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

from aslcv.generator.feedback import coach_text, focus_parameter, readable_value
from aslcv.generator.handshape_descriptions import describe_handshape
from aslcv.generator.sign_description import describe_sign
from aslcv.grading.phonology_labels import ALL_PARAMETERS, PhonologyLabels
from aslcv.learner.mastery import MasteryState
from aslcv.learner.scheduler import find_minimal_pairs

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO / "models" / "mobile"


def _targets(meta, sign):
    """parameter -> (target value as the verdict carries it, judged)."""
    out = {}
    for p, t in meta["signs"][sign].items():
        value = str(t["target"]) if p == "repeated_movement" else t["target"]
        out[p] = (value, t["judged"])
    return out


def _coach_cases(meta, rng, n):
    classes = dict(meta["classes"], repeated_movement=["False", "True"])
    signs = sorted(meta["signs"])
    cases = []
    for i in range(n):
        sign = signs[i % len(signs)]
        verdicts = []
        for p, (target, judged) in _targets(meta, sign).items():
            # Cycle through all-right, all-wrong, and mixed patterns.
            wrong = {0: False, 1: True}.get(i % 4, rng.random() < 0.5)
            predicted = rng.choice([c for c in classes[p] if c != target]) if wrong else target
            correct = (predicted == target) if judged else None
            if i % 11 == 5:
                correct = None  # nothing judged at all
            verdicts.append(dict(parameter=p, predicted=predicted, target=target,
                                 correct=correct, confidence=round(rng.uniform(0.4, 1.0), 4)))
        ns = [SimpleNamespace(**v) for v in verdicts]
        cases.append(dict(verdicts=verdicts, text=coach_text(ns), focus=focus_parameter(ns)))
    return cases


def _mastery_case(meta, rng, steps):
    signs = sorted(meta["signs"])[:8]
    state = MasteryState()
    updates = []
    for _ in range(steps):
        sign = rng.choice(signs)
        verdicts = {p: rng.choice([True, True, False, None]) for p in ALL_PARAMETERS}
        updates.append(dict(sign=sign, verdicts=verdicts))
        state.update(sign, verdicts)
    due = {s: state.is_due(s) for s in signs}
    return dict(updates=updates, state=state.to_dict(), due=due,
                sign_mastery={s: state.sign_mastery(s) for s in signs})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((args.bundle / "bundle.json").read_text())
    pl = PhonologyLabels()
    signs = sorted(meta["signs"])

    handshapes = {c: d for c in meta["classes"]["handshape"] if (d := describe_handshape(c))}
    text = dict(format_version=1, signs={s: describe_sign(pl, s) for s in signs}, handshapes=handshapes)
    (args.bundle / "text.json").write_text(json.dumps(text, indent=1))

    rng = random.Random(args.seed)
    classes = dict(meta["classes"], repeated_movement=["False", "True"])
    fixture = dict(
        readable={p: {v: readable_value(p, v) for v in values} for p, values in classes.items()},
        coach=_coach_cases(meta, rng, 60),
        minimal_pairs={p: [list(pair) for pair in pairs] for p, pairs in find_minimal_pairs(pl, signs).items()},
        mastery=_mastery_case(meta, rng, 120),
    )
    out = args.bundle / "fixtures" / "learner.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1))
    print(f"wrote text.json ({len(signs)} sign descriptions, {len(handshapes)} handshape descriptions) "
          f"and {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
