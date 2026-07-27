#!/usr/bin/env python3
"""compose_sentence.py -- text-only gloss composer: English in, a PLAYBACK PLAN
out, no video. The most end-to-end thing testable before Phase 5a exists.

    STEP 1  english -> GlossSequence          (gloss_rules.GlossRuleEngine)
    STEP 2  each gloss -> asllex_code          (curriculum.yaml, join key per
                                                 CLAUDE.md: asllex_code, never
                                                 gloss strings)
    STEP 3  asllex_code -> cached reference clip(s)
                                                (data/cache/{extractor}/_manifest.csv)
    STEP 4  ordered ClipPlan                   (what Phase 5a will retrieve +
                                                 concatenate -- this script stops
                                                 at the plan, deliberately)

Fail-closed end to end: if the rule engine refuses, or ANY gloss has no cached
clip, composition stops there and says exactly why. No partial plan is ever
printed as if it were usable.

IMPORTANT: this validates the PIPELINE (can every step in the chain resolve
something concrete?), not ASL correctness. A plan built from stitched
citation clips is not fluent ASL and cannot be judged by eye -- see
tests/test_gloss_rules_corpus.py's PENDING_CASES for what still needs Deaf
review before any of this reaches a learner.

    .venv/bin/python scripts/compose_sentence.py "I work today."
    .venv/bin/python scripts/compose_sentence.py "Are you tired?" --extractor dwpose
"""
import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from aslcv.production.gloss_rules import GlossRuleEngine, GlossSequence  # noqa: E402

CURRICULUM_PATH = REPO / "curriculum.yaml"
CACHE = REPO / "data" / "cache"
EXTRACTORS = ("mediapipe", "dwpose", "rtmw", "vitpose")

BANNER = """\
================================================================================
 This validates the PIPELINE -- every gloss resolves to a real reference clip
 end to end. It does NOT validate ASL correctness. Stitched citation clips are
 not fluent ASL and cannot be judged by eye; word order/NMM placement is
 pending Deaf review (see tests/test_gloss_rules_corpus.py PENDING_CASES).
================================================================================"""


def _curriculum_by_gloss() -> dict[str, dict]:
    doc = yaml.safe_load(open(CURRICULUM_PATH))
    return {s["gloss"]: s for u in doc["units"] for s in u["signs"]}


def _manifest_rows(extractor: str) -> list[dict]:
    mpath = CACHE / extractor / "_manifest.csv"
    if not mpath.exists():
        sys.exit(f"no cache manifest at {mpath} -- run "
                  f"scripts/extract_landmarks.py --extractor {extractor} first")
    with open(mpath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class ClipChoice:
    """One gloss's resolved reference clip -- the unit Phase 5a will retrieve
    and concatenate. Deliberately flat and self-contained so a future
    retrieval step can swap `pick_clip`'s selection strategy (e.g. best-match
    to the learner's signer, or several candidates for variety) without
    touching anything upstream of it in this file."""

    gloss: str
    asllex_code: str
    n_clips_available: int
    video_id: str
    video_path: str
    signer_id: str
    split: str


@dataclass
class ComposePlan:
    sentence: str
    seq: GlossSequence
    extractor: str
    clips: list[ClipChoice] = field(default_factory=list)

    @property
    def clip_ids(self) -> list[str]:
        return [c.video_id for c in self.clips]


def pick_clip(rows: list[dict]) -> dict:
    """Deterministic default clip for a gloss: prefer a train-split clip (what
    a reference bank actually uses), else the first match. Same rule as
    render_clip.py's pick_row, so `--video-id`-style manual inspection of a
    plan's clip agrees with what render_clip.py shows for that entry."""
    return next((r for r in rows if r["split"] == "train"), rows[0])


def compose(sentence: str, extractor: str, engine: GlossRuleEngine) -> ComposePlan | None:
    """Runs all four steps. Returns None (having already printed the reason)
    if composition fails at any step -- fail-closed all the way through, no
    partial plan is ever returned."""

    # STEP 1 -- english -> GlossSequence
    seq = engine.gloss(sentence)
    print(f"STEP 1: rule engine\n  english: {sentence!r}")
    if not seq.in_scope:
        print(f"  REFUSED: {seq.reason}")
        print("\nNOT COMPOSABLE -- the rule engine refused this sentence before any "
              "retrieval was attempted. No partial plan is produced.")
        return None
    print(f"  glosses: {' '.join(g.text for g in seq.glosses)}")
    print(f"  sentence_type={seq.sentence_type}  negated={seq.negated}  "
          f"confidence={seq.confidence}")

    curriculum = _curriculum_by_gloss()
    rows = _manifest_rows(extractor)
    by_code: dict[str, list[dict]] = {}
    for r in rows:
        by_code.setdefault(r["asllex_code"], []).append(r)

    # STEP 2 + 3 -- each gloss -> asllex_code -> cached clip(s)
    print(f"\nSTEP 2+3: resolve each gloss to a curriculum asllex_code, then to "
          f"cached clips (extractor={extractor})")
    plan = ComposePlan(sentence=sentence, seq=seq, extractor=extractor)
    missing: list[tuple[str, str]] = []
    for g in seq.glosses:
        sign = curriculum.get(g.asllex_id)
        if sign is None:
            # cannot happen given gloss_rules.py's own invariant (every
            # emittable gloss resolves to a curriculum sign) -- checked
            # anyway since this script must never silently trust that.
            missing.append((g.text, "not a curriculum sign (engine invariant violated)"))
            print(f"  [{g.text:<10}] NO curriculum entry for asllex_id={g.asllex_id!r}")
            continue
        code = sign["asllex_code"]
        clips = by_code.get(code, [])
        if not clips:
            missing.append((g.text, f"in curriculum (asllex_code={code}) but no "
                                     f"cached clip for extractor={extractor!r}"))
            print(f"  [{g.text:<10}] asllex_code={code:<10} 0 cached clips -- MISSING")
            continue
        chosen = pick_clip(clips)
        plan.clips.append(ClipChoice(
            gloss=g.text, asllex_code=code, n_clips_available=len(clips),
            video_id=chosen["video_id"], video_path=chosen["video_path"],
            signer_id=chosen["signer_id"], split=chosen["split"],
        ))
        print(f"  [{g.text:<10}] asllex_code={code:<10} {len(clips):>2} cached clip(s)  "
              f"e.g. {chosen['video_id']}")

    if missing:
        print(f"\nNOT COMPOSABLE -- {len(missing)} gloss(es) have no cached reference clip:")
        for gloss_text, why in missing:
            print(f"  {gloss_text}: {why}")
        return None

    # STEP 4 -- ordered playback plan (Phase 5a's retrieval+concatenation
    # input). Printed here, not built into video -- that is the whole point
    # of this being the pre-Phase-5a skeleton.
    print("\nSTEP 4: playback plan (ordered clip IDs Phase 5a would retrieve + "
          "concatenate)")
    print(f"  gloss sequence : {' '.join(c.gloss for c in plan.clips)}")
    if seq.nmm_tags:
        print("  nmm tags       :")
        for nm in seq.nmm_tags:
            covered = " ".join(c.gloss for c in plan.clips[nm.start:nm.end])
            print(f"    {nm.marker:<11} [{nm.start}:{nm.end})  over: {covered}")
    else:
        print("  nmm tags       : none")
    print("  clip playlist  :")
    for i, c in enumerate(plan.clips):
        print(f"    [{i}] {c.video_id:<40} ({c.gloss}, signer={c.signer_id}, split={c.split})")

    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence", help="English sentence to compose, e.g. \"I work today.\"")
    ap.add_argument("--extractor", default="mediapipe", choices=EXTRACTORS,
                     help="which cached extractor's clips to resolve against (default: mediapipe)")
    args = ap.parse_args()

    print(BANNER)
    print()
    engine = GlossRuleEngine()
    plan = compose(args.sentence, args.extractor, engine)
    print()
    if plan is None:
        raise SystemExit(1)
    print(f"composable: {len(plan.clips)} gloss(es), {len(plan.clip_ids)} clip(s) resolved.")


if __name__ == "__main__":
    main()
