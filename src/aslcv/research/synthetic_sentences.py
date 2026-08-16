"""Shared synthetic-continuous-sentence generation -- factored out of
`scripts/eval_forced_alignment.py` (Phase 7 step 4's validation benchmark)
so the CTC-CSLR comparison (`scripts/train_ctc_cslr.py`,
`scripts/eval_ctc_vs_alignment.py`) evaluates on IDENTICALLY-generated
trials, not a second independent implementation that could silently drift
and produce a comparison that isn't actually apples-to-apples.

The trick throughout: concatenate REAL, TRIMMED, isolated reference clips
end-to-end for a random sample of curriculum signs -- the same approach
`production.retrieval.ComposedReference` uses for video, at benchmark scale,
keeping the ground-truth per-gloss frame boundaries this time since the
concatenation is done here rather than by a live continuous attempt.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from ..production.gloss_rules import Gloss, GlossSequence
from ..production.retrieval import ReferenceClip, _trimmed_poses, skeleton_for

REPO = Path(__file__).resolve().parents[3]


def rows_by_sign_and_split(extractor: str) -> dict:
    """id_gloss -> split -> [manifest rows with a cached npz for `extractor`]."""
    rows = list(csv.DictReader(open(REPO / "data" / "manifest.csv")))
    cache = REPO / "data" / "cache" / extractor
    by: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (cache / f"{r['video_id']}.npz").exists():
            by[r["id_gloss"]][r["split"]].append(r)
    return by


def row_to_clip(row: dict, id_gloss: str, extractor: str, n_available: int) -> ReferenceClip:
    return ReferenceClip(
        id_gloss=id_gloss, asllex_code=row["asllex_code"], video_id=row["video_id"],
        video_path=REPO / row["video_path"],
        npz_path=REPO / "data" / "cache" / extractor / f"{row['video_id']}.npz",
        signer_id=row["signer_id"], split=row["split"], n_available=n_available,
    )


def make_trial(by_sign_split: dict, extractor: str, skeleton, k: int, rng: random.Random,
                *, split: str = "val"):
    """A random k-sign synthetic sentence: a GlossSequence, the concatenated
    trimmed-poses "attempt" (drawn from `split`), each gloss's true trimmed
    frame count, and the per-gloss trimmed-pose segments. None if the sign
    pool can't supply k signs with BOTH a train clip (what the reference side
    of an alignment/CTC comparison is built from) and a distinct clip in
    `split` (the "attempt" -- val by default, so training on train-split
    clips and evaluating on val-split clips never reuses identical footage).
    """
    eligible = [s for s, splits in by_sign_split.items() if splits.get("train") and splits.get(split)]
    if len(eligible) < k:
        return None
    chosen = rng.sample(eligible, k)

    glosses = [Gloss(text=s, asllex_id=s, source=s, pos="NOUN") for s in chosen]
    seq = GlossSequence(english=" ".join(chosen), in_scope=True, confidence=1.0, reason=None,
                         sentence_type="statement", negated=False, glosses=glosses)

    attempt_poses, true_lengths, segments = [], [], []
    for s in chosen:
        row = rng.choice(by_sign_split[s][split])
        clip = row_to_clip(row, s, extractor, len(by_sign_split[s][split]))
        trimmed = _trimmed_poses(clip, skeleton)
        attempt_poses.extend(trimmed)
        true_lengths.append(len(trimmed))
        segments.append(trimmed)
    return seq, attempt_poses, true_lengths, segments
