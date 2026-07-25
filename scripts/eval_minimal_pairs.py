#!/usr/bin/env python3
"""Minimal-pair confusability, all four extractors, every in-slice contrastive pair.

WHY THIS EXISTS: eval_slice.py's mother/father check (dwpose) reported father
leaking into mother 2/3 of the time; live_demo.py --selftest (mediapipe) reported
the contrast holding. Those are different backends, so a real difference is
possible -- but each verdict rested on 3 val clips per sign, nowhere near enough to
tell a real effect from noise. This script:

  1. Runs the IDENTICAL methodology across all four extractors on the same clips,
     so backend is the only variable (see the per-extractor table).
  2. Pools TRAIN + VAL + TEST clips per sign for statistical power. This is a
     DIAGNOSTIC READ of the features, not a model-selection decision -- no accuracy
     number reported anywhere else in this project uses test-split clips, and this
     script reports none either (only confusion RATES, printed as such below).
     Train clips are graded with LEAVE-ONE-OUT so a clip already sitting in its own
     sign's reference bank is never compared against itself.
  3. Generalizes from one pair to every contrastive pair in curriculum.yaml whose
     BOTH members fall in the reference bank (the 20-sign phase2_slice by default,
     or the full 60-sign curriculum with --full-curriculum), grouped by which
     phonological parameter differs -- telling us which parameter the
     training-free DTW features are weak on, not just whether one pair separates.

Method per query clip: build one N-way DTW reference bank (TRAIN split of the bank's
signs, per extractor) exactly as eval_slice.py does (N=20 by default, N=60 with
--full-curriculum). Grade every clip of every sign that is a member of at least one
in-bank pair against that bank. A clip is "confused" for pair (A, B) if its DTW
distance to the OTHER member is smaller than its distance to its own true sign.

--full-curriculum: the 20-sign phase2_slice has ZERO in-slice pairs for `movement`
(3 pairs in the full curriculum) and `repeated` (4 pairs) -- every pair for those
two parameters has at least one member outside the slice, so they can't be
evaluated without widening the bank. This flag widens it to all 60 signs. The
bank size changes the task's own difficulty (more distractor signs to be graded
against), so a 60-way confusion rate is NOT comparable to a 20-way one -- this
script always reports which bank size produced a given number and never diffs
across them.

VITPOSE CAVEAT (read before trusting that column): its 133-keypoint index order
was unverified against COCO_WHOLEBODY until scripts/verify_vitpose_topology.py
checked it against dwpose on shared cached clips -- confirmed correct (no
left/right permutation). It remains confounded by INPUT RESOLUTION, though:
vitpose runs its pose model at (192, 256) vs. dwpose/rtmw's (288, 384) -- 2.25x
the pixel area, and no matching-resolution easy_ViTPose checkpoint is known to
exist. Any vitpose-vs-others gap below is confounded by that, not architecture
alone. See scripts/verify_vitpose_topology.py and project_workflow.md Phase 3.

CONFIDENCE-MODE CAVEAT: MediaPipe's hand scores are HANDEDNESS confidence
(~0.5-1.0, "which hand", not keypoint quality) and its face scores a hardcoded
1.0, while dwpose/rtmw/vitpose give a graded per-point score -- so under the
default --confidence graded, part of any mediapipe-vs-rtmlib gap below may be
this SEMANTIC difference in what the confidence channel even means, not
keypoint quality. Pass --confidence binary to threshold every backend's scores
to {0,1} and see how much of the gap survives.

PIPELINE CONSTRUCTION: this script, eval_slice.py, and live_demo.py all build
their FeaturePipeline through aslcv.pipeline_config.build_pipeline, so their
face/legs_feet/confidence/velocity/local_hand toggles can never silently
diverge -- see that module for the shared flags (all listed under --help
below) and the printed "pipeline config: ..." line every run emits for
reproducibility.

    .venv/bin/python scripts/eval_minimal_pairs.py                        # 20-way, all 4 extractors
    .venv/bin/python scripts/eval_minimal_pairs.py --extractor dwpose
    .venv/bin/python scripts/eval_minimal_pairs.py --full-curriculum       # 60-way, covers movement/repeated
    .venv/bin/python scripts/eval_minimal_pairs.py --confidence binary     # binarized confidence, all backends
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.features import Standardizer
from aslcv.grading.dtw_grader import dtw_distance
from aslcv.pipeline_config import add_pipeline_args, build_pipeline

REPO = Path(__file__).resolve().parents[1]
ALL_EXTRACTORS = ["mediapipe", "dwpose", "rtmw", "vitpose"]
PARAMETERS = ("handshape", "major_location", "minor_location", "movement", "repeated")

# Fewer pairs than this and a parameter's aggregate confusion rate is one or two
# outlier pairs away from flipping -- flag it rather than let it feed a ranking
# claim ("handshape is hardest") the sample can't actually support.
SMALL_N_PAIRS = 4

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(it, **kw):
        return it


def skeleton_for(extractor):
    return MEDIAPIPE_HOLISTIC if extractor == "mediapipe" else COCO_WHOLEBODY


def load_bank_and_pairs(full_curriculum: bool):
    """(bank sign set, {parameter: [(a, b, differs), ...]}) restricted to pairs
    where BOTH members are in the bank.

    Default bank = the 20-sign phase2_slice (matches eval_slice.py's val-split
    baseline). --full-curriculum widens the bank to all 60 curriculum signs --
    needed because `movement` and `repeated` have ZERO pairs with both members in
    the 20-sign slice (every such pair has at least one member outside it)."""
    doc = yaml.safe_load(open(REPO / "curriculum.yaml"))
    if full_curriculum:
        bank_signs = {s["gloss"] for unit in doc["units"] for s in unit["signs"]}
    else:
        bank_signs = set(doc["phase2_slice"])
    pairs_by_param = {}
    for param in PARAMETERS:
        in_bank = [
            (e["pair"][0], e["pair"][1], e["differs"])
            for e in doc["contrastive_pairs"].get(param, [])
            if e["pair"][0] in bank_signs and e["pair"][1] in bank_signs
        ]
        pairs_by_param[param] = in_bank
    return bank_signs, pairs_by_param


class LOOBank:
    """sign -> [(video_id, standardized (T, F) features)] -- TRAIN clips of the
    bank's signs (20 or 60), kept with their video_id so a train query can be
    excluded from its OWN sign's bank (leave-one-out) instead of trivially
    matching itself."""

    def __init__(self, references, standardizer):
        self.references = references
        self.standardizer = standardizer

    @classmethod
    def build(cls, pipeline, cache_dir, bank_signs, train_rows):
        raw_refs = defaultdict(list)
        raw_all = []
        for r in train_rows:
            if r["id_gloss"] not in bank_signs:
                continue
            npz = cache_dir / f"{r['video_id']}.npz"
            if not npz.exists():
                continue
            feats = pipeline.assemble_npz(npz).features
            raw_refs[r["id_gloss"]].append((r["video_id"], feats))
            raw_all.append(feats)
        if not raw_all:
            raise RuntimeError(f"no reference clips found under {cache_dir}")
        standardizer = Standardizer.fit(raw_all)
        references = {
            s: [(vid, standardizer.transform(f)) for vid, f in items]
            for s, items in raw_refs.items()
        }
        return cls(references, standardizer)

    def grade(self, query_video_id, query_features, *, agg="min", band=None):
        """Ranked [(sign, distance), ...] over every sign in the bank; the query is
        left out of its own sign's bank if it happens to be one of its references."""
        return self.distances_to(query_video_id, query_features, self.references.keys(),
                                  agg=agg, band=band)

    def distances_to(self, query_video_id, query_features, target_signs, *, agg="min", band=None):
        """Like `grade`, but computes DTW against only `target_signs` instead of every
        sign in the bank. Numerically IDENTICAL to reading those same signs' entries
        out of `grade`'s full ranking (same distance function, same leave-one-out
        exclusion) -- this exists purely so a caller who only needs a handful of a
        large bank's distances (e.g. confusion_for needs exactly 2 of 60 signs) isn't
        forced to pay for a full ranking it will mostly discard. Still returns a
        sorted [(sign, distance), ...] list for interface parity with `grade`."""
        att = self.standardizer.transform(np.asarray(query_features, dtype=np.float32))
        ranked = []
        for sign in target_signs:
            refs = self.references[sign]
            dists = [dtw_distance(att, feat, band) for vid, feat in refs if vid != query_video_id]
            val = float("inf") if not dists else (min(dists) if agg == "min" else float(np.mean(dists)))
            ranked.append((sign, val))
        ranked.sort(key=lambda sd: sd[1])
        return ranked


def confusion_for(graded, a, b):
    """(#confused, #total) over every clip of a or b: confused = the clip's DTW
    distance to the OTHER member of the pair is smaller than to its own true sign."""
    n = confused = 0
    for true_sign, ranked in graded.values():
        if true_sign not in (a, b):
            continue
        other = b if true_sign == a else a
        dist = dict(ranked)
        n += 1
        confused += dist[other] < dist[true_sign]
    return confused, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="all", choices=ALL_EXTRACTORS + ["all"])
    ap.add_argument("--agg", default="min", choices=["min", "mean"])
    ap.add_argument("--band", type=int, default=None, help="Sakoe-Chiba radius (frames)")
    ap.add_argument("--full-curriculum", action="store_true",
                    help="widen the reference bank from the 20-sign phase2_slice to all 60 "
                         "curriculum signs -- needed to cover movement/repeated pairs, which "
                         "have ZERO pairs with both members in-slice. NOT comparable to a "
                         "20-way run's confusion rates (different task difficulty).")
    add_pipeline_args(ap)
    args = ap.parse_args()
    extractors = ALL_EXTRACTORS if args.extractor == "all" else [args.extractor]

    bank_signs, pairs_by_param = load_bank_and_pairs(args.full_curriculum)
    bank_label = f"{len(bank_signs)}-way ({'full 60-sign curriculum' if args.full_curriculum else '20-sign phase2_slice'})"
    involved_signs = sorted({s for pairs in pairs_by_param.values() for a, b, _ in pairs for s in (a, b)})
    if not involved_signs:
        raise SystemExit("no in-bank pairs found for any parameter -- nothing to evaluate")

    print("=" * 78)
    print(f"Minimal-pair confusability across extractors -- {bank_label} reference bank.")
    print("DIAGNOSTIC, not a model-selection metric. Pools TRAIN + VAL + TEST clips per")
    print("sign for statistical power (train clips graded leave-one-out).")
    print("Bank size changes task difficulty -- NEVER compare these rates against a run")
    print("with a different bank size (20-way vs 60-way).")
    if "vitpose" in extractors:
        print("VITPOSE CAVEAT: topology confirmed correct (verify_vitpose_topology.py),")
        print("but its input resolution (192x256) is 2.25x smaller than dwpose/rtmw's")
        print("(288x384) -- any vitpose-vs-others gap below is confounded by that.")
    if "mediapipe" in extractors and len(extractors) > 1 and args.confidence == "graded":
        print("CONFIDENCE-MODE CAVEAT: mediapipe's confidence channel is semantically")
        print("different (handedness/hardcoded-1.0, not per-point quality) from the")
        print("rtmlib backends' graded scores -- part of any mediapipe-vs-others gap")
        print("below may be this, not keypoint quality. Re-run with --confidence binary")
        print("to see how much of the gap survives.")
    print("=" * 78)
    print(f"\nIn-bank contrastive pairs (both members in the {bank_label} bank of signs):")
    for param in PARAMETERS:
        pairs = pairs_by_param[param]
        if not pairs:
            reason = ("no minimal pair for this parameter has both members in the 60-sign "
                      "curriculum (shouldn't happen -- check curriculum.yaml)" if args.full_curriculum
                      else "no minimal pair for this parameter has both members in the 20-sign "
                           "slice; pass --full-curriculum to cover it")
            print(f"  {param:<15} 0 pairs -- {reason}")
            continue
        for a, b, differs in pairs:
            print(f"  {param:<15} {a} vs {b}   (differs: {differs})")

    rows = list(csv.DictReader(open(REPO / "data" / "manifest.csv")))
    by_sign = defaultdict(list)
    for r in rows:
        if r["id_gloss"] in involved_signs:
            by_sign[r["id_gloss"]].append(r)
    train_rows_bank = [r for r in rows if r["id_gloss"] in bank_signs and r["split"] == "train"]
    n_query_clips = sum(len(by_sign[s]) for s in involved_signs)
    print(f"\n{len(involved_signs)} signs involved in an in-bank pair: {', '.join(involved_signs)}")
    print(f"{n_query_clips} query clips total (train+val+test, all signs above)")

    # A query clip of sign S only ever needs its distance to S itself and to
    # whichever sign(s) S is paired against (confusion_for reads exactly those two
    # per pair) -- never the OTHER (bank_size - 2) signs' distances. On the 60-way
    # bank that is a >10x reduction in DTW work per query versus ranking the whole
    # bank, with numerically identical results (see LOOBank.distances_to).
    targets_for_sign: dict[str, set] = defaultdict(set)
    for pairs in pairs_by_param.values():
        for a, b, _ in pairs:
            targets_for_sign[a] |= {a, b}
            targets_for_sign[b] |= {a, b}

    # -- grade every involved clip, per extractor ----------------------------
    all_graded = {}
    for extractor in extractors:
        cache_dir = REPO / "data" / "cache" / extractor
        if not cache_dir.is_dir():
            print(f"\n[{extractor}] SKIPPED: no cache at {cache_dir}")
            continue
        skeleton = skeleton_for(extractor)
        pipeline = build_pipeline(args, skeleton, extractor_name=extractor)
        print(f"[{extractor}] building {bank_label} reference bank from TRAIN ...")
        bank = LOOBank.build(pipeline, cache_dir, bank_signs, train_rows_bank)

        graded = {}
        query_rows = [r for s in involved_signs for r in by_sign[s]]
        for r in tqdm(query_rows, desc=f"[{extractor}] grading", unit="clip"):
            npz = cache_dir / f"{r['video_id']}.npz"
            if not npz.exists():
                continue
            feats = pipeline.assemble_npz(npz).features
            targets = targets_for_sign[r["id_gloss"]]
            ranked = bank.distances_to(r["video_id"], feats, targets, agg=args.agg, band=args.band)
            graded[r["video_id"]] = (r["id_gloss"], ranked)
        all_graded[extractor] = graded

    # -- per-pair, per-extractor table ---------------------------------------
    print("\n" + "=" * 78)
    print("Per-pair confusion rate by extractor")
    print("(fraction of clips, across both members, graded closer to the OTHER")
    print(" member of the pair than to their own true sign; confused/n)")
    print("=" * 78)
    col_w = 16
    print(f"{'parameter':<15}{'pair':<16}" + "".join(f"{e:>{col_w}}" for e in extractors))
    pair_stats = []  # (param, a, b, differs, {extractor: (confused, n)})
    for param in PARAMETERS:
        for a, b, differs in pairs_by_param[param]:
            per_ext = {}
            for extractor in extractors:
                graded = all_graded.get(extractor)
                if graded is None:
                    continue
                per_ext[extractor] = confusion_for(graded, a, b)
            pair_stats.append((param, a, b, differs, per_ext))
            cells = ""
            for extractor in extractors:
                c, n = per_ext.get(extractor, (0, 0))
                cell = f"{c}/{n}={c / n:.0%}" if n else "n/a"
                cells += f"{cell:>{col_w}}"
            print(f"{param:<15}{a + '/' + b:<16}{cells}")

    # -- parameter-level rollup: PER BACKEND, not pooled ---------------------
    # A pooled-across-backends number previously supported a claim ("handshape is
    # the weakest parameter") that only held in dwpose's column -- mediapipe
    # disagreed -- on aggregates built from just 1-3 pairs. Report each backend's
    # own rate so that kind of false consensus can't hide again, and flag any
    # parameter whose rate rests on too few pairs to support ranking it at all.
    print("\n" + "=" * 78)
    print(f"Parameter -> confusion rate, PER BACKEND ({bank_label} bank)")
    print("Do NOT rank parameters against each other unless the pattern holds across")
    print("EVERY backend below -- most rows rest on very few pairs (see the flag column).")
    print("=" * 78)
    print(f"{'parameter':<16}{'n pairs':>9}" + "".join(f"{e:>{col_w}}" for e in extractors) + "   flag")
    param_rates: dict[str, dict[str, float]] = {}  # param -> {extractor: rate}
    for param in PARAMETERS:
        pairs = pairs_by_param[param]
        if not pairs:
            print(f"{param:<16}{0:>9}" + "".join(f"{'--':>{col_w}}" for _ in extractors)
                  + f"   no {bank_label} pairs")
            continue
        cells = ""
        param_rates[param] = {}
        for extractor in extractors:
            total_n = total_confused = 0
            for p, a, b, differs, per_ext in pair_stats:
                if p != param:
                    continue
                c, n = per_ext.get(extractor, (0, 0))
                total_confused += c
                total_n += n
            rate = total_confused / total_n if total_n else float("nan")
            param_rates[param][extractor] = rate
            cell = f"{rate:.0%} (n={total_n})" if total_n else "n/a"
            cells += f"{cell:>{col_w}}"
        flag = f"only {len(pairs)} pair(s) -- do not rank on this" if len(pairs) < SMALL_N_PAIRS else ""
        print(f"{param:<16}{len(pairs):>9}{cells}   {flag}")

    # cross-backend agreement check: does the SAME parameter come out hardest
    # (highest rate) in every backend's own column? Only say so if it actually does.
    print()
    hardest_per_backend = {}
    for extractor in extractors:
        ranked_params = sorted(
            (p for p in param_rates if not np.isnan(param_rates[p].get(extractor, float("nan")))),
            key=lambda p: param_rates[p][extractor], reverse=True,
        )
        if ranked_params:
            hardest_per_backend[extractor] = ranked_params[0]
    if hardest_per_backend and len(set(hardest_per_backend.values())) == 1:
        (only_hardest,) = set(hardest_per_backend.values())
        print(f"VERDICT: '{only_hardest}' comes out as the hardest parameter (highest confusion) "
              f"in EVERY backend's own column -- a claim the data actually supports.")
    else:
        print("VERDICT: the hardest parameter DISAGREES across backends "
              f"({', '.join(f'{e}->{p}' for e, p in hardest_per_backend.items())}) -- "
              "there is no cross-backend-consistent ranking here. Read each backend's "
              "column on its own; do not claim one parameter is globally weakest.")
    print("\nA LOW confusion rate in a given backend's column is one that backend's")
    print("shoulder-frame + DTW features already separate; a HIGH rate means Phase 4's")
    print("phonological heads need to specifically target that parameter for that backend.")


if __name__ == "__main__":
    main()
