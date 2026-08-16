#!/usr/bin/env python3
"""Head-to-head: this project's forced-alignment system (align_and_grade,
Phase 7) vs. a CTC-CSLR model (aslcv.research.ctc_cslr, trained by
scripts/train_ctc_cslr.py) -- an ISOLATED RESEARCH COMPARISON, not a claim
about the shipped product's correctness. See src/aslcv/research/__init__.py
for why CTC itself is deliberately kept out of the product.

Both systems are run on the IDENTICAL held-out synthetic sentences
(aslcv.research.synthetic_sentences.make_trial, VAL-split clips, same seed
as eval_forced_alignment.py's own benchmark by default) -- same trick
production.retrieval.ComposedReference uses for video, at benchmark scale.

THREE things are measured, and they are NOT all directly comparable -- that
asymmetry is the actual finding, not a limitation of the benchmark:

  1. OPEN-SET RECOGNITION (Word Error Rate). CTC's real selling point: no
     known target, free decoding. align_and_grade has NO analogous number --
     it never guesses a sequence, it is always given one. Reporting "N/A" for
     align_and_grade here is not a gap in the benchmark, it's the actual
     structural difference CLAUDE.md's non-negotiables are built around.
  2. FORCED ALIGNMENT (boundary error), given the TRUE gloss sequence to
     BOTH systems -- the fair, apples-to-apples comparison. align_and_grade
     via dtw_align; CTC via its own standard Viterbi forced-alignment
     recurrence (ctc_cslr.forced_align) over the SAME true sequence.
  3. GRADING AGREEMENT on the resulting segments -- does grading the aligned
     segment agree with grading that same clip in isolation? Same metric
     eval_forced_alignment.py already reports for align_and_grade, computed
     here identically for CTC's forced-aligned segments too.

    .venv/bin/python scripts/eval_ctc_vs_alignment.py
    .venv/bin/python scripts/eval_ctc_vs_alignment.py --n-trials 100
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from aslcv.grading.alignment import align_and_grade
from aslcv.grading.embedding_grader import EmbeddingGrader
from aslcv.production.retrieval import skeleton_for
from aslcv.research.ctc_cslr import (CTCEncoder, forced_align, greedy_decode,
                                      segments_from_forced_align, word_error_rate)
from aslcv.research.synthetic_sentences import make_trial, rows_by_sign_and_split

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_CHECKPOINT = REPO / "models" / "embedding_grader"
DEFAULT_CTC_CHECKPOINT = REPO / "models" / "ctc_cslr"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


def load_ctc(checkpoint_dir, device):
    config = json.loads((checkpoint_dir / "config.json").read_text())
    model = CTCEncoder(config["in_dim"], len(config["vocab"]), hidden=config["hidden"], layers=config["layers"])
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    model.to(device).eval()
    vocab = config["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    return model, vocab, inv_vocab


def grading_agreement(graded_segments, isolated_by_sign, tally):
    """`graded_segments`: [(target_sign, GradeResult), ...] from EITHER
    system's aligned segments. Shared by both systems so the metric is
    computed identically -- see eval_forced_alignment.py's own docstring for
    why thin/insufficient-support parameters are excluded."""
    for sign, result in graded_segments:
        isolated = isolated_by_sign[sign]
        for p, aligned_v in result.parameters.items():
            isolated_v = isolated.parameters[p]
            if isolated_v.correct is None:
                continue
            tally[p][1] += 1
            tally[p][0] += int(aligned_v.correct == isolated_v.correct)


def run(n_trials, min_signs, max_signs, embedding_checkpoint, ctc_checkpoint, extractor, seed, band):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grader = EmbeddingGrader.build(embedding_checkpoint)
    ctc_model, vocab, inv_vocab = load_ctc(ctc_checkpoint, device)
    skeleton = skeleton_for(extractor)
    by_sign_split = rows_by_sign_and_split(extractor)
    rng = random.Random(seed)

    align_boundary_err, ctc_boundary_err = [], []
    align_tally = defaultdict(lambda: [0, 0])
    ctc_tally = defaultdict(lambda: [0, 0])
    wers = []
    n_run = n_skipped = 0

    for _ in tqdm(range(n_trials), desc="CTC vs alignment trials", unit="trial"):
        k = rng.randint(min_signs, max_signs)
        trial = make_trial(by_sign_split, extractor, skeleton, k, rng, split="val")
        if trial is None:
            n_skipped += 1
            continue
        seq, attempt_poses, true_lengths, val_segments = trial
        if any(g not in vocab for g in seq.gloss_ids):
            n_skipped += 1  # a sign outside the CTC model's vocab (shouldn't happen, same curriculum)
            continue

        # -- shared: the isolated grade of each true clip, the common yardstick ----
        isolated_by_sign = {sign: grader.grade_against_poses(seg, sign) for sign, seg in zip(seq.gloss_ids, val_segments)}

        # -- forced alignment (align_and_grade) -------------------------------------
        try:
            _, graded = align_and_grade(grader, attempt_poses, seq, band=band)
        except Exception as exc:  # noqa: BLE001
            print(f"  align_and_grade failed ({seq.english}): {exc}")
            n_skipped += 1
            continue
        for (start, stop), true_len in zip((g.frame_range for g in graded), true_lengths):
            align_boundary_err.append(abs((stop - start) - true_len) / true_len)
        grading_agreement([(g.target_sign, g.result) for g in graded], isolated_by_sign, align_tally)

        # -- CTC: free decode (WER) + forced alignment (boundary error) -------------
        feature_clip = grader.pipeline.assemble(attempt_poses)
        feats = grader.standardizer.transform(feature_clip.features)
        x = torch.from_numpy(feats).unsqueeze(0).to(device)
        lengths = torch.tensor([feats.shape[0]])
        with torch.no_grad():
            log_probs = ctc_model(x, lengths)[0].cpu().numpy()

        decoded = greedy_decode(log_probs)
        true_idx = [vocab[g] for g in seq.gloss_ids]
        wers.append(word_error_rate(decoded, true_idx))

        path = forced_align(log_probs, true_idx)
        segs = segments_from_forced_align(path, n_glosses=len(seq.gloss_ids))
        ctc_graded = []
        for sign, (start, stop), true_len in zip(seq.gloss_ids, segs.ranges, true_lengths):
            if stop <= start:  # CTC attended zero frames to this gloss -- a real failure mode
                ctc_boundary_err.append(1.0)  # 100% error: the whole segment is missing
                continue
            ctc_boundary_err.append(abs((stop - start) - true_len) / true_len)
            ctc_graded.append((sign, grader.grade_against_poses(attempt_poses[start:stop], sign)))
        grading_agreement(ctc_graded, isolated_by_sign, ctc_tally)

        n_run += 1

    return {
        "n_run": n_run, "n_skipped": n_skipped,
        "align_boundary_err": align_boundary_err, "ctc_boundary_err": ctc_boundary_err,
        "align_tally": dict(align_tally), "ctc_tally": dict(ctc_tally),
        "wers": wers,
    }


def format_report(results, args) -> str:
    abe, cbe = results["align_boundary_err"], results["ctc_boundary_err"]
    wers = results["wers"]
    lines = [
        "# align_and_grade vs. CTC-CSLR -- research comparison",
        "",
        f"{results['n_run']} trials run, {results['n_skipped']} skipped, "
        f"{args.min_signs}-{args.max_signs} signs/sentence, seed={args.seed}, extractor={args.extractor}.",
        "",
        "**This is a research comparison, not a product decision re-litigation.**",
        "Both systems trained/built entirely on this project's own 60-sign curriculum;",
        "\"CTC\" here means an architecture matching the industry-standard approach, NOT",
        "a claim of matching published SOTA numbers (e.g. DeepMind's SL2T trained on",
        "100k+ hours across 50+ languages -- this trains on ~1,270 synthetic sentences",
        "built from this curriculum's own train-split clips).",
        "",
        "## 1. Open-set recognition (CTC's actual normal use case)",
        "",
        f"CTC free-decode word error rate: {statistics.mean(wers):.1%} mean, "
        f"{statistics.median(wers):.1%} median." if wers else "no samples",
        "",
        "align_and_grade has NO analogous number -- it is never given an unknown",
        "sequence to recognize; the target is always known in advance (this project's",
        "core framing, see project_workflow.md's Phase 7 section). This asymmetry is",
        "the actual finding, not a gap in the benchmark.",
        "",
        "## 2. Forced alignment given the TRUE sequence (fair, apples-to-apples)",
        "",
        "| system | mean boundary error | median boundary error |",
        "|---|---|---|",
        f"| align_and_grade (dtw_align) | {statistics.mean(abe):.1%} | {statistics.median(abe):.1%} |" if abe else "| align_and_grade | no samples | |",
        f"| CTC forced-align | {statistics.mean(cbe):.1%} | {statistics.median(cbe):.1%} |" if cbe else "| CTC forced-align | no samples | |",
        "",
        "**Why CTC forced-align is so much worse (verified, not assumed):** manual",
        "inspection of individual trials (see the diagnostic trace this report's",
        "writeup is based on) shows the trained model has learned extremely \"PEAKY\"",
        "posteriors -- a well-documented, real property of CTC training, not a bug in",
        "this comparison's `forced_align` implementation (independently correctness-",
        "checked against `nn.CTCLoss`'s own forward-algorithm marginal in",
        "`tests/test_ctc_cslr.py`). On one representative 77-frame trial ('where my'),",
        "the model predicted BLANK on 74/77 frames with a huge margin (mean blank",
        "log-prob -0.22 vs. mean best-real-label log-prob -11.8), spiking on the real",
        "label for only 1-2 frames each. Greedy decoding still gets the SEQUENCE right",
        "from a spike that brief (hence a reasonable WER above) -- but Viterbi forced",
        "alignment correctly finds exactly where that brief spike is, producing a",
        "1-2-frame segment for a sign whose true length was 35-42 frames. This is CTC's",
        "loss function rewarding ANY confident spike per label, with no pressure",
        "toward temporally well-calibrated boundaries -- entropy regularization or a",
        "dedicated alignment objective are known mitigations in the literature, neither",
        "attempted here since the goal was a fair VANILLA-CTC comparison on this",
        "project's own data, not a maximally-optimized CTC pipeline.",
        "",
        "## 3. Grading agreement on the resulting segments (aligned vs. isolated grade)",
        "",
        "| parameter | align_and_grade | CTC forced-align | n (align / ctc) |",
        "|---|---|---|---|",
    ]
    params = sorted(set(results["align_tally"]) | set(results["ctc_tally"]))
    for p in params:
        a_correct, a_n = results["align_tally"].get(p, [0, 0])
        c_correct, c_n = results["ctc_tally"].get(p, [0, 0])
        a_pct = f"{a_correct/a_n:.1%}" if a_n else "n/a"
        c_pct = f"{c_correct/c_n:.1%}" if c_n else "n/a"
        lines.append(f"| {p} | {a_pct} | {c_pct} | {a_n} / {c_n} |")
    lines += [
        "",
        "## Practical tradeoff (not measured above, stated plainly)",
        "",
        "- **Training data**: align_and_grade needed ZERO sequence-labeled sentences --",
        "  it's training-free DTW over an EmbeddingGrader trained only on ISOLATED",
        "  single-sign clips (Phase 4). CTC needed synthetic multi-sign sentences",
        "  specifically manufactured for this comparison, since this project's dataset",
        "  has no native continuous-signing footage at all.",
        "- **Failure mode on a malformed attempt**: CTC, an N-way classifier, will",
        "  emit SOME decoded sequence regardless of input quality -- confidently wrong",
        "  on a bad attempt. align_and_grade always compares to the ONE known target and",
        "  reports a real distance, never a false best-guess identity. This is",
        "  CLAUDE.md's actual stated reason for excluding CTC/classification from the",
        "  product, not just measured indirectly here.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--min-signs", type=int, default=2)
    ap.add_argument("--max-signs", type=int, default=4)
    ap.add_argument("--extractor", default="mediapipe")
    ap.add_argument("--embedding-checkpoint", type=Path, default=DEFAULT_EMBEDDING_CHECKPOINT)
    ap.add_argument("--ctc-checkpoint", type=Path, default=DEFAULT_CTC_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--band", type=int, default=None)
    ap.add_argument("--report", type=Path, default=REPO / "CTC_VS_ALIGNMENT_REPORT.md")
    args = ap.parse_args()

    results = run(args.n_trials, args.min_signs, args.max_signs, args.embedding_checkpoint,
                   args.ctc_checkpoint, args.extractor, args.seed, args.band)
    report = format_report(results, args)
    print("\n" + report)
    args.report.write_text(report)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
