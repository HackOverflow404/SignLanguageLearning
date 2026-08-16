#!/usr/bin/env python3
"""Train the CTC-CSLR research comparison model (aslcv.research.ctc_cslr).

ISOLATED RESEARCH COMPARISON, not part of the shipped product -- see
src/aslcv/research/__init__.py's module docstring for why. Trains a
BiGRU+CTC model on SYNTHETIC continuous sentences built by concatenating
this project's own TRAIN-split isolated-sign clips
(aslcv.research.synthetic_sentences.make_trial -- the same trick Phase 7's
eval_forced_alignment.py uses at benchmark scale). CTC needs sequence-
labeled CONTINUOUS training data, which this project's dataset (isolated
single-sign clips) doesn't natively have; synthetic concatenation is the
only way to train it on this project's own data at all.

Uses the SAME FeaturePipeline/Standardizer the already-trained EmbeddingGrader
checkpoint uses (loaded from --checkpoint only for its pipeline/standardizer
-- its model weights are irrelevant here) so the eventual comparison in
scripts/eval_ctc_vs_alignment.py isolates the APPROACH, not a difference in
feature engineering.

Reports TRAIN and VAL word error rate (WER) every epoch, same "never hide
the gap" discipline as scripts/train_embedding_grader.py.

    .venv/bin/python scripts/train_ctc_cslr.py
    .venv/bin/python scripts/train_ctc_cslr.py --epochs 5   # smoke test
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from aslcv.grading.embedding_grader import EmbeddingGrader
from aslcv.production.retrieval import skeleton_for
from aslcv.research.ctc_cslr import BLANK, CTCEncoder, greedy_decode, word_error_rate
from aslcv.research.synthetic_sentences import make_trial, rows_by_sign_and_split

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "models" / "embedding_grader"
DEFAULT_OUT = REPO / "models" / "ctc_cslr"


def build_vocab(signs) -> dict:
    return {s: i + 1 for i, s in enumerate(sorted(signs))}  # 0 reserved for blank


def featurize(pipeline, standardizer, poses):
    clip = pipeline.assemble(poses)
    return standardizer.transform(clip.features)


def generate_sentences(n, by_sign_split, extractor, skeleton, min_signs, max_signs, split, rng,
                        pipeline, standardizer):
    """N synthetic sentences, pre-featurized ONCE here (not per-epoch) --
    (GlossSequence, (T, F) feature tensor) pairs. `attempts` cap avoids an
    infinite loop if the sign pool genuinely can't supply enough distinct
    trials."""
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 5:
        attempts += 1
        k = rng.randint(min_signs, max_signs)
        trial = make_trial(by_sign_split, extractor, skeleton, k, rng, split=split)
        if trial is None:
            continue
        seq, poses, _, _ = trial
        feats = torch.from_numpy(featurize(pipeline, standardizer, poses))
        out.append((seq, feats))
    return out


def make_batch(batch, vocab, device):
    feats = [f for _, f in batch]
    lengths = torch.tensor([f.shape[0] for f in feats])
    padded = nn.utils.rnn.pad_sequence(feats, batch_first=True).to(device)
    targets = torch.cat([torch.tensor([vocab[g] for g in seq.gloss_ids]) for seq, _ in batch])
    target_lengths = torch.tensor([len(seq.gloss_ids) for seq, _ in batch])
    return padded, lengths, targets, target_lengths


@torch.no_grad()
def evaluate(model, sentences, vocab, device) -> float:
    model.eval()
    total_wer, n = 0.0, 0
    for seq, feats in sentences:
        x = feats.unsqueeze(0).to(device)
        lengths = torch.tensor([feats.shape[0]])
        log_probs = model(x, lengths)[0].cpu().numpy()
        decoded = greedy_decode(log_probs)
        true = [vocab[g] for g in seq.gloss_ids]
        total_wer += word_error_rate(decoded, true)
        n += 1
    model.train()
    return total_wer / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                     help="EmbeddingGrader checkpoint to borrow its pipeline/standardizer from")
    ap.add_argument("--extractor", default="mediapipe")
    ap.add_argument("--n-train-sentences", type=int, default=800)
    ap.add_argument("--n-val-sentences", type=int, default=150)
    ap.add_argument("--min-signs", type=int, default=2)
    ap.add_argument("--max-signs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    rng = random.Random(args.seed)

    grader = EmbeddingGrader.build(args.checkpoint)  # pipeline/standardizer only -- its model is unused
    pipeline, standardizer = grader.pipeline, grader.standardizer
    skeleton = skeleton_for(args.extractor)
    by_sign_split = rows_by_sign_and_split(args.extractor)
    signs = sorted(s for s, splits in by_sign_split.items() if splits.get("train"))
    vocab = build_vocab(signs)
    print(f"vocab: {len(vocab)} signs")

    print(f"generating + featurizing {args.n_train_sentences} synthetic train sentences "
          f"({args.min_signs}-{args.max_signs} signs each, TRAIN-split clips) ...")
    train_sentences = generate_sentences(args.n_train_sentences, by_sign_split, args.extractor, skeleton,
                                          args.min_signs, args.max_signs, "train", rng, pipeline, standardizer)
    print(f"generating + featurizing {args.n_val_sentences} synthetic val sentences (VAL-split clips) ...")
    val_sentences = generate_sentences(args.n_val_sentences, by_sign_split, args.extractor, skeleton,
                                        args.min_signs, args.max_signs, "val", rng, pipeline, standardizer)
    print(f"  {len(train_sentences)} train / {len(val_sentences)} val sentences built")

    in_dim = train_sentences[0][1].shape[1]
    model = CTCEncoder(in_dim, len(vocab), hidden=args.hidden, layers=args.layers).to(device)
    ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_sentences)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(train_sentences), args.batch_size):
            batch = train_sentences[i:i + args.batch_size]
            padded, lengths, targets, target_lengths = make_batch(batch, vocab, device)
            log_probs = model(padded, lengths).transpose(0, 1)  # CTCLoss wants (T, B, C)
            loss = ctc_loss(log_probs, targets, lengths, target_lengths)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_wer = evaluate(model, train_sentences[:100], vocab, device)
        val_wer = evaluate(model, val_sentences, vocab, device)
        history.append({"epoch": epoch, "loss": epoch_loss / n_batches,
                         "train_wer": train_wer, "val_wer": val_wer})
        print(f"epoch {epoch:3d}/{args.epochs}  loss={epoch_loss / n_batches:.3f}  "
              f"train_wer={train_wer:.1%}  val_wer={val_wer:.1%}  ({time.time() - t0:.0f}s elapsed)")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    config = {
        "in_dim": in_dim, "vocab": vocab, "hidden": args.hidden, "layers": args.layers,
        "extractor": args.extractor, "checkpoint_used_for_pipeline": str(args.checkpoint),
        "min_signs": args.min_signs, "max_signs": args.max_signs,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
