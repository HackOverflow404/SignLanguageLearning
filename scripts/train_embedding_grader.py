#!/usr/bin/env python3
"""Train the Phase 4 learned grader: a metric-learned embedding + phonological heads.

Trains on the FULL 60-sign curriculum (not the 20-sign Phase 2 slice) -- per-parameter
label coverage is far better at 60 signs (e.g. `repeated_movement` is 30/30 balanced
vs. 16/4 skewed in the 20-slice; see project_workflow.md's Phase 4 section for the
full justification), and this is what "did it work" means for this phase.

    .venv/bin/python scripts/train_embedding_grader.py
    .venv/bin/python scripts/train_embedding_grader.py --epochs 5   # smoke test

Reports TRAIN and VAL metrics every epoch -- overfitting (learning signers, not signs)
is the main risk with ~15 clips/sign, so a growing train/val gap is a red flag to
surface, never to bury.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from aslcv import dataset as dataset_mod
from aslcv.grading.embedding_dataset import (
    EmbeddingClipDataset,
    PKBatchSampler,
    collate_fn,
    fit_standardizer,
)
from aslcv.grading.embedding_grader import embed_dataset, skeleton_for
from aslcv.grading.embedding_model import PoseGraderNet, batch_hard_triplet_loss
from aslcv.grading.phonology_labels import CATEGORICAL_PARAMETERS, PhonologyLabels
from aslcv.pipeline_config import add_pipeline_args, build_pipeline

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "models" / "embedding_grader"


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def topk_accuracy(sign_dist, true_idx, k):
    topk = sign_dist.topk(min(k, sign_dist.shape[1]), dim=1, largest=False).indices
    hit = (topk == true_idx.unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


def head_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def evaluate_split(model, ds, device, n_signs, ref=None, k=5):
    """Nearest-clip ranking (same agg="min" convention as DTWGrader) against a
    reference bank.

    `ref=None` means "rank against yourself" -- used for the TRAIN split, where
    ranking against the very embeddings being scored would let every query
    self-match at distance 0 (trivial ~100% top-1, not a real signal). The diagonal
    is excluded so this becomes a leave-one-out nearest-OTHER-clip ranking instead,
    making train and val top-1/top-5 genuinely comparable -- the gap between them is
    the overfitting signal this script exists to surface.
    """
    out = embed_dataset(model, ds, device)
    if ref is None:
        d = torch.cdist(out["embeds"], out["embeds"])
        d.fill_diagonal_(float("inf"))
        ref_gloss_idx = out["gloss_idx"]
    else:
        d = torch.cdist(out["embeds"], ref["embeds"])
        ref_gloss_idx = ref["gloss_idx"]

    sign_dist = torch.full((d.shape[0], n_signs), float("inf"))
    for s in range(n_signs):
        mask = ref_gloss_idx == s
        if mask.any():
            sign_dist[:, s] = d[:, mask].min(dim=1).values

    metrics = {
        "top1": topk_accuracy(sign_dist, out["gloss_idx"], 1),
        "top5": topk_accuracy(sign_dist, out["gloss_idx"], k),
    }
    for p in CATEGORICAL_PARAMETERS:
        metrics[p] = head_accuracy(out["logits"][p], out["labels"][p])
    metrics["repeated_movement"] = ((out["repeated_logits"] > 0).float() == out["repeated_labels"]).float().mean().item()
    return metrics, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="mediapipe", choices=["mediapipe", "dwpose", "rtmw", "vitpose"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--p", type=int, default=15, help="signs per PK batch")
    ap.add_argument("--k", type=int, default=4, help="clips per sign per PK batch")
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.2, help="triplet loss margin")
    ap.add_argument("--head-loss-weight", type=float, default=0.5,
                    help="weight applied to the SUM of the 5 head losses vs. the triplet loss (weight 1.0)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_pipeline_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    skeleton = skeleton_for(args.extractor)
    pipeline = build_pipeline(args, skeleton, extractor_name=args.extractor)

    phon = PhonologyLabels()
    signs = sorted(phon.by_gloss)
    print(f"training vocabulary: {len(signs)} signs (the full curriculum, not the 20-sign Phase 2 slice)")

    id_gloss_encoder = dataset_mod.LabelEncoder(signs)

    print("fitting standardizer on train split ...")
    t0 = time.time()
    standardizer = fit_standardizer(pipeline, args.extractor, signs)
    print(f"  done in {time.time() - t0:.1f}s")

    print("building datasets ...")
    t0 = time.time()
    train_ds = EmbeddingClipDataset(args.extractor, "train", pipeline, signs, id_gloss_encoder, phon, standardizer)
    val_ds = EmbeddingClipDataset(args.extractor, "val", pipeline, signs, id_gloss_encoder, phon, standardizer)
    print(f"  train={len(train_ds)} val={len(val_ds)} clips, in {time.time() - t0:.1f}s")

    head_classes = {p: len(phon.encoders[p]) for p in CATEGORICAL_PARAMETERS}
    print(f"head classes: {head_classes}")

    model = PoseGraderNet(train_ds.blocks, head_classes, hidden=args.hidden, embed_dim=args.embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler = PKBatchSampler(train_ds.id_gloss_indices, p=args.p, k=args.k,
                              batches_per_epoch=args.batches_per_epoch, seed=args.seed)
    loader = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)

    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()

    history = []
    best_val_top1 = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"triplet": 0.0, "heads": 0.0}
        n_batches = 0
        for batch in loader:
            batch = _to_device(batch, device)
            out = model(batch["features"], batch["lengths"], batch["tempo"])

            triplet = batch_hard_triplet_loss(out["embed"], batch["id_gloss"], margin=args.margin)
            head_loss = (
                ce(out["handshape"], batch["handshape"])
                + ce(out["major_location"], batch["major_location"])
                + ce(out["minor_location"], batch["minor_location"])
                + ce(out["movement"], batch["movement"])
                + bce(out["repeated"], batch["repeated"])
            )
            loss = triplet + args.head_loss_weight * head_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["triplet"] += triplet.item()
            epoch_losses["heads"] += head_loss.item()
            n_batches += 1

        # -- eval: leave-one-out on train, then val ranked against the full train bank --
        train_metrics, train_ref = evaluate_split(model, train_ds, device, len(signs))
        val_metrics, _ = evaluate_split(model, val_ds, device, len(signs), ref=train_ref)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics,
                         "loss": {k: v / n_batches for k, v in epoch_losses.items()}})

        gap = train_metrics["top1"] - val_metrics["top1"]
        print(
            f"epoch {epoch:3d}  loss(triplet={epoch_losses['triplet']/n_batches:.3f} "
            f"heads={epoch_losses['heads']/n_batches:.3f})  "
            f"top1 train={train_metrics['top1']:.1%} val={val_metrics['top1']:.1%} (gap={gap:+.1%})  "
            f"top5 train={train_metrics['top5']:.1%} val={val_metrics['top5']:.1%}  "
            f"heads(train/val) hs={train_metrics['handshape']:.0%}/{val_metrics['handshape']:.0%} "
            f"majloc={train_metrics['major_location']:.0%}/{val_metrics['major_location']:.0%} "
            f"minloc={train_metrics['minor_location']:.0%}/{val_metrics['minor_location']:.0%} "
            f"move={train_metrics['movement']:.0%}/{val_metrics['movement']:.0%} "
            f"rep={train_metrics['repeated_movement']:.0%}/{val_metrics['repeated_movement']:.0%}"
        )

        # Track the BEST-VAL checkpoint separately from the final epoch. Small data
        # (~15 clips/sign) means overfitting -- learning signers, not signs -- is the
        # main risk here, and the train/val gap above (already visible by epoch 6 in
        # smoke testing) confirms it's real, not hypothetical. Shipping whatever the
        # LAST epoch happens to be would silently ship an overfit model; best-val is
        # what EmbeddingGrader / the eval script should load by default.
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out / "model_best.pt")

    # -- save final-epoch checkpoint + shared config -------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model_final.pt")
    standardizer.save(args.out / "standardizer.npz")
    config = {
        "extractor": args.extractor,
        "blocks": {k: [v.start, v.stop] for k, v in train_ds.blocks.items()},
        "head_classes": head_classes,
        "hidden": args.hidden,
        "embed_dim": args.embed_dim,
        "id_gloss_classes": id_gloss_encoder.classes_,
        "phonology_classes": {p: phon.encoders[p].classes_ for p in CATEGORICAL_PARAMETERS},
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "final_epoch": args.epochs,
        "pipeline_args": {
            "face": args.face, "legs_feet": args.legs_feet, "confidence": args.confidence,
            "binary_threshold": args.binary_threshold, "velocity": args.velocity,
            "depth_proxies": args.depth_proxies, "trim_to_motion": args.trim_to_motion,
            "motion_threshold": args.motion_threshold, "motion_pad_frames": args.motion_pad_frames,
            "local_hand": args.local_hand,
        },
    }
    with open(args.out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(args.out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    final_gap = history[-1]["train"]["top1"] - history[-1]["val"]["top1"]
    print(f"\nbest val top-1: {best_val_top1:.1%} at epoch {best_epoch} (of {args.epochs}) -> model_best.pt")
    print(f"final epoch {args.epochs}: train/val top-1 gap = {final_gap:+.1%} -> model_final.pt")
    print(f"saved checkpoints + config + standardizer to {args.out}")


if __name__ == "__main__":
    main()
