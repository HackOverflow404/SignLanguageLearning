"""Phase 8: what the phone needs to grade a sentence, plus golden fixtures.

Writes models/mobile/sentence_refs.f32 + sentence_refs.json: for every bundle
sign, the standardized feature sequence of its reference clip, exactly as
compose_reference_features builds each piece (the same fetch_reference clip
the app plays, trimmed to the same motion-active span). Composing a sentence's
reference on the phone is concatenating these per gloss -- the same thing
compose_reference_features does, checked below on a multi-sign sequence.

Also writes models/mobile/fixtures/alignment.json: synthetic continuous
attempts (raw val clips of 2-4 signs back to back, face zeroed as the phone
has no face mesh) run through the real align_and_grade, recording the
alignment distance, per-gloss frame ranges before and after each segment's
live trim, and each segment's ONNX outputs + verdicts (BundleGrader, the
runtime the phone uses).

    .venv/bin/python scripts/export_sentence_references.py
"""
import argparse
import base64
import json
import random
from pathlib import Path

import numpy as np
import torch

from aslcv import dataset as dataset_mod
from aslcv.extractor.base import Pose
from aslcv.features import hand_motion_energy
from aslcv.grading.alignment import _segment_ranges, align_and_grade
from aslcv.grading.bundle_grader import BundleGrader
from aslcv.grading.dtw_grader import dtw_align
from aslcv.grading.embedding_dataset import _tempo_features
from aslcv.grading.embedding_grader import EmbeddingGrader
from aslcv.production.gloss_rules import Gloss, GlossSequence
from aslcv.production.retrieval import compose_reference_features

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "models" / "embedding_grader"
DEFAULT_BUNDLE = REPO / "models" / "mobile"


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def sequence(signs) -> GlossSequence:
    """A minimal in-scope GlossSequence over known curriculum signs."""
    return GlossSequence(english=" ".join(signs), in_scope=True, confidence=1.0, reason=None,
                         sentence_type="statement", negated=False,
                         glosses=[Gloss(text=s.upper(), asllex_id=s, source=s, pos="NOUN") for s in signs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--sentences", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"  # CPU GRU is unusable here (CLAUDE.md, Testing)
    grader = EmbeddingGrader.build(DEFAULT_CHECKPOINT, which="best", device=device)
    meta = json.loads((args.bundle / "bundle.json").read_text())
    signs = sorted(meta["signs"])

    # -- per-sign reference features ------------------------------------------
    pieces, index, start = [], [], 0
    for sign in signs:
        feats = compose_reference_features(sequence([sign]), grader.extractor, grader.pipeline,
                                           grader.standardizer).features.astype("<f4")
        pieces.append(feats)
        index.append({"sign": sign, "start": start, "count": int(len(feats))})
        start += len(feats)
    refs = np.concatenate(pieces)
    by_sign = {e["sign"]: refs[e["start"]:e["start"] + e["count"]] for e in index}
    check = ["me", "want_2", "water"]
    composed = compose_reference_features(sequence(check), grader.extractor, grader.pipeline, grader.standardizer)
    assert np.array_equal(composed.features.astype("<f4"), np.concatenate([by_sign[s] for s in check]))
    refs.tofile(args.bundle / "sentence_refs.f32")
    (args.bundle / "sentence_refs.json").write_text(json.dumps(
        {"file": "sentence_refs.f32", "rows": int(len(refs)), "dim": int(refs.shape[1]), "signs": index}, indent=1))
    print(f"wrote sentence_refs.f32: {len(refs)} frames x {refs.shape[1]} ({refs.nbytes / 1e6:.1f} MB)")

    # -- alignment fixtures ----------------------------------------------------
    sk = grader.pipeline.skeleton
    face = np.array(sk.region("face"))
    val = {}
    for r in dataset_mod.load_index(grader.extractor, "val"):
        if r["id_gloss"] in meta["signs"]:
            val.setdefault(r["id_gloss"], r)
    bundle = BundleGrader(args.bundle)
    rng = random.Random(args.seed)
    sentences = []
    for _ in range(args.sentences):
        chosen = rng.sample(sorted(val), rng.randint(2, 4))
        kps, scs = [], []
        for s in chosen:
            with np.load(REPO / "data" / "cache" / grader.extractor / f"{val[s]['video_id']}.npz") as d:
                kps.append(d["keypoints"].astype(np.float32))
                scs.append(d["scores"].astype(np.float32))
        kp, sc = np.concatenate(kps), np.concatenate(scs)
        kp[:, face] = 0.0
        sc[:, face] = 0.0
        poses = [Pose(k, s) for k, s in zip(kp, sc)]
        seq = sequence(chosen)

        distance, graded = align_and_grade(grader, poses, seq)
        attempt = grader.standardizer.transform(grader.pipeline.assemble(poses).features)
        reference = np.concatenate([by_sign[s] for s in chosen])
        gloss_index = np.concatenate([np.full(len(by_sign[s]), i) for i, s in enumerate(chosen)])
        d2, path = dtw_align(attempt, reference)
        assert abs(d2 - distance) < 1e-9
        raw_ranges = _segment_ranges(path, gloss_index, len(chosen))

        segments = []
        for g in graded:
            a, b = g.frame_range
            seg = poses[a:b]
            raw = grader.pipeline.assemble(seg).features
            tempo = _tempo_features(hand_motion_energy(grader.pipeline.normalizer, sk, seg))
            outputs = bundle.run(raw, tempo)
            bg = bundle.grade(raw, tempo, g.target_sign)
            segments.append({
                "sign": g.target_sign, "range": [int(a), int(b)],
                "outputs": {k: np.atleast_1d(v).astype(float).tolist() for k, v in outputs.items()},
                "fidelity": bg.fidelity,
                "parameters": {p: {"predicted": v.predicted, "target": v.target, "correct": v.correct,
                                   "confidence": v.confidence} for p, v in bg.parameters.items()}})
        sentences.append({
            "signs": chosen,
            "input": {"frames": int(len(kp)), "points": int(kp.shape[1]), "keypoints": _b64(kp), "scores": _b64(sc)},
            "expected": {"distance": distance, "path_length": len(path),
                         "raw_ranges": [list(map(int, r)) for r in raw_ranges], "segments": segments}})
        print(f"  {' '.join(chosen):<40} {len(kp)} frames -> "
              + ", ".join(f"{s['sign']} {s['range']}" for s in segments))
    out = args.bundle / "fixtures" / "alignment.json"
    out.write_text(json.dumps({"sentences": sentences}))
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
