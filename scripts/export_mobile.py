"""Phase 8: export the trained EmbeddingGrader to a self-contained mobile bundle
(ONNX graph + plain data files) and verify it numerically against PyTorch.

Writes models/mobile/ (see aslcv.grading.bundle_grader for the format):
  grader.onnx, references.f32, bundle.json

Also writes models/mobile/fixtures/ (--fixtures N, 0 to skip): golden files for
the TypeScript port -- skeleton.json (names/anchors/regions, so the port looks
points up by meaning, never by hardcoded index) and one JSON per val clip with
the raw keypoints in and every intermediate out (live_capture_span trim, raw
features, motion energy, tempo, model outputs, verdicts). Float arrays are
base64 little-endian float32, so the port compares exact values, not decimal
renderings. Face-mesh points are zeroed in the inputs: the phone runs only
MediaPipe's pose + hand models, and the features never read the face region.

Verification, on real cached val clips (live-trimmed, exactly as grading sees them):
  1. InferenceGraderNet (unpadded batch-1 forward) vs PoseGraderNet (packed
     forward) in torch -- the export wrapper must be the same function;
  2. ONNX Runtime on CPU (what a phone runs) vs torch -- every output.
Max absolute differences are printed; the script exits non-zero past --atol.

    .venv/bin/python scripts/export_mobile.py
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

from aslcv import dataset as dataset_mod
from aslcv.extractor.base import Pose
from aslcv.features import hand_motion_energy, live_capture_span
from aslcv.grading.bundle_grader import FORMAT_VERSION, BundleGrader
from aslcv.grading.embedding_dataset import LIVE_PREROLL, LIVE_SETTLE_FRAMES, _live_trimmed_poses, _tempo_features
from aslcv.grading.embedding_grader import EmbeddingGrader
from aslcv.grading.inference_net import OUTPUT_NAMES, InferenceGraderNet
from aslcv.grading.phonology_labels import CATEGORICAL_PARAMETERS, MIN_SUPPORT

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "models" / "embedding_grader"
DEFAULT_OUT = REPO / "models" / "mobile"


def model_inputs(grader, poses):
    """(raw features (T,F), standardized features (T,F), tempo (2,)) -- the same
    computation as EmbeddingGrader._forward_poses."""
    clip = grader.pipeline.assemble(poses)
    energy = hand_motion_energy(grader.pipeline.normalizer, grader.pipeline.skeleton, poses)
    return clip.features, grader.standardizer.transform(clip.features), _tempo_features(energy)


def build_meta(grader, config, ref_signs):
    pl = grader.phon_labels
    signs = {}
    for sign in grader.signs:
        entry = {}
        for p in CATEGORICAL_PARAMETERS:
            target = pl.label_for(sign, p)
            entry[p] = {"target": target, "judged": pl.support(p, target) >= MIN_SUPPORT}
        rep = pl.repeated_bool(sign)
        entry["repeated_movement"] = {
            "target": rep, "judged": pl.support("repeated_movement", "1" if rep else "0") >= MIN_SUPPORT}
        signs[sign] = entry
    return {
        "format_version": FORMAT_VERSION,
        "model": "grader.onnx",
        "outputs": list(OUTPUT_NAMES),
        "extractor": config["extractor"],
        "pipeline_args": config["pipeline_args"],
        "blocks": config["blocks"],
        "feature_dim": int(grader.standardizer.mean.shape[0]),
        "embed_dim": config["embed_dim"],
        "capture": {"motion_threshold": grader.pipeline.motion_threshold,
                    "live_preroll": LIVE_PREROLL, "live_settle_frames": LIVE_SETTLE_FRAMES},
        "standardizer": {"mean": grader.standardizer.mean.tolist(), "std": grader.standardizer.std.tolist()},
        "classes": {p: list(config["phonology_classes"][p]) for p in CATEGORICAL_PARAMETERS},
        "min_support": MIN_SUPPORT,
        "references": {"file": "references.f32", "rows": None, "dim": config["embed_dim"], "signs": ref_signs},
        "signs": signs,
    }


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def write_fixtures(grader, bundle_dir, rows, out_dir):
    """Golden input -> every intermediate -> verdicts, for cross-language parity tests.
    Expected model outputs/verdicts come from BundleGrader (ONNX Runtime on CPU) --
    the runtime the phone actually uses -- not from torch."""
    sk = grader.pipeline.skeleton
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skeleton.json").write_text(json.dumps({
        "names": list(sk.names), "anchors": dict(sk.anchors),
        "regions": {name: list(idx) for name, idx in sk.regions}}, indent=1))
    face = np.array(sk.region("face"))
    bundle = BundleGrader(bundle_dir)
    signs = grader.signs
    for i, r in enumerate(rows):
        with np.load(REPO / "data" / "cache" / grader.extractor / f"{r['video_id']}.npz") as d:
            kp, sc = d["keypoints"].astype(np.float32), d["scores"].astype(np.float32)
        kp[:, face] = 0.0
        sc[:, face] = 0.0
        poses = [Pose(k, s) for k, s in zip(kp, sc)]
        full_energy = hand_motion_energy(grader.pipeline.normalizer, sk, poses)
        start, stop = live_capture_span(full_energy, grader.pipeline.motion_threshold, LIVE_PREROLL, LIVE_SETTLE_FRAMES)
        trimmed = poses[start:stop]
        raw, _, tempo = model_inputs(grader, trimmed)
        energy = hand_motion_energy(grader.pipeline.normalizer, sk, trimmed)
        outputs = bundle.run(raw, tempo)
        grades = {}
        for target in (r["id_gloss"], signs[(signs.index(r["id_gloss"]) + 7) % len(signs)]):
            g = bundle.grade(raw, tempo, target)
            grades[target] = {"fidelity": g.fidelity, "parameters": {
                p: {"predicted": v.predicted, "target": v.target, "correct": v.correct, "confidence": v.confidence}
                for p, v in g.parameters.items()}}
        fixture = {
            "video_id": r["video_id"], "sign": r["id_gloss"],
            "input": {"frames": int(len(kp)), "points": int(kp.shape[1]),
                      "keypoints": _b64(kp), "scores": _b64(sc)},
            "expected": {
                "full_energy": _b64(full_energy), "live_capture_span": [int(start), int(stop)],
                "features": _b64(raw), "feature_shape": list(raw.shape), "energy": _b64(energy),
                "tempo": [float(x) for x in tempo],
                "outputs": {k: np.atleast_1d(v).astype(float).tolist() for k, v in outputs.items()},
                "grades": grades}}
        (out_dir / f"clip_{i:02d}_{r['id_gloss']}.json").write_text(json.dumps(fixture))
    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"wrote {len(rows)} golden fixtures + skeleton.json -> {out_dir.relative_to(REPO)}/ ({size:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-clips", type=int, default=40, help="val clips used for numerical verification")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--fixtures", type=int, default=8, help="golden fixture clips for the TS port (0 = skip)")
    args = ap.parse_args()

    # CUDA when available: torch's CPU GRU kernel is unusable in this environment
    # (see CLAUDE.md's Testing note); the ONNX graph itself is device-independent.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Full-FP32 reference: cuDNN's default TF32 RNN/matmul math differs from exact
    # FP32 by ~1e-3 on the logits, which would be blamed on the export otherwise.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    grader = EmbeddingGrader.build(args.checkpoint, which=args.which, device=device)
    config = json.loads((args.checkpoint / "config.json").read_text())
    net = InferenceGraderNet(grader.model).eval()

    args.out.mkdir(parents=True, exist_ok=True)

    # -- reference embeddings, grouped by sign --------------------------------------
    rows, ref_signs, start = [], [], 0
    for sign in grader.signs:
        emb = grader.references[sign].cpu().numpy().astype("<f4")
        ref_signs.append({"sign": sign, "start": start, "count": int(len(emb))})
        rows.append(emb)
        start += len(emb)
    refs = np.concatenate(rows)
    refs.tofile(args.out / "references.f32")
    meta = build_meta(grader, config, ref_signs)
    meta["references"]["rows"] = int(len(refs))

    # -- ONNX graph -----------------------------------------------------------------
    T = 48
    dummy = (torch.zeros(1, T, meta["feature_dim"], device=device), torch.zeros(1, 2, device=device))
    torch.onnx.export(
        net, dummy, str(args.out / "grader.onnx"),
        input_names=["features", "tempo"], output_names=list(OUTPUT_NAMES),
        dynamic_axes={"features": {1: "frames"}}, opset_version=17, dynamo=False)
    (args.out / "bundle.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) / 1e6
    print(f"wrote {args.out.relative_to(REPO)}/ ({size:.1f} MB): grader.onnx, references.f32 "
          f"({len(refs)} x {meta['embed_dim']}), bundle.json")

    # -- verification -----------------------------------------------------------------
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out / "grader.onnx"), providers=["CPUExecutionProvider"])
    all_val = dataset_mod.load_index(grader.extractor, "val")
    val = all_val[::max(1, len(all_val) // args.n_clips)][:args.n_clips]
    wrap_err = {k: 0.0 for k in OUTPUT_NAMES}
    onnx_err = {k: 0.0 for k in OUTPUT_NAMES}
    lengths = []
    with torch.no_grad():
        for r in val:
            poses = _live_trimmed_poses(REPO / "data" / "cache" / grader.extractor / f"{r['video_id']}.npz",
                                        grader.pipeline)
            _, feats, tempo = model_inputs(grader, poses)
            lengths.append(len(feats))
            f_t = torch.from_numpy(feats)[None].to(device)
            t_t = torch.from_numpy(tempo)[None].to(device)
            ref = grader.model(f_t, torch.tensor([len(feats)], device=device), t_t)
            wrapped = dict(zip(OUTPUT_NAMES, net(f_t, t_t)))
            onnx_out = dict(zip(OUTPUT_NAMES, sess.run(None, {"features": feats[None], "tempo": tempo[None]})))
            for k in OUTPUT_NAMES:
                a = ref[k].cpu().numpy()
                wrap_err[k] = max(wrap_err[k], float(np.abs(a - wrapped[k].cpu().numpy()).max()))
                onnx_err[k] = max(onnx_err[k], float(np.abs(a - onnx_out[k]).max()))

    print(f"verified on {len(val)} val clips (T = {min(lengths)}..{max(lengths)} frames)")
    print(f"  {'output':<16}{'wrapper vs torch':>18}{'onnx(cpu) vs torch':>20}")
    for k in OUTPUT_NAMES:
        print(f"  {k:<16}{wrap_err[k]:>18.2e}{onnx_err[k]:>20.2e}")
    worst = max(max(wrap_err.values()), max(onnx_err.values()))
    if args.fixtures and worst <= args.atol:
        write_fixtures(grader, args.out, val[::max(1, len(val) // args.fixtures)][:args.fixtures],
                       args.out / "fixtures")
    print(f"max abs diff {worst:.2e} (atol {args.atol:g}) -> {'PASS' if worst <= args.atol else 'FAIL'}")
    sys.exit(0 if worst <= args.atol else 1)


if __name__ == "__main__":
    main()
