"""Phase 8 mobile bundle: InferenceGraderNet is the same function as
PoseGraderNet at batch size 1, and BundleGrader -- grading from the exported
bundle files + ONNX Runtime alone -- reproduces EmbeddingGrader's verdicts.

Needs the trained checkpoint and an exported bundle (models/mobile/, written by
scripts/export_mobile.py); SKIPPED if either is absent, same pattern as
tests/test_embedding_grader.py. CUDA when available for the torch side -- see
CLAUDE.md's Testing note on the CPU GRU crash.

Runs under pytest OR as a plain script.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO / "models" / "embedding_grader"
BUNDLE_DIR = REPO / "models" / "mobile"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model_best.pt").exists() or not (BUNDLE_DIR / "grader.onnx").exists(),
    reason="needs models/embedding_grader and models/mobile -- run scripts/export_mobile.py")


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def grader():
    from aslcv.grading.embedding_grader import EmbeddingGrader
    # exact FP32 on the torch side: cuDNN's default TF32 drifts ~1e-3 from what
    # ONNX Runtime computes. Restored afterwards so other test modules are unaffected.
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield EmbeddingGrader.build(CHECKPOINT_DIR, which="best", device=_device())
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


@pytest.fixture(scope="module")
def bundle():
    from aslcv.grading.bundle_grader import BundleGrader
    return BundleGrader(BUNDLE_DIR)


def _val_cases(grader, n=24):
    from aslcv import dataset as dataset_mod
    from aslcv.grading.embedding_dataset import _live_trimmed_poses
    rows = dataset_mod.load_index(grader.extractor, "val")[::9][:n]
    for r in rows:
        npz = REPO / "data" / "cache" / grader.extractor / f"{r['video_id']}.npz"
        yield r["id_gloss"], _live_trimmed_poses(npz, grader.pipeline)


def _inputs(grader, poses):
    from aslcv.features import hand_motion_energy
    from aslcv.grading.embedding_dataset import _tempo_features
    raw = grader.pipeline.assemble(poses).features
    energy = hand_motion_energy(grader.pipeline.normalizer, grader.pipeline.skeleton, poses)
    return raw, _tempo_features(energy)


def test_inference_net_matches_packed_forward(grader):
    from aslcv.grading.inference_net import OUTPUT_NAMES, InferenceGraderNet
    net = InferenceGraderNet(grader.model).eval()
    with torch.no_grad():
        for _, poses in list(_val_cases(grader))[:8]:
            raw, tempo = _inputs(grader, poses)
            f = torch.from_numpy(grader.standardizer.transform(raw))[None].to(grader.device)
            t = torch.from_numpy(tempo)[None].to(grader.device)
            ref = grader.model(f, torch.tensor([f.shape[1]], device=grader.device), t)
            for name, out in zip(OUTPUT_NAMES, net(f, t)):
                assert torch.allclose(ref[name], out, atol=1e-4), name


def test_bundle_verdicts_match_embedding_grader(grader, bundle):
    """Every predicted label and judged/unjudged flag identical; fidelity and
    confidences equal to float precision -- across correct targets AND a
    mismatched target (so OFF verdicts are exercised too)."""
    signs = grader.signs
    for sign, poses in _val_cases(grader):
        raw, tempo = _inputs(grader, poses)
        for target in (sign, signs[(signs.index(sign) + 7) % len(signs)]):
            want = grader.grade_against_poses(poses, target)
            got = bundle.grade(raw, tempo, target)
            assert abs(want.fidelity - got.fidelity) < 1e-4
            for p, wv in want.parameters.items():
                gv = got.parameters[p]
                assert (gv.predicted, gv.target, gv.correct) == (wv.predicted, wv.target, wv.correct), (sign, target, p)
                assert abs(gv.confidence - wv.confidence) < 1e-4


def test_bundle_metadata_is_self_consistent(bundle):
    m = bundle.meta
    assert bundle.references.shape == (m["references"]["rows"], m["embed_dim"])
    assert sum(s["count"] for s in m["references"]["signs"]) == m["references"]["rows"]
    assert len(bundle.mean) == len(bundle.std) == m["feature_dim"]
    assert set(m["signs"]) == {s["sign"] for s in m["references"]["signs"]}
    assert np.allclose(np.linalg.norm(bundle.references, axis=1), 1.0, atol=1e-4)  # L2-normalized embeds


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
