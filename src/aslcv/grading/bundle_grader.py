"""Reference implementation of grading from an exported mobile bundle (Phase 8).

`scripts/export_mobile.py` writes a self-contained bundle: an ONNX graph of the
trained grader plus plain data files (bundle.json, references.f32). Everything
a phone app does AFTER feature assembly -- standardize, run the graph, nearest
reference distance, softmax/sigmoid, MIN_SUPPORT-gated verdicts -- is done here
from those files alone, never from the training checkpoint or torch. That makes
this module the executable spec a Kotlin/Swift/JS client must reproduce, and
`tests/test_bundle_grader.py` proves it agrees with `EmbeddingGrader`.

Bundle layout (format_version 1):
  grader.onnx     inputs  features (1, T, F) float32 standardized, tempo (1, 2) float32
                  outputs embed (1, D) + one logit tensor per head (see "outputs")
  references.f32  (N, D) float32 little-endian row-major reference embeddings,
                  grouped by sign per bundle.json's references.signs [start, count]
  bundle.json     pipeline config, standardizer mean/std, head class names, and
                  per-sign target labels with their judged/unjudged gating
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


@dataclass
class BundleVerdict:
    parameter: str
    predicted: str
    target: str
    correct: "bool | None"
    confidence: float


@dataclass
class BundleGrade:
    target_sign: str
    fidelity: float
    parameters: dict  # parameter -> BundleVerdict


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class BundleGrader:
    def __init__(self, bundle_dir: "str | Path", providers=("CPUExecutionProvider",)):
        import onnxruntime as ort

        bundle_dir = Path(bundle_dir)
        self.meta = json.loads((bundle_dir / "bundle.json").read_text())
        if self.meta["format_version"] != FORMAT_VERSION:
            raise ValueError(f"bundle format {self.meta['format_version']}, expected {FORMAT_VERSION}")
        self.mean = np.asarray(self.meta["standardizer"]["mean"], np.float32)
        self.std = np.asarray(self.meta["standardizer"]["std"], np.float32)
        refs = self.meta["references"]
        flat = np.fromfile(bundle_dir / refs["file"], dtype="<f4")
        self.references = flat.reshape(refs["rows"], refs["dim"])
        self.ref_span = {s["sign"]: (s["start"], s["count"]) for s in refs["signs"]}
        self.session = ort.InferenceSession(str(bundle_dir / self.meta["model"]), providers=list(providers))

    def run(self, raw_features: np.ndarray, tempo: np.ndarray) -> dict:
        """Raw (unstandardized) (T, F) features + (2,) tempo -> every named output."""
        feats = ((raw_features - self.mean) / self.std).astype(np.float32)[None]
        outs = self.session.run(None, {"features": feats, "tempo": tempo.astype(np.float32)[None]})
        return {name: o[0] for name, o in zip(self.meta["outputs"], outs)}

    def grade(self, raw_features: np.ndarray, tempo: np.ndarray, target_sign: str) -> BundleGrade:
        if target_sign not in self.meta["signs"]:
            raise KeyError(f"unknown target sign {target_sign!r}")
        out = self.run(raw_features, tempo)
        start, count = self.ref_span[target_sign]
        refs = self.references[start:start + count]
        fidelity = float(np.sqrt(((refs - out["embed"]) ** 2).sum(axis=1)).min())

        targets = self.meta["signs"][target_sign]
        params = {}
        for p, classes in self.meta["classes"].items():
            probs = _softmax(out[p].astype(np.float64))
            idx = int(np.argmax(out[p]))
            t = targets[p]
            params[p] = BundleVerdict(p, classes[idx], t["target"],
                                      (classes[idx] == t["target"]) if t["judged"] else None,
                                      float(probs[idx]))
        prob = float(1.0 / (1.0 + np.exp(-float(out["repeated"]))))
        pred = prob > 0.5
        t = targets["repeated_movement"]
        params["repeated_movement"] = BundleVerdict(
            "repeated_movement", str(pred), str(t["target"]),
            (pred == t["target"]) if t["judged"] else None, prob if pred else 1.0 - prob)
        return BundleGrade(target_sign, fidelity, params)
