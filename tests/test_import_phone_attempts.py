"""scripts/import_phone_attempts.py: the phone's attempt files (layout in
app/aslcv_core/lib/src/attempt_log.dart) become the desktop's format, which
aslcv.attempt_log.load_attempt reads unchanged."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import import_phone_attempts as imp  # noqa: E402
from aslcv.attempt_log import load_attempt  # noqa: E402


def _phone_files(folder, n=5, k=3, stamp="20261013-120000-000", **meta):
    rng = np.random.default_rng(0)
    times = 100 + np.arange(n) / 24
    kp = rng.random((n, k, 2), dtype=np.float32)
    sc = rng.random((n, k), dtype=np.float32)
    states = np.array([0, 1, 1, 2, 2][:n], np.int8)
    raw = times.astype("<f8").tobytes() + kp.astype("<f4").tobytes() + sc.astype("<f4").tobytes() + states.tobytes()
    (folder / f"{stamp}.bin.gz").write_bytes(gzip.compress(raw))
    sidecar = {"format": imp.FORMAT, "n_frames": n, "n_keypoints": k, "dropped_frames": 0, "stream_fps": 24.0,
               "event": "cleared", "mode": "sign", "target": "come", "sentence": None, "gloss_ids": ["come"],
               "extractor": "mediapipe", "capture": {"motion_threshold": 0.1, "settle_frames": 8, "preroll": 12,
                                                     "max_frames": 150},
               "n_captured": 3, "captured_span": [1, 4], "segments": None, **meta}
    (folder / f"{stamp}.json").write_text(json.dumps(sidecar))
    return times, kp, sc, states


def test_converts_to_the_desktop_format(tmp_path):
    src, out = tmp_path / "attempts", tmp_path / "out"
    src.mkdir()
    times, kp, sc, states = _phone_files(src)
    path = imp.convert(src / "20261013-120000-000.json", out)
    att = load_attempt(path)
    assert np.array_equal(att.states, states) and np.allclose(att.timestamps, times)
    assert np.array_equal(np.stack([p.keypoints for p in att.poses]), kp)
    assert np.array_equal(np.stack([p.scores for p in att.poses]), sc)
    assert att.meta["sentence"] == "come" and att.meta["source"] == "phone"
    assert att.fps == pytest.approx(24.0)
    assert imp.convert(src / "20261013-120000-000.json", out) is None  # already imported


def test_sentence_attempt_keeps_its_sentence(tmp_path):
    _phone_files(tmp_path, mode="sentence", target=None, sentence="I want water.")
    att = load_attempt(imp.convert(tmp_path / "20261013-120000-000.json", tmp_path / "out"))
    assert att.meta["sentence"] == "I want water."


def test_rejects_a_size_mismatch_and_unknown_format(tmp_path):
    _phone_files(tmp_path)
    side = tmp_path / "20261013-120000-000.json"
    meta = json.loads(side.read_text())
    side.write_text(json.dumps(dict(meta, n_keypoints=4)))
    with pytest.raises(ValueError):
        imp.convert(side, tmp_path / "out")
    side.write_text(json.dumps(dict(meta, format="other/1")))
    with pytest.raises(ValueError, match="unknown format"):
        imp.convert(side, tmp_path / "out")
