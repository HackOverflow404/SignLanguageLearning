"""Extraction provenance: recorded at write time (extract_landmarks.py), checked
against current pipeline expectations at read time (verify_cache.py).

Runs under pytest OR as a plain script (`python tests/test_extraction_provenance.py`).

extract_landmarks.py and verify_cache.py are scripts, not aslcv/ package modules,
so this file adds scripts/ to sys.path itself (mirroring how scripts/render_clip.py
imports its sibling scripts/extract_landmarks.py) rather than importing them as a
package.
"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import extract_landmarks as el  # noqa: E402
import verify_cache as vc  # noqa: E402

from aslcv.extractor.base import RunningMode  # noqa: E402


class FakeRtmlibExtractor:
    """Stands in for a real rtmlib extractor (no GPU/model download needed) --
    just needs the attributes checkpoint_id/extraction_provenance actually read."""
    POSE_URL = "https://example.com/fake-checkpoint.zip"

    def __init__(self, running_mode, process_every_n_frames=1):
        self.running_mode = running_mode
        self.process_every_n_frames = process_every_n_frames


class FakeMediapipeExtractor:
    """MediaPipe extractors have no process_every_n_frames concept at all --
    extraction_provenance must fall back to 1 (== "every frame") via getattr."""
    def __init__(self, running_mode):
        self.running_mode = running_mode


# --- extract_landmarks.py: writing provenance ------------------------------


def test_checkpoint_id_rtmlib_reads_pose_url():
    ext = FakeRtmlibExtractor(RunningMode.IMAGE)
    assert el.checkpoint_id("dwpose", ext) == "https://example.com/fake-checkpoint.zip"


def test_checkpoint_id_mediapipe_reads_task_path():
    from aslcv.extractor.mediapipe import POSE_MODEL_PATH
    ext = FakeMediapipeExtractor(RunningMode.VIDEO)
    assert el.checkpoint_id("mediapipe", ext) == POSE_MODEL_PATH


def test_extraction_provenance_rtmlib():
    ext = FakeRtmlibExtractor(RunningMode.IMAGE, process_every_n_frames=1)
    prov = el.extraction_provenance("dwpose", ext, "abc123")
    assert prov == {
        "running_mode": "image",
        "process_every_n_frames": 1,
        "checkpoint": "https://example.com/fake-checkpoint.zip",
        "commit_hash": "abc123",
    }


def test_extraction_provenance_mediapipe_defaults_n_to_1():
    """MediaPipe has no process_every_n_frames attribute at all -- extraction_provenance
    must record 1 ("every frame", which is what VIDEO mode actually does for it),
    not crash on the missing attribute."""
    ext = FakeMediapipeExtractor(RunningMode.VIDEO)
    prov = el.extraction_provenance("mediapipe", ext, "")
    assert prov["process_every_n_frames"] == 1
    assert prov["running_mode"] == "video"
    assert prov["commit_hash"] == ""


def test_git_commit_hash_never_raises():
    """Best-effort: must return a string (real hash or "") in every environment,
    including one where git is missing or this isn't a checkout."""
    h = el.git_commit_hash()
    assert isinstance(h, str)
    with mock.patch("subprocess.run", side_effect=FileNotFoundError("no git")):
        assert el.git_commit_hash() == ""


def test_read_provenance_roundtrips_and_handles_missing_fields():
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    try:
        np.savez_compressed(
            tmp.name, keypoints=np.zeros((1, 133, 2), np.float32), scores=np.ones((1, 133), np.float32),
            running_mode="image", process_every_n_frames=1,
            checkpoint="https://x/y.zip", commit_hash="abc123",
        )
        with np.load(tmp.name) as d:
            prov = el.read_provenance(d)
        assert prov == {
            "running_mode": "image", "process_every_n_frames": "1",
            "checkpoint": "https://x/y.zip", "commit_hash": "abc123",
        }

        # an OLDER cache with none of the provenance fields -> every field "unknown"
        np.savez_compressed(
            tmp.name, keypoints=np.zeros((1, 133, 2), np.float32), scores=np.ones((1, 133), np.float32))
        with np.load(tmp.name) as d:
            prov2 = el.read_provenance(d)
        assert all(v == "unknown" for v in prov2.values()), prov2
    finally:
        Path(tmp.name).unlink()


# --- verify_cache.py: checking provenance against current expectations -----


def _write_npz(path, backend, **provenance_overrides):
    fields = dict(
        running_mode=vc.EXPECTED_RUNNING_MODE[backend],
        process_every_n_frames=vc.EXPECTED_PROCESS_EVERY_N_FRAMES,
        checkpoint=vc.expected_checkpoint(backend),
        commit_hash="whatever",
    )
    fields.update(provenance_overrides)
    np.savez_compressed(
        path, keypoints=np.zeros((10, vc.K_EXPECT[backend], 2), np.float32),
        scores=np.ones((10, vc.K_EXPECT[backend]), np.float32),
        extractor=backend, **fields,
    )


def test_expected_checkpoint_matches_real_extractor_classes():
    """Cross-check against the actual classes, so this can't silently drift from
    what extract_landmarks.py would really record (see checkpoint_id)."""
    from aslcv.extractor.dwpose import DWPoseExtractor
    from aslcv.extractor.rtmw import RTMWExtractor
    from aslcv.extractor.vitpose import ViTPoseExtractor
    from aslcv.extractor.mediapipe import POSE_MODEL_PATH
    assert vc.expected_checkpoint("dwpose") == DWPoseExtractor.POSE_URL
    assert vc.expected_checkpoint("rtmw") == RTMWExtractor.POSE_URL
    assert vc.expected_checkpoint("vitpose") == ViTPoseExtractor.POSE_URL
    assert vc.expected_checkpoint("mediapipe") == POSE_MODEL_PATH


def test_check_provenance_unknown_for_pre_provenance_cache():
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    try:
        np.savez_compressed(
            tmp.name, keypoints=np.zeros((10, 133, 2), np.float32),
            scores=np.ones((10, 133), np.float32), extractor="dwpose")
        with np.load(tmp.name) as z:
            status, mismatches = vc.check_provenance(z, "dwpose")
        assert status == "unknown"
        assert mismatches == []
    finally:
        Path(tmp.name).unlink()


def test_check_provenance_clean_matches_current_pipeline():
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    try:
        _write_npz(tmp.name, "dwpose")
        with np.load(tmp.name) as z:
            status, mismatches = vc.check_provenance(z, "dwpose")
        assert status == "checked"
        assert mismatches == [], mismatches
    finally:
        Path(tmp.name).unlink()


def test_check_provenance_flags_the_exact_bug_this_guards_against():
    """VIDEO mode + process_every_n_frames=3 -- the config that used to duplicate
    frames and zero velocity deltas. Must be flagged, not silently accepted."""
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    try:
        _write_npz(tmp.name, "dwpose", running_mode="video", process_every_n_frames=3)
        with np.load(tmp.name) as z:
            status, mismatches = vc.check_provenance(z, "dwpose")
        assert status == "checked"
        joined = " ".join(mismatches)
        assert "running_mode" in joined
        assert "process_every_n_frames=3" in joined
    finally:
        Path(tmp.name).unlink()


def test_check_provenance_flags_checkpoint_drift():
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    try:
        _write_npz(tmp.name, "dwpose", checkpoint="https://old/checkpoint.zip")
        with np.load(tmp.name) as z:
            status, mismatches = vc.check_provenance(z, "dwpose")
        assert status == "checked"
        assert any("checkpoint changed" in m for m in mismatches)
    finally:
        Path(tmp.name).unlink()


def test_check_npz_keeps_structural_problems_separate_from_provenance():
    """A structurally corrupt file and a provenance mismatch are reported through
    DIFFERENT channels -- corruption must never be masked as (or by) staleness."""
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    try:
        # wrong K (structural problem) AND a stale checkpoint (provenance problem)
        np.savez_compressed(
            tmp.name, keypoints=np.zeros((10, 5, 2), np.float32), scores=np.ones((10, 5), np.float32),
            extractor="dwpose", running_mode="image", process_every_n_frames=1,
            checkpoint="https://old/checkpoint.zip", commit_hash="x",
        )
        problems, status, mismatches = vc.check_npz(tmp.name, "dwpose")
        assert any("K " in p for p in problems), problems
        assert status == "checked"
        assert any("checkpoint changed" in m for m in mismatches)
    finally:
        Path(tmp.name).unlink()


if __name__ == "__main__":
    passed = failed = 0
    for _name, fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {_name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
