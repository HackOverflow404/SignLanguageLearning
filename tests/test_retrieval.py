"""Phase 5a retrieval tests -- runs against the real cached mediapipe data
(same convention as test_dataset.py; no data means no ASL to retrieve).

Runs under pytest OR as a plain script (`python tests/test_retrieval.py`).
"""
import numpy as np
import pytest

from aslcv.production import GlossRuleEngine, fetch_reference, fetch_sequence, write_composed_video
from aslcv.production.gloss_rules import Gloss, GlossSequence
from aslcv.production.retrieval import _pick_row

E = GlossRuleEngine()


# -- fetch_reference ----------------------------------------------------------

def test_fetch_reference_returns_real_files():
    clip = fetch_reference("mother", extractor="mediapipe")
    assert clip.id_gloss == "mother"
    assert clip.video_path.exists()
    assert clip.npz_path is not None and clip.npz_path.exists()
    assert clip.n_available > 1  # mother has many clips in ASL Citizen


def test_fetch_reference_is_deterministic():
    a = fetch_reference("mother", extractor="mediapipe")
    b = fetch_reference("mother", extractor="mediapipe")
    assert a.video_id == b.video_id


def test_fetch_reference_prefers_train_split():
    rows = [
        {"split": "test", "video_id": "z"},
        {"split": "train", "video_id": "a"},
        {"split": "val", "video_id": "b"},
    ]
    assert _pick_row(rows)["video_id"] == "a"


def test_fetch_reference_falls_back_to_first_when_no_train():
    rows = [{"split": "val", "video_id": "b"}, {"split": "test", "video_id": "z"}]
    assert _pick_row(rows)["video_id"] == "b"


def test_fetch_reference_unknown_sign_raises():
    with pytest.raises(KeyError):
        fetch_reference("not_a_real_curriculum_sign")


def test_fetch_reference_npz_none_for_unrelated_id_gloss_but_real_sign():
    # every curriculum sign has SOME manifest row; npz_path should resolve for
    # every default (mediapipe) extractor since all 60 signs are cached there
    clip = fetch_reference("father", extractor="mediapipe")
    assert clip.npz_path is not None


# -- fetch_sequence -------------------------------------------------------------

def test_fetch_sequence_orders_and_trims_clips():
    seq = E.gloss("I want water.")
    assert seq.in_scope
    composed = fetch_sequence(seq, extractor="mediapipe")

    assert [c.id_gloss for c in composed.clips] == seq.gloss_ids
    assert len(composed.frames) > 0
    assert len(composed.clip_frame_ranges) == len(composed.clips)

    # ranges tile [0, total) contiguously, in order, with no gaps/overlaps
    expected_start = 0
    for start, stop in composed.clip_frame_ranges:
        assert start == expected_start
        assert stop > start
        expected_start = stop
    assert expected_start == len(composed.frames)

    # trimming should never grow a clip past its own raw frame count
    for clip, (start, stop) in zip(composed.clips, composed.clip_frame_ranges):
        n_raw = len(np.load(clip.npz_path)["keypoints"])
        assert (stop - start) <= n_raw


def test_fetch_sequence_refuses_out_of_scope():
    seq = E.gloss("The dog that I saw was tired.")
    assert not seq.in_scope
    with pytest.raises(ValueError):
        fetch_sequence(seq)


def test_fetch_sequence_refuses_empty_sequence():
    empty = GlossSequence(
        english="", in_scope=True, confidence=1.0, reason=None,
        sentence_type="statement", negated=False, glosses=[],
    )
    with pytest.raises(ValueError):
        fetch_sequence(empty)


def test_fetch_sequence_refuses_missing_reference_clip():
    bad_gloss = Gloss(text="NOPE", asllex_id="not_a_real_sign", source="nope", pos="NOUN")
    seq = GlossSequence(
        english="nope", in_scope=True, confidence=1.0, reason=None,
        sentence_type="statement", negated=False, glosses=[bad_gloss],
    )
    with pytest.raises(ValueError):
        fetch_sequence(seq)


def test_write_composed_video_roundtrips(tmp_path):
    seq = E.gloss("I want water.")
    composed = fetch_sequence(seq, extractor="mediapipe")
    out = write_composed_video(composed, tmp_path / "composed.mp4")
    assert out.exists() and out.stat().st_size > 0

    import cv2
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == len(composed.frames)


def test_write_composed_video_refuses_empty():
    empty = type("C", (), {"frames": []})()
    with pytest.raises(ValueError):
        write_composed_video(empty, "/tmp/should_not_be_written.mp4")


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                    import tempfile
                    from pathlib import Path
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                print(f"  OK   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
