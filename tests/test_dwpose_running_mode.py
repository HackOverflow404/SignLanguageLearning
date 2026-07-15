"""Running-mode behaviour for the DWPose extractor.

Runs under pytest OR as a plain script (`python tests/test_dwpose_running_mode.py`).
The rtmlib `Custom` predictor (which needs onnxruntime + model downloads) is
replaced with an in-memory fake, so these tests exercise the mode dispatch --
IMAGE detects every frame, VIDEO skips + reuses the cache, LIVE runs detection
on a background thread and drops stale frames -- with no GPU or network.
"""

import threading
import time
from unittest import mock

import numpy as np

from aslcv.extractor.base import RunningMode
from aslcv.extractor import dwpose as dw


def frame(tag):
    """A tiny frame whose top-left pixel encodes an identifying tag."""
    return np.full((10, 10, 3), tag, np.uint8)


class FakeCustom:
    """Callable stand-in for rtmlib.Custom. Returns keypoints/scores filled with
    the frame's tag so a produced Pose can be traced back to its frame."""

    def __init__(self, **kwargs):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        tag = float(frame[0, 0, 0])
        keypoints = np.full((1, 133, 2), tag, np.float32)
        scores = np.full((1, 133), 0.9, np.float32)
        return keypoints, scores


def build_dw(running_mode, custom_cls=FakeCustom, **kwargs):
    with mock.patch.object(dw, "local_model", new=lambda url: "fake.onnx"), \
         mock.patch.object(dw, "Custom", new=custom_cls):
        return dw.DWPoseExtractor(running_mode=running_mode, **kwargs)


def _wait(cond, timeout=5.0, interval=0.005):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return cond()


# --- IMAGE -----------------------------------------------------------------


def test_image_mode_processes_every_frame():
    ext = build_dw(RunningMode.IMAGE)
    p1 = ext.extract(frame(1))
    p2 = ext.extract(frame(2))
    p3 = ext.extract(frame(3))

    assert ext.custom.calls == 3
    assert (p1.keypoints[0, 0], p2.keypoints[0, 0], p3.keypoints[0, 0]) == (1, 2, 3)
    ext.close()


# --- VIDEO -----------------------------------------------------------------


def test_video_mode_skips_and_reuses_cache():
    ext = build_dw(RunningMode.VIDEO, process_every_n_frames=3)
    r1 = ext.extract(frame(1))  # frame 1 -> skipped, cache empty
    r2 = ext.extract(frame(2))  # frame 2 -> skipped
    r3 = ext.extract(frame(3))  # frame 3 -> processed

    assert ext.custom.calls == 1
    assert r1 is None and r2 is None
    assert r3.keypoints[0, 0] == 3

    r4 = ext.extract(frame(4))  # frame 4 -> skipped, returns cached frame-3 pose
    assert r4.keypoints[0, 0] == 3
    assert ext.custom.calls == 1
    ext.close()


# --- LIVE ------------------------------------------------------------------


def test_live_mode_returns_processed_pose():
    ext = build_dw(RunningMode.LIVE)
    ext.extract(frame(7))  # non-blocking submit
    assert _wait(lambda: ext._last_pose is not None)
    assert ext._last_pose.keypoints[0, 0] == 7
    ext.close()


class GatedCustom:
    """Fake Custom whose calls block until the test releases them, so frame
    scheduling is deterministic."""

    def __init__(self, **kwargs):
        self.processed = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, frame):
        tag = int(frame[0, 0, 0])
        self.entered.set()
        self.release.wait()
        self.release.clear()
        self.processed.append(tag)
        return np.full((1, 133, 2), float(tag), np.float32), np.full((1, 133), 0.9, np.float32)


def test_live_mode_drops_stale_frames():
    ext = build_dw(RunningMode.LIVE, custom_cls=GatedCustom)
    gc = ext.custom

    ext.extract(frame(1))               # worker picks frame 1
    assert gc.entered.wait(timeout=5)   # worker now inside custom(frame 1)
    gc.entered.clear()

    ext.extract(frame(2))               # pending = frame 2
    ext.extract(frame(3))               # pending = frame 3 (frame 2 dropped)

    gc.release.set()                    # frame 1 completes
    assert gc.entered.wait(timeout=5)   # worker now inside custom(frame 3)
    gc.entered.clear()
    gc.release.set()                    # frame 3 completes

    assert _wait(lambda: ext._last_pose is not None and ext._last_pose.keypoints[0, 0] == 3)
    assert gc.processed == [1, 3], gc.processed  # frame 2 never processed
    ext.close()


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {name}: {e!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
