"""Tests for the MediaPipe wholebody extractor.

Runs under pytest OR as a plain script (`python tests/test_mediapipe_extractor.py`)
so it needs no extra dependency (there is no pytest in this venv). The three
MediaPipe landmarkers are replaced with fakes at construction time via
`create_from_options`, so the tests exercise the real `extract()` logic --
handedness resolution, hand-slot assignment, running-mode dispatch -- without
loading any model or touching a camera.

The headline test (`test_flip_invariance_*`) pins the fix for the silent
train/serve handedness mismatch: the same physical hand must land in the same
slot whether the frame is a raw (non-flipped) training frame or a flipped live
frame.
"""

from types import SimpleNamespace as ns
from unittest import mock

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

from aslcv.extractor.base import RunningMode
from aslcv.extractor import mediapipe as mpx
from aslcv.extractor.mediapipe import MediaPipePoseExtractor, HAND_LM_COUNT


# --- fakes -----------------------------------------------------------------


class FakeLandmarker:
    """Stand-in for a MediaPipe landmarker. Records how it was called and
    delegates result production to `detect_fn(mp_image)`."""

    def __init__(self, options, detect_fn):
        self.options = options
        self.detect_fn = detect_fn
        self.calls = []  # (kind, timestamp) per call

    def detect(self, image):
        self.calls.append(("image", None))
        return self.detect_fn(image)

    def detect_for_video(self, image, timestamp_ms):
        self.calls.append(("video", timestamp_ms))
        return self.detect_fn(image)

    def detect_async(self, image, timestamp_ms):
        self.calls.append(("async", timestamp_ms))
        # Emulate MediaPipe delivering the result through the registered callback.
        self.options.result_callback(self.detect_fn(image), image, timestamp_ms)

    def close(self):
        pass


def _empty_pose(image):
    return ns(pose_landmarks=[])


def _empty_face(image):
    return ns(face_landmarks=[])


def _no_hands(image):
    return ns(hand_landmarks=[], handedness=[])


def build_mp(mirrored, running_mode=RunningMode.IMAGE, hand_detect=_no_hands,
             pose_detect=_empty_pose, face_detect=_empty_face):
    """Construct a real MediaPipePoseExtractor whose landmarkers are fakes.

    The patch only needs to wrap construction: after __init__, each landmarker
    is a FakeLandmarker holding its detect_fn, and extract() calls those fakes.
    """
    with mock.patch.object(vision.PoseLandmarker, "create_from_options",
                           new=lambda options: FakeLandmarker(options, pose_detect)), \
         mock.patch.object(vision.FaceLandmarker, "create_from_options",
                           new=lambda options: FakeLandmarker(options, face_detect)), \
         mock.patch.object(vision.HandLandmarker, "create_from_options",
                           new=lambda options: FakeLandmarker(options, hand_detect)):
        return MediaPipePoseExtractor(mirrored, running_mode)


def centroid_hand_detect(image):
    """A one-hand detector faithful to real MediaPipe's observed labelling: on a
    non-flipped frame the hand at image-LEFT (the anatomical RIGHT hand of a
    facing subject) is labelled "Right", and image-RIGHT -> "Left". Flipping the
    frame moves the hand to the other half and so flips the label -- matching the
    empirically verified behaviour on mediapipe 0.10.35."""
    arr = image.numpy_view()
    height, width = arr.shape[:2]
    ys, xs = np.nonzero(arr[:, :, 0])
    if len(xs) == 0:
        return _no_hands(image)
    cx, cy = float(xs.mean()), float(ys.mean())
    label = "Right" if cx < width / 2 else "Left"
    landmarks = [ns(x=cx / width, y=cy / height) for _ in range(HAND_LM_COUNT)]
    return ns(hand_landmarks=[landmarks], handedness=[[ns(category_name=label, score=0.99)]])


def hand_slot(pose):
    """Which hand slot is populated in a Pose: 'left', 'right', 'both', 'none'."""
    left = pose.keypoints[mpx._LEFT_HAND_OFFSET: mpx._LEFT_HAND_OFFSET + HAND_LM_COUNT]
    right = pose.keypoints[mpx._RIGHT_HAND_OFFSET: mpx._RIGHT_HAND_OFFSET + HAND_LM_COUNT]
    left_on = bool(np.any(left != 0))
    right_on = bool(np.any(right != 0))
    return {(True, False): "left", (False, True): "right",
            (True, True): "both", (False, False): "none"}[(left_on, right_on)]


# --- the handedness fix (task item 1) --------------------------------------


def test_flip_invariance_same_physical_hand_same_slot():
    """A raw training frame and its flipped live counterpart must place the same
    physical hand in the same slot. Without the mirrored-swap this fails: the
    non-flipped path would put the hand in 'left' while the flipped path uses
    'right'."""
    raw = np.zeros((200, 200, 3), np.uint8)
    # Left half of a RAW (non-flipped) camera frame == the signer's anatomical
    # RIGHT hand (viewer and facing subject are mirror images).
    cv2.circle(raw, (40, 100), 18, (255, 255, 255), -1)
    flipped = cv2.flip(raw, 1)  # the same hand now on the right half

    train = build_mp(mirrored=False, hand_detect=centroid_hand_detect)  # ASL Citizen
    live = build_mp(mirrored=True, hand_detect=centroid_hand_detect)    # live demo

    slot_train = hand_slot(train.extract(raw))
    slot_live = hand_slot(live.extract(flipped))

    assert slot_train == slot_live, f"train={slot_train} live={slot_live}"
    assert slot_train == "right"  # anatomical right hand -> right slot in both paths

    train.close()
    live.close()


def test_mirrored_flag_actually_swaps_interpretation():
    """The same frame + label must map to opposite slots depending on `mirrored`,
    proving the swap is wired to the flag (not a no-op)."""
    frame = np.zeros((200, 200, 3), np.uint8)
    cv2.circle(frame, (160, 100), 18, (255, 255, 255), -1)  # right half -> "Right"

    live = build_mp(mirrored=True, hand_detect=centroid_hand_detect)
    train = build_mp(mirrored=False, hand_detect=centroid_hand_detect)

    assert hand_slot(live.extract(frame)) == "right"   # mirrored: label trusted
    assert hand_slot(train.extract(frame)) == "left"   # non-mirrored: label swapped

    live.close()
    train.close()


def test_mirrored_is_required():
    """`mirrored` has no default: the caller must state which feed this is."""
    try:
        MediaPipePoseExtractor(running_mode=RunningMode.IMAGE)  # no mirrored=
    except TypeError:
        return
    raise AssertionError("expected TypeError when mirrored is omitted")


# --- same-handedness tiebreak (task item 4) --------------------------------


def _two_hands(labels_scores, coords):
    def detect(image):
        landmarks = [[ns(x=x, y=y) for _ in range(HAND_LM_COUNT)] for (x, y) in coords]
        handedness = [[ns(category_name=lbl, score=sc)] for (lbl, sc) in labels_scores]
        return ns(hand_landmarks=landmarks, handedness=handedness)
    return detect


def _kept_hand_x(pose):
    """First-keypoint x of the single populated hand slot (asserts exactly one)."""
    slot = hand_slot(pose)
    assert slot in ("left", "right"), f"expected one populated hand slot, got {slot}"
    offset = mpx._LEFT_HAND_OFFSET if slot == "left" else mpx._RIGHT_HAND_OFFSET
    return pose.keypoints[offset][0]


def test_same_handedness_keeps_higher_score():
    # Two hands both reported "Left" (so both collapse to one slot); the
    # higher-scoring one (0.95, at x=0.8) must be the one kept.
    detect = _two_hands(
        labels_scores=[("Left", 0.30), ("Left", 0.95)],
        coords=[(0.20, 0.20), (0.80, 0.80)],
    )
    ext = build_mp(mirrored=True, hand_detect=detect)
    pose = ext.extract(np.zeros((100, 100, 3), np.uint8))

    assert abs(_kept_hand_x(pose) - 0.80 * 100) < 1e-3  # kept 0.95, not 0.30
    ext.close()


def test_same_handedness_tiebreak_is_order_independent():
    # Higher score first this time; the later lower-scoring hand must NOT overwrite.
    detect = _two_hands(
        labels_scores=[("Left", 0.95), ("Left", 0.30)],
        coords=[(0.80, 0.80), (0.20, 0.20)],
    )
    ext = build_mp(mirrored=True, hand_detect=detect)
    pose = ext.extract(np.zeros((100, 100, 3), np.uint8))

    assert abs(_kept_hand_x(pose) - 0.80 * 100) < 1e-3
    ext.close()


# --- running-mode dispatch (task item 2 + LIVE) ----------------------------


def test_running_mode_maps_to_mediapipe_mode():
    MpRM = mp.tasks.vision.RunningMode
    expected = {
        RunningMode.IMAGE: MpRM.IMAGE,
        RunningMode.VIDEO: MpRM.VIDEO,
        RunningMode.LIVE: MpRM.LIVE_STREAM,
    }
    for mode, mp_mode in expected.items():
        ext = build_mp(mirrored=False, running_mode=mode)
        for lm in (ext._pose, ext._face, ext._hands):
            assert lm.options.running_mode == mp_mode, (mode, lm.options.running_mode)
        if mode is RunningMode.LIVE:
            for lm in (ext._pose, ext._face, ext._hands):
                assert lm.options.result_callback is not None
        ext.close()


def test_video_uses_detect_for_video_with_increasing_timestamps():
    frame = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(frame, (80, 50), 10, (255, 255, 255), -1)
    ext = build_mp(mirrored=False, running_mode=RunningMode.VIDEO,
                   hand_detect=centroid_hand_detect)
    for _ in range(3):
        ext.extract(frame)

    kinds = [k for k, _ in ext._hands.calls]
    stamps = [t for _, t in ext._hands.calls]
    assert kinds == ["video", "video", "video"], kinds
    assert stamps == sorted(stamps) and len(set(stamps)) == 3, stamps
    ext.close()


def test_live_returns_latest_async_result():
    frame = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(frame, (80, 50), 10, (255, 255, 255), -1)  # right half
    ext = build_mp(mirrored=True, running_mode=RunningMode.LIVE,
                   hand_detect=centroid_hand_detect)
    pose = ext.extract(frame)  # fake detect_async delivers via callback

    assert pose is not None
    assert hand_slot(pose) == "right"
    assert [k for k, _ in ext._hands.calls] == ["async"]
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
