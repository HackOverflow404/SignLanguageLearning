import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

from .base import Extractor, Pose, RunningMode, Skeleton

# Keep all model weights inside the project. Resolved relative to this file
# (src/aslcv/extractor/mediapipe.py -> project root) so it works wherever the repo lives.
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
POSE_MODEL_PATH = str(MODELS_DIR / "pose_landmarker_full.task")
FACE_MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")
HAND_MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")

POSE_LM_COUNT = 33
FACE_LM_COUNT = 478  # 468 face-mesh points + 10 iris points
HAND_LM_COUNT = 21

# The 52 MediaPipe face-blendshape coefficients, in the fixed order the model
# emits them (ARKit naming). A Pose's `blendshapes` (52,) vector is aligned to
# this tuple index-for-index. These carry non-manual grammar (brow raise -> y/n
# question, brow furrow -> wh-question, mouth morphemes) that geometric face
# landmarks alone don't make explicit.
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight",
)
BLENDSHAPE_COUNT = len(BLENDSHAPE_NAMES)  # 52

# Offsets of each landmark group within the combined keypoint array.
_POSE_OFFSET = 0
_FACE_OFFSET = _POSE_OFFSET + POSE_LM_COUNT
_LEFT_HAND_OFFSET = _FACE_OFFSET + FACE_LM_COUNT
_RIGHT_HAND_OFFSET = _LEFT_HAND_OFFSET + HAND_LM_COUNT
_TOTAL_KEYPOINTS = _RIGHT_HAND_OFFSET + HAND_LM_COUNT


# Semantic anchors as MediaPipe pose-landmark indices (BlazePose ordering).
# Mapped to combined-array indices via _POSE_OFFSET so they stay correct if the
# layout is ever reordered. Names match the COCO-WholeBody anchors so downstream
# normalization is identical across backends.
_POSE_ANCHOR_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}


def _build_skeleton() -> Skeleton:
    names = (
        [f"pose_{i}" for i in range(POSE_LM_COUNT)]
        + [f"face_{i}" for i in range(FACE_LM_COUNT)]
        + [f"left_hand_{i}" for i in range(HAND_LM_COUNT)]
        + [f"right_hand_{i}" for i in range(HAND_LM_COUNT)]
    )

    def offset_edges(connections, offset):
        return [(c.start + offset, c.end + offset) for c in connections]

    edges = (
        offset_edges(vision.PoseLandmarksConnections.POSE_LANDMARKS, _POSE_OFFSET)
        + offset_edges(vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS, _FACE_OFFSET)
        + offset_edges(vision.HandLandmarksConnections.HAND_CONNECTIONS, _LEFT_HAND_OFFSET)
        + offset_edges(vision.HandLandmarksConnections.HAND_CONNECTIONS, _RIGHT_HAND_OFFSET)
    )

    anchors = tuple(
        (name, _POSE_OFFSET + landmark) for name, landmark in _POSE_ANCHOR_LANDMARKS.items()
    )

    return Skeleton(names=tuple(names), edges=tuple(edges), anchors=anchors)


MEDIAPIPE_HOLISTIC = _build_skeleton()


def _fill_segment(keypoints, scores, offset, landmarks, width, height, use_visibility=False):
    for i, landmark in enumerate(landmarks):
        keypoints[offset + i] = (landmark.x * width, landmark.y * height)
        visibility = landmark.visibility if use_visibility else None
        scores[offset + i] = visibility if visibility is not None else 1.0


class MediaPipePoseExtractor(Extractor):
    """Body pose + face mesh + both hands via MediaPipe Tasks, combined into a
    single wholebody Pose keyed by MEDIAPIPE_HOLISTIC.

    Handedness convention (`mirrored`)
    ----------------------------------
    MediaPipe's hand "Left"/"Right" labels are anatomically correct on a raw,
    NON-flipped frame and INVERTED on a mirrored (horizontally flipped) one.
    (MediaPipe's docs describe handedness as "assuming a mirrored input"; on the
    pinned mediapipe 0.10.35 the observed behaviour is the reverse -- raw labels
    match anatomy for non-flipped input and flip for flipped input, verified
    against BlazePose's anatomically-labelled wrists across the ASL-LEX clips.
    This class follows the observed behaviour, not the doc wording.)

    Our two paths disagree on orientation: the live demo flips the webcam frame
    (mirrored), while ASL Citizen / ASL-LEX training videos are NOT flipped.
    Left uncorrected, the *same physical hand* lands in different slots between
    training and inference -- a silent train/serve mismatch. So `mirrored` is a
    required constructor argument (no default: the caller must state which feed
    this is):

        mirrored=True   frame is already flipped (selfie / live demo) -> labels
                        are swapped back to anatomy
        mirrored=False  frame is raw / not flipped (ASL Citizen, most files) ->
                        labels used as-is

    Whichever you pass, the `left_hand_*` slot ALWAYS holds the signer's
    anatomical LEFT hand and `right_hand_*` the anatomical right, so features
    computed over these slots line up across both paths.

    Confidence channel
    ------------------
    Only pose landmarks carry a real per-point confidence (BlazePose
    `visibility`). Face and hand landmarks have no confidence and are stored as
    a hard-coded 1.0. So this backend's confidence channel is effectively
    BINARY -- a point is either present (1.0) or absent (0.0) -- for everything
    except the pose block. DWPose, by contrast, emits a graded score for every
    keypoint. Downstream code that thresholds on confidence should not assume
    graded values here.

    Face blendshapes
    ----------------
    A detected face also yields a (52,) `Pose.blendshapes` vector -- graded
    ARKit-style coefficients (see `BLENDSHAPE_NAMES`) capturing facial action
    (brow raise/furrow, mouth shapes) that the geometric face mesh leaves
    implicit. This is the non-manual-marker signal MediaPipe is chosen for.
    It is None when no face is detected (and always None for backends without
    blendshapes, e.g. DWPose).

    Running mode
    ------------
    `running_mode=RunningMode.IMAGE` (default) re-detects every frame with no
    tracking -- correct for single stills. `RunningMode.VIDEO` tracks landmarks
    across an ordered stream (smoother, faster) and is the right choice for
    batch extraction over a video file; it feeds MediaPipe a monotonically
    increasing timestamp per frame, so frames must be passed in capture order.
    `RunningMode.LIVE` uses MediaPipe's async LIVE_STREAM: `extract` fires the
    detectors without blocking and returns the most recently completed pose
    (which may lag the current frame), so a webcam loop keeps its full frame
    rate. LIVE frames are stamped with a real wall-clock timestamp; use it only
    for a live feed, never for offline files (see `RunningMode`).
    """

    def __init__(
        self,
        mirrored: bool,
        running_mode: RunningMode = RunningMode.IMAGE,
        num_faces: int = 1,
        num_hands: int = 2,
    ):
        super().__init__(running_mode)
        self._mirrored = mirrored

        BaseOptions = mp.tasks.BaseOptions
        MpRunningMode = mp.tasks.vision.RunningMode
        mp_mode = {
            RunningMode.IMAGE: MpRunningMode.IMAGE,
            RunningMode.VIDEO: MpRunningMode.VIDEO,
            RunningMode.LIVE: MpRunningMode.LIVE_STREAM,
        }[running_mode]

        # LIVE_STREAM delivers results asynchronously via callbacks; the other
        # modes return them from the detect call. The latest async result per
        # landmarker is cached here (a single reference assignment, atomic under
        # the GIL, so no lock is needed between the callback and extract()).
        self._latest_pose = None
        self._latest_face = None
        self._latest_hands = None
        is_live = running_mode is RunningMode.LIVE

        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp_mode,
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_mode,
            num_faces=num_faces,
            output_face_blendshapes=True,  # 52 semantic coeffs for NMM / facial grammar
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_mode,
            num_hands=num_hands,
        )
        if is_live:
            pose_options.result_callback = self._on_pose_result
            face_options.result_callback = self._on_face_result
            hand_options.result_callback = self._on_hands_result

        self._pose = vision.PoseLandmarker.create_from_options(pose_options)
        self._face = vision.FaceLandmarker.create_from_options(face_options)
        self._hands = vision.HandLandmarker.create_from_options(hand_options)

        # VIDEO needs a monotonically increasing timestamp per frame. The
        # extract() interface carries no real frame PTS, so we advance a
        # synthetic clock at a nominal 30 fps; only the ordering matters to
        # MediaPipe's tracker, not the absolute values.
        self._video_timestamp_ms = 0
        # LIVE stamps frames with real elapsed wall-clock time, clamped to stay
        # strictly increasing even if the clock does not advance between frames.
        self._last_live_ts_ms = 0

    @property
    def skeleton(self) -> Skeleton:
        return MEDIAPIPE_HOLISTIC

    # ---- LIVE async result callbacks (one per landmarker) ----
    def _on_pose_result(self, result, output_image, timestamp_ms):
        self._latest_pose = result

    def _on_face_result(self, result, output_image, timestamp_ms):
        self._latest_face = result

    def _on_hands_result(self, result, output_image, timestamp_ms):
        self._latest_hands = result

    def _live_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        ts = max(self._last_live_ts_ms + 1, now_ms)
        self._last_live_ts_ms = ts
        return ts

    def _detect_all(self, mp_image):
        """Run the three landmarkers using the configured running mode.

        Returns the (pose, face, hand) result triple. In LIVE mode the detectors
        run asynchronously, so this fires them and returns the latest cached
        results, which may be None (before the first result arrives) or from an
        earlier frame.
        """
        if self.running_mode is RunningMode.LIVE:
            ts = self._live_timestamp_ms()
            self._pose.detect_async(mp_image, ts)
            self._face.detect_async(mp_image, ts)
            self._hands.detect_async(mp_image, ts)
            return self._latest_pose, self._latest_face, self._latest_hands
        if self.running_mode is RunningMode.VIDEO:
            self._video_timestamp_ms += 33  # ~30 fps; strictly increasing
            ts = self._video_timestamp_ms
            return (
                self._pose.detect_for_video(mp_image, ts),
                self._face.detect_for_video(mp_image, ts),
                self._hands.detect_for_video(mp_image, ts),
            )
        return (
            self._pose.detect(mp_image),
            self._face.detect(mp_image),
            self._hands.detect(mp_image),
        )

    def _hand_offset(self, label: str) -> int:
        """Combined-array offset for a MediaPipe handedness `label`.

        MediaPipe's raw "Left"/"Right" is anatomically correct on a non-mirrored
        frame and inverted on a mirrored (flipped) one. Undo that inversion for
        the mirrored path so the signer's anatomical left hand always lands in
        the left slot (see the class docstring).
        """
        anatomical_left = (label == "Right") if self._mirrored else (label == "Left")
        return _LEFT_HAND_OFFSET if anatomical_left else _RIGHT_HAND_OFFSET

    def extract(self, frame: np.ndarray) -> Pose | None:
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result, face_result, hand_result = self._detect_all(mp_image)

        # In LIVE mode any result may still be None (no result delivered yet).
        has_pose = pose_result is not None and pose_result.pose_landmarks
        has_face = face_result is not None and face_result.face_landmarks
        has_hands = hand_result is not None and hand_result.hand_landmarks

        if not (has_pose or has_face or has_hands):
            return None

        keypoints = np.zeros((_TOTAL_KEYPOINTS, 2), dtype=np.float32)
        scores = np.zeros(_TOTAL_KEYPOINTS, dtype=np.float32)

        if has_pose:
            _fill_segment(
                keypoints, scores, _POSE_OFFSET, pose_result.pose_landmarks[0],
                width, height, use_visibility=True,
            )

        if has_face:
            _fill_segment(
                keypoints, scores, _FACE_OFFSET, face_result.face_landmarks[0],
                width, height,
            )

        # Two detected hands can report the same handedness (e.g. hands crossing
        # or a mislabel). Keep the higher-scoring one per slot instead of letting
        # whichever comes last silently overwrite the other.
        best_hand_score: dict[int, float] = {}
        for landmarks, handedness in zip(
            hand_result.hand_landmarks if has_hands else [],
            hand_result.handedness if has_hands else [],
        ):
            category = handedness[0]
            offset = self._hand_offset(category.category_name)
            if best_hand_score.get(offset, -1.0) >= category.score:
                continue
            best_hand_score[offset] = category.score
            _fill_segment(keypoints, scores, offset, landmarks, width, height)

        # Face blendshapes: (52,) graded coefficients aligned to BLENDSHAPE_NAMES,
        # or None when no face was detected this frame.
        blendshapes = None
        if has_face and face_result.face_blendshapes:
            blendshapes = np.array(
                [c.score for c in face_result.face_blendshapes[0]], dtype=np.float32
            )

        return Pose(
            keypoints=keypoints, scores=scores, width=width, height=height,
            blendshapes=blendshapes,
        )

    def draw(self, frame: np.ndarray, pose: Pose) -> np.ndarray:
        annotated = frame.copy()
        # Undetected segments are left zero-filled; a tiny epsilon (not 0.0) excludes them
        # while still keeping genuinely low-visibility pose landmarks.
        confident = pose.confident(threshold=1e-6)

        for start, end in self.skeleton.edges:
            if not (confident[start] and confident[end]):
                continue
            p1 = tuple(pose.keypoints[start].astype(int))
            p2 = tuple(pose.keypoints[end].astype(int))
            cv2.line(annotated, p1, p2, color=(0, 255, 0), thickness=1)

        for i in range(len(self.skeleton.names)):
            if not confident[i]:
                continue
            center = tuple(pose.keypoints[i].astype(int))
            cv2.circle(annotated, center, radius=1, color=(0, 0, 255), thickness=-1)

        return annotated

    def close(self):
        self._pose.close()
        self._face.close()
        self._hands.close()
        cv2.destroyAllWindows()
