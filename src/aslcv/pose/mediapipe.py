from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from .base import Pose, PoseExtractor, Skeleton

# Keep all model weights inside the project. Resolved relative to this file
# (src/aslcv/pose/mediapipe.py -> project root) so it works wherever the repo lives.
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
POSE_MODEL_PATH = str(MODELS_DIR / "pose_landmarker_full.task")
FACE_MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")
HAND_MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")

POSE_LM_COUNT = 33
FACE_LM_COUNT = 478  # 468 face-mesh points + 10 iris points
HAND_LM_COUNT = 21

# Offsets of each landmark group within the combined keypoint array.
_POSE_OFFSET = 0
_FACE_OFFSET = _POSE_OFFSET + POSE_LM_COUNT
_LEFT_HAND_OFFSET = _FACE_OFFSET + FACE_LM_COUNT
_RIGHT_HAND_OFFSET = _LEFT_HAND_OFFSET + HAND_LM_COUNT
_TOTAL_KEYPOINTS = _RIGHT_HAND_OFFSET + HAND_LM_COUNT


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

    return Skeleton(names=tuple(names), edges=tuple(edges))


MEDIAPIPE_HOLISTIC = _build_skeleton()


def _fill_segment(keypoints, scores, offset, landmarks, width, height, use_visibility=False):
    for i, landmark in enumerate(landmarks):
        keypoints[offset + i] = (landmark.x * width, landmark.y * height)
        visibility = landmark.visibility if use_visibility else None
        scores[offset + i] = visibility if visibility is not None else 1.0


class MediaPipePoseExtractor(PoseExtractor):
    """Detects body pose, face mesh, and both hands with MediaPipe Tasks,
    combined into a single wholebody Pose keyed by MEDIAPIPE_HOLISTIC."""

    def __init__(self, num_faces: int = 1, num_hands: int = 2):
        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode

        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
            )
        )
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_faces=num_faces,
            )
        )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_hands=num_hands,
            )
        )

    @property
    def skeleton(self) -> Skeleton:
        return MEDIAPIPE_HOLISTIC

    def extract(self, frame: np.ndarray) -> Pose | None:
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self._pose.detect(mp_image)
        face_result = self._face.detect(mp_image)
        hand_result = self._hands.detect(mp_image)

        if not (
            pose_result.pose_landmarks
            or face_result.face_landmarks
            or hand_result.hand_landmarks
        ):
            return None

        keypoints = np.zeros((_TOTAL_KEYPOINTS, 2), dtype=np.float32)
        scores = np.zeros(_TOTAL_KEYPOINTS, dtype=np.float32)

        if pose_result.pose_landmarks:
            _fill_segment(
                keypoints, scores, _POSE_OFFSET, pose_result.pose_landmarks[0],
                width, height, use_visibility=True,
            )

        if face_result.face_landmarks:
            _fill_segment(
                keypoints, scores, _FACE_OFFSET, face_result.face_landmarks[0],
                width, height,
            )

        for landmarks, handedness in zip(
            hand_result.hand_landmarks, hand_result.handedness
        ):
            is_left = handedness[0].category_name == "Left"
            offset = _LEFT_HAND_OFFSET if is_left else _RIGHT_HAND_OFFSET
            _fill_segment(keypoints, scores, offset, landmarks, width, height)

        return Pose(keypoints=keypoints, scores=scores, width=width, height=height)

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
