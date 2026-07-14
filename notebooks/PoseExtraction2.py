# %% Cell 1: Import libraries
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np

# %% Cell 2: Configuration
MODELS_DIR = "/home/hackoverflow/Documents/Projects/SignLanguageLearning/models"
POSE_MODEL_PATH = f"{MODELS_DIR}/pose_landmarker_full.task"
FACE_MODEL_PATH = f"{MODELS_DIR}/face_landmarker.task"
HAND_MODEL_PATH = f"{MODELS_DIR}/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult

FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarkerResult = vision.FaceLandmarkerResult

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult


def draw_pose_landmarks(annotated_image, pose_result):
    if pose_result is None:
        return

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )


def draw_face_landmarks(annotated_image, face_result):
    if face_result is None:
        return

    tesselation_style = drawing_styles.get_default_face_mesh_tesselation_style()
    contours_style = drawing_styles.get_default_face_mesh_contours_style()

    for face_landmarks in face_result.face_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=tesselation_style,
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=contours_style,
        )


def draw_hand_landmarks(annotated_image, hand_result):
    if hand_result is None:
        return

    hand_landmark_style = drawing_styles.get_default_hand_landmarks_style()
    hand_connection_style = drawing_styles.get_default_hand_connections_style()

    for hand_landmarks in hand_result.hand_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_style,
            connection_drawing_spec=hand_connection_style,
        )


def draw_landmarks_on_image(rgb_image, pose_result, face_result, hand_result):
    annotated_image = np.copy(rgb_image)
    draw_pose_landmarks(annotated_image, pose_result)
    draw_face_landmarks(annotated_image, face_result)
    draw_hand_landmarks(annotated_image, hand_result)
    return annotated_image


latest_pose_result = None
latest_face_result = None
latest_hand_result = None


def on_pose_result(
    result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global latest_pose_result
    latest_pose_result = result


def on_face_result(
    result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global latest_face_result
    latest_face_result = result


def on_hand_result(
    result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global latest_hand_result
    latest_hand_result = result


# Create landmarker instances in live-stream mode, one per model:
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_pose_result,
)
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=on_face_result,
)
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_hand_result,
)

cap = cv2.VideoCapture(0)
PROCESS_EVERY_N_FRAMES = 3

# %% Cell 3: Initialization
frame_idx = 0
start_time = time.time()

# %% Cell 4: Inferencing
with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     FaceLandmarker.create_from_options(face_options) as face_landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            frame_idx += 1
            if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            pose_landmarker.detect_async(mp_image, timestamp_ms)
            face_landmarker.detect_async(mp_image, timestamp_ms)
            hand_landmarker.detect_async(mp_image, timestamp_ms)

            annotated_rgb = draw_landmarks_on_image(
                rgb_frame, latest_pose_result, latest_face_result, latest_hand_result
            )
            img_show = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            img_show = cv2.resize(img_show, (960, 540))
            cv2.imshow("img", img_show)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):  # If 'q' is pressed, exit
                break
            elif key == 27:  # If 'ESC' (ASCII 27) is pressed, exit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
