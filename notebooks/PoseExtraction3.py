# %% Cell 1: Import libraries
import cv2
from rtmlib import Custom, Wholebody, draw_skeleton
from rtmlib.tools.file import download_checkpoint

# %% Cell 2: Configuration
device = "cuda"
backend = "onnxruntime"  # opencv, onnxruntime
cap = cv2.VideoCapture(0)
openpose_skeleton = False  # True for openpose-style, False for mmpose-style
PROCESS_EVERY_N_FRAMES = 3

# Keep all model weights inside the project. local_model() returns the local
# .onnx path, downloading + extracting it flat into models/ the first time and
# reusing it on every later run (nothing is written to ~/.cache).
MODELS_DIR = "/home/hackoverflow/Documents/Projects/SignLanguageLearning/models"


def local_model(url):
    return download_checkpoint(url, dst_dir=MODELS_DIR)


DET_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk//rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip"

# %% Cell 3: Initialization
# wholebody = Wholebody(
#     mode="performance", to_openpose=openpose_skeleton, backend=backend, device=device
# )
custom = Custom(
    to_openpose=False,
    det_class="YOLOX",
    det=local_model(DET_URL),
    det_input_size=(640, 640),
    pose_class="RTMPose",  # DWPose uses the RTMPose class
    pose=local_model(POSE_URL),
    pose_input_size=(192, 256),  # (W, H) for a 256x192 (H×W) model
    backend=backend,
    device=device,
)
frame_idx = 0

# %% Cell 4: Inferencing
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame_idx += 1
    if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
        continue

    # keypoints, scores = wholebody(frame)
    keypoints, scores = custom(frame)

    # scores[:, 13:23] = 0.0

    print("Keypoints:", keypoints, "Scores:", scores)

    img_show = frame.copy()

    img_show = draw_skeleton(
        img_show, keypoints, scores, openpose_skeleton=openpose_skeleton, kpt_thr=0.43
    )

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow("img", img_show)
    cv2.waitKey(10)
    key = cv2.waitKey(10) & 0xFF

    if key == ord("q"):  # If 'q' is pressed, exit
        break
    elif key == 27:  # If 'ESC' (ASCII 27) is pressed, exit
        break

cap.release()
cv2.destroyAllWindows()
