from pathlib import Path

import cv2
import numpy as np
from rtmlib import Custom, draw_skeleton  # (removed unused `Wholebody` import)
from rtmlib.tools.file import download_checkpoint

from .base import Pose, PoseExtractor, Skeleton
from .coco_wholebody import COCO_WHOLEBODY

# Keep all model weights inside the project. Resolved relative to this file
# (src/aslcv/pose/dwpose.py -> project root) so it works wherever the repo lives.
MODELS_DIR = str(Path(__file__).resolve().parents[3] / "models")

DET_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip"


def local_model(url):
    return download_checkpoint(url, dst_dir=MODELS_DIR)


class DWPoseExtractor(PoseExtractor):
    def __init__(
        self,
        device="cuda",
        backend="onnxruntime",
        process_every_n_frames=3,
        openpose_skeleton=False,
        kpt_thr=0.43,
    ):
        self.device = device
        self.backend = backend
        self.openpose_skeleton = openpose_skeleton
        self.kpt_thr = kpt_thr
        self.process_every_n_frames = process_every_n_frames
        self.frame_idx = 0
        self._last_pose = None

        self.custom = Custom(
            to_openpose=False,
            det_class="YOLOX",
            det=local_model(DET_URL),
            det_input_size=(640, 640),
            pose_class="RTMPose",  # DWPose uses the RTMPose class
            pose=local_model(POSE_URL),
            pose_input_size=(288, 384),  # (W, H) for a 384x288 (H×W) model
            backend=backend,
            device=device,
        )

    @property
    def skeleton(self) -> Skeleton:
        return COCO_WHOLEBODY

    def extract(self, frame) -> Pose | None:
        self.frame_idx += 1

        if self.frame_idx % self.process_every_n_frames != 0:
            return self._last_pose

        keypoints, scores = self.custom(frame)  # (P, 133, 2), (P, 133)

        if keypoints is None or len(keypoints) == 0:
            self._last_pose = None
            return None

        #  squeeze to (133, 2) / (133,)
        best = int(np.argmax(scores.mean(axis=1)))
        self._last_pose = Pose(keypoints[best], scores[best])
        return self._last_pose

    def draw(self, frame, pose):
        if pose is None:
            return frame
        return draw_skeleton(
            frame.copy(),
            pose.keypoints[None],  # (re-add the person axis draw_skeleton expects)
            pose.scores[None],
            openpose_skeleton=self.openpose_skeleton,
            kpt_thr=self.kpt_thr,
        )

    def close(self):
        cv2.destroyAllWindows()
