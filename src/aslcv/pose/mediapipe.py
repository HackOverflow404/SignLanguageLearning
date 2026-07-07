from .base import PoseExtractor, Pose, Skeleton
from mediapipe.tasks.python import vision


class MediaPipePoseExtractor(PoseExtractor):
    def __init__(self, device="cpu", backend="onnxruntime", kpt_thr=0.43):
        self.device = device
        self.backend = backend
        self.kpt_thr = kpt_thr
        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=kpt_thr,
            min_tracking_confidence=kpt_thr,
        )

    @property
    def skeleton(self) -> Skeleton:
        return Skeleton(
            names=mp.solutions.pose.PoseLandmark._member_names_ ,
            edges=(),
        )

