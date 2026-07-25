import threading
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Custom, draw_skeleton
from rtmlib.tools.file import download_checkpoint

from .base import Extractor, Pose, RunningMode, Skeleton
from .coco_wholebody import COCO_WHOLEBODY

# Keep all model weights inside the project. Resolved relative to this file
# (src/aslcv/extractor/rtmlib_base.py -> project root) so it works wherever the repo lives.
MODELS_DIR = str(Path(__file__).resolve().parents[3] / "models")

# One shared top-down person detector for every rtmlib whole-body pose head.
DET_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
DET_INPUT_SIZE = (640, 640)


def local_model(url):
    """Resolve a checkpoint URL to a local path under models/, downloading once.

    Handles both openmmlab `.zip` bundles (unzipped to a single .onnx) and direct
    `.onnx` files (e.g. easy_ViTPose on HuggingFace). Reuses an already-present
    .onnx without re-downloading.
    """
    return download_checkpoint(url, dst_dir=MODELS_DIR)


class RTMLibWholebodyExtractor(Extractor):
    """Shared machinery for rtmlib top-down COCO-WholeBody extractors.

    A YOLOX person detector feeds a pose head, producing a 133-keypoint
    COCO-WholeBody `Pose` (see `coco_wholebody.COCO_WHOLEBODY`). DWPose, RTMW, and
    ViTPose differ only in the pose head, which a subclass selects via three class
    attributes:

        POSE_CLASS       rtmlib pose class name ("RTMPose", "ViTPose", ...)
        POSE_URL         checkpoint URL (.zip bundle or direct .onnx)
        POSE_INPUT_SIZE  (W, H) the pose model expects

    Running mode
    ------------
    rtmlib inference is synchronous and has no native streaming API, so the
    running modes are realised with the one temporal lever this backend has --
    detection cadence + reuse of the last pose:

    IMAGE (default) -- detect on every frame, no reuse. Correct for single
        stills and unordered frames; `process_every_n_frames` is ignored.
    VIDEO -- ordered offline processing. `process_every_n_frames` MUST be 1
        (enforced at construction -- the constructor raises otherwise): every
        frame is detected, no reuse. This is what offline batch extraction over
        a video file needs -- silently reusing the last pose for n>1 frames
        would duplicate real frames into the cache and zero the velocity delta
        between them, which is never safe for data you will train on. Blocks
        while a frame is being processed (fine for batch extraction).
    LIVE -- a real-time camera. Detection runs on a background thread; `extract`
        submits the newest frame and returns immediately with the most recently
        completed pose, dropping any frame that arrives while the worker is
        busy. This keeps the capture loop at full frame rate; the trade-off is a
        pose that can lag the current frame. (`process_every_n_frames` is not
        used in LIVE -- the worker already drops whatever it cannot keep up
        with.)
    """

    POSE_CLASS: str = "RTMPose"
    POSE_URL: str = ""
    POSE_INPUT_SIZE: tuple = (288, 384)

    def __init__(
        self,
        device="cuda",
        backend="onnxruntime",
        process_every_n_frames=1,
        openpose_skeleton=False,
        kpt_thr=0.43,
        running_mode: RunningMode = RunningMode.IMAGE,
    ):
        super().__init__(running_mode)
        if not self.POSE_URL:
            raise ValueError(f"{type(self).__name__} must set POSE_URL")
        if running_mode is RunningMode.VIDEO and process_every_n_frames != 1:
            raise ValueError(
                f"{type(self).__name__}: VIDEO-mode rtmlib extraction requires "
                f"process_every_n_frames=1 (got {process_every_n_frames}). n>1 "
                f"duplicates real frames into the cache and zeroes velocity deltas "
                f"between them -- never safe for data you will train on. Use IMAGE "
                f"mode (offline batch extraction; ignores this setting) or LIVE mode "
                f"(drops frames instead of duplicating them) if you need throughput."
            )
        self.device = device
        self.backend = backend
        self.openpose_skeleton = openpose_skeleton
        self.kpt_thr = kpt_thr
        self.process_every_n_frames = process_every_n_frames
        self._last_pose = None

        self.custom = Custom(
            to_openpose=False,
            det_class="YOLOX",
            det=local_model(DET_URL),
            det_input_size=DET_INPUT_SIZE,
            pose_class=self.POSE_CLASS,
            pose=local_model(self.POSE_URL),
            pose_input_size=self.POSE_INPUT_SIZE,
            backend=backend,
            device=device,
        )

        # LIVE plumbing: a single worker thread runs inference off the capture
        # loop. `_pending` holds only the newest unprocessed frame (older ones
        # are dropped); `_pose_lock` guards `_last_pose` shared with `extract`.
        self._worker = None
        self._pending = None
        self._pending_lock = threading.Lock()
        self._pose_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        if running_mode is RunningMode.LIVE:
            self._worker = threading.Thread(
                target=self._live_loop, name=f"{type(self).__name__}-live", daemon=True
            )
            self._worker.start()

    @property
    def skeleton(self) -> Skeleton:
        return COCO_WHOLEBODY

    def _pose_from_frame(self, frame) -> Pose | None:
        """Run detection on a single frame. Pure w.r.t. shared state."""
        keypoints, scores = self.custom(frame)  # (P, 133, 2), (P, 133)
        if keypoints is None or len(keypoints) == 0:
            return None
        height, width = frame.shape[:2]
        best = int(np.argmax(scores.mean(axis=1)))  # squeeze to (133, 2)/(133,)
        return Pose(keypoints[best], scores[best], width=width, height=height)

    def _live_loop(self):
        """Background worker: process the newest submitted frame, forever."""
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._pending_lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            pose = self._pose_from_frame(frame)
            with self._pose_lock:
                self._last_pose = pose

    def extract(self, frame) -> Pose | None:
        if self.running_mode is RunningMode.LIVE:
            with self._pending_lock:
                self._pending = frame  # overwrite -> drop stale frames
            self._wake.set()
            with self._pose_lock:
                return self._last_pose

        # IMAGE and VIDEO both fall through to here and detect every frame --
        # VIDEO used to have a skip-and-reuse-last-pose branch, but the
        # constructor now guarantees process_every_n_frames == 1 whenever
        # running_mode is VIDEO (see __init__), which made that branch an
        # unconditional no-op, so it was removed rather than left dead.
        self._last_pose = self._pose_from_frame(frame)
        return self._last_pose

    def draw(self, frame, pose, threshold: float | None = None):
        if pose is None:
            return frame
        return draw_skeleton(
            frame.copy(),
            pose.keypoints[None],  # (re-add the person axis draw_skeleton expects)
            pose.scores[None],
            openpose_skeleton=self.openpose_skeleton,
            kpt_thr=self.kpt_thr if threshold is None else threshold,
        )

    def close(self):
        if self._worker is not None:
            self._stop.set()
            self._wake.set()
            self._worker.join(timeout=2.0)
            self._worker = None
        cv2.destroyAllWindows()
