from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RunningMode(Enum):
    """How an extractor treats the frames it is given.

    IMAGE -- every frame is detected independently, with no temporal state
        carried between calls. Correct for single stills and for frames that
        are not in time order (shuffled, sampled, or from different clips).
    VIDEO -- frames are assumed to arrive in capture order, letting the backend
        track landmarks across frames. Smoother and usually faster than
        re-detecting each frame, at the cost of requiring an ordered stream
        (and, for MediaPipe, monotonically increasing timestamps). This is the
        right choice for batch extraction over a video file, where every frame
        matters and the call may block until detection finishes.
    LIVE -- a real-time stream (a webcam). Detection is decoupled from the
        capture loop so `extract` never blocks it: each call submits the newest
        frame and returns the most recently completed pose, dropping frames the
        detector cannot keep up with. Lowest latency, at the cost of a pose that
        may lag the current frame by one or more detections. MediaPipe runs this
        via its native LIVE_STREAM (async callbacks); DWPose via a background
        inference thread.

    IMAGE is the default everywhere so a lone frame is never mis-tracked.
    """

    IMAGE = "image"
    VIDEO = "video"
    LIVE = "live"


@dataclass(frozen=True)
class Skeleton:
    """Describes a model's keypoint layout so downstream code stays model-agnostic."""

    names: tuple[str, ...]  # keypoint name per index, e.g. ("nose", "left_wrist", ...)
    edges: tuple[tuple[int, int], ...]  # index pairs to connect when drawing
    # Semantic keypoints downstream code needs by meaning, not by index. These
    # differ per topology (COCO-WholeBody shoulders = 5, 6; MediaPipe pose = 11, 12),
    # so normalization reads them from here instead of hard-coding a layout.
    anchors: tuple[tuple[str, int], ...] = ()  # e.g. (("left_shoulder", 5), ...)

    def anchor(self, name: str) -> int:
        """Index of a named semantic anchor (e.g. 'left_shoulder'); raises if absent."""
        for anchor_name, index in self.anchors:
            if anchor_name == name:
                return index
        available = [n for n, _ in self.anchors]
        raise KeyError(f"{name!r} is not a named anchor of this skeleton; have {available}")


@dataclass
class Pose:
    """A single detected pose. keypoints: (N, 2) pixel coords; scores: (N,) confidence.

    `blendshapes` is an optional (52,) vector of semantic face coefficients
    (ARKit-style: brow raise/furrow, mouth morphemes, etc.) — MediaPipe supplies
    it, geometric backends like DWPose leave it None. It is a separate channel
    from `scores`, kept for non-manual-marker (facial grammar) features.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    width: int | None = None
    height: int | None = None
    blendshapes: np.ndarray | None = None

    def confident(self, threshold: float = 0.5) -> np.ndarray:
        """Boolean mask of keypoints at or above the confidence threshold."""
        return self.scores >= threshold


class Extractor(ABC):
    """Interface for pose extractors that turn a frame into a Pose.

    Every extractor honours a `running_mode` (see `RunningMode`): IMAGE detects
    each frame independently; VIDEO tracks across an ordered stream. Subclasses
    accept it in their constructor and pass it to `super().__init__`.
    """

    def __init__(self, running_mode: RunningMode = RunningMode.IMAGE):
        self.running_mode = running_mode

    # ---- abstract: every model must provide these ----

    @property
    @abstractmethod
    def skeleton(self) -> Skeleton:
        """The keypoint layout this extractor produces (read-only)."""
        ...

    @abstractmethod
    def extract(self, frame: np.ndarray) -> Pose | None:
        """Return the detected Pose for a frame, or None if no pose was found."""
        ...

    @abstractmethod
    def draw(self, frame: np.ndarray, pose: Pose) -> np.ndarray:
        """Return a copy of the frame with the pose drawn on it."""
        ...

    # ---- concrete: written once, shared by all implementations ----

    def show(self, frame: np.ndarray, pose: Pose, window: str = "pose") -> None:
        """Draw the pose and display it in a window."""
        import cv2

        cv2.imshow(window, self.draw(frame, pose))
        cv2.waitKey(1)

    def close(self) -> None:
        """Release windows / resources. No-op by default; override where needed."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
