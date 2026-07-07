from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Skeleton:
    """Describes a model's keypoint layout so downstream code stays model-agnostic."""

    names: tuple[str, ...]  # keypoint name per index, e.g. ("nose", "left_wrist", ...)
    edges: tuple[tuple[int, int], ...]  # index pairs to connect when drawing


@dataclass
class Pose:
    """A single detected pose. keypoints: (N, 2) pixel coords; scores: (N,) confidence."""

    keypoints: np.ndarray
    scores: np.ndarray

    def confident(self, threshold: float = 0.5) -> np.ndarray:
        """Boolean mask of keypoints at or above the confidence threshold."""
        return self.scores >= threshold


class PoseExtractor(ABC):
    """Interface for pose extractors that turn a frame into a Pose."""

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
