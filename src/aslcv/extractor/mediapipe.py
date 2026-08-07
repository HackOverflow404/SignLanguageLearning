import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from .base import Extractor, Pose, RunningMode, Skeleton

# Common glvnd vendor JSON locations for the NVIDIA proprietary driver across
# distros. Benchmarked (see project_workflow.md Phase 3 "Loose end C" /
# scripts/benchmark_extractors.py): requesting BaseOptions.Delegate.GPU alone
# does NOT get MediaPipe onto real hardware on a Linux desktop where its
# default EGL context resolution can silently pick Mesa's llvmpipe SOFTWARE
# renderer instead -- confirmed via the GL context log line, and ~4x SLOWER
# than the CPU delegate, not faster. Forcing the vendor explicitly fixes it:
# hand landmarker alone 173fps vs 49.5fps CPU; the full 3-model pipeline
# 53.0fps vs 16.1fps CPU. Best-effort only -- if none of these paths exist
# (different distro, no NVIDIA driver, a laptop with no discrete GPU at all),
# this does nothing and the GPU delegate attempt below still has its own
# try/except fallback to CPU.
_NVIDIA_EGL_VENDOR_PATHS = (
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    "/etc/glvnd/egl_vendor.d/10_nvidia.json",
    "/usr/share/egl/egl_external_platform.d/10_nvidia.json",
)


def _maybe_force_nvidia_egl():
    """Best-effort: point glvnd at the NVIDIA EGL vendor and force the X11 EGL
    platform, but only if the caller hasn't already set an opinion (respects
    an explicit environment override) and only if a known vendor JSON is
    actually present on this machine. Idempotent; safe to call more than
    once. Must run before any GPU-delegate landmarker is constructed --
    mediapipe resolves the EGL vendor lazily on first use, not at import
    time, so setting these here (not just documenting them) is sufficient."""
    if "__EGL_VENDOR_LIBRARY_FILENAMES" in os.environ:
        return
    for path in _NVIDIA_EGL_VENDOR_PATHS:
        if os.path.exists(path):
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = path
            os.environ.setdefault("EGL_PLATFORM", "x11")
            return

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
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
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
    "left_elbow": 13,
    "right_elbow": 14,
    "left_body_wrist": 15,   # arm-chain wrist -- NOT left_hand_wrist (the hand-block
    "right_body_wrist": 16,  # root, added below); see coco_wholebody.py's comment.
    "left_hip": 23,
    "right_hip": 24,
}

# BlazePose's 33 pose landmarks split by semantic block (canonical cross-topology
# REGIONS, normalizer.base.REQUIRED_REGIONS):
#   0-10   face/head, coarse. FIVE of these are the same coarse head reference
#          COCO-WholeBody carries in ITS body_upper (nose=0, left_eye=2,
#          right_eye=5, left_ear=7, right_ear=8 -- BlazePose's single "eye"/"ear"
#          landmark per side, as opposed to the inner/outer variants) and join
#          body_upper here too, so --face=False never removes coarse head
#          location on EITHER topology -- see coco_wholebody.py's _BODY_UPPER_NAMES
#          for the mirrored COCO side of this. The remaining six (eye_inner/outer
#          x2 sides, mouth_left/right -- extra granularity COCO's 17-point body
#          block doesn't have) stay bundled with the 478-point face mesh, since
#          both are genuinely fine facial detail with no COCO body_upper analog.
#          PRIOR BEHAVIOR (do not reintroduce): all of 0-10 used to fold into
#          "face" together, so --face=False silently dropped MediaPipe's nose/
#          eyes/ears while COCO's stayed -- a real cross-backend confound in any
#          face=False comparison (COCO's global block always carried head
#          location; MediaPipe's did not). Confirmed and fixed together with
#          scripts/eval_minimal_pairs.py's --confidence sweep.
#   11-22  upper body + arms -- split further into body_upper (11,12: shoulders,
#          the torso reference) and arms (13-22: elbow, wrist, and the pose
#          model's own pinky/index/thumb tips -- NOT the dedicated hand-model
#          points, which live in left_hand/right_hand below).
#   23-32  legs/feet (hips, knees, ankles, heels, foot index) -- dropped by
#          default in features.py; see REQUIRED_REGIONS.
_POSE_BODY_UPPER_HEAD_LANDMARKS = (0, 2, 5, 7, 8)  # nose, left_eye, right_eye, left_ear, right_ear
_POSE_FACE_FINE_LANDMARKS = (1, 3, 4, 6, 9, 10)  # eye_inner/outer x2, mouth_left/right
_POSE_BODY_UPPER_LANDMARKS = (11, 12) + _POSE_BODY_UPPER_HEAD_LANDMARKS
_POSE_ARM_LANDMARKS = tuple(range(13, 23))
_POSE_LEG_LANDMARKS = tuple(range(23, 33))


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
        + offset_edges(
            vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS, _FACE_OFFSET
        )
        + offset_edges(
            vision.HandLandmarksConnections.HAND_CONNECTIONS, _LEFT_HAND_OFFSET
        )
        + offset_edges(
            vision.HandLandmarksConnections.HAND_CONNECTIONS, _RIGHT_HAND_OFFSET
        )
    )

    # Pose anchors + per-hand anchors. MediaPipe hand landmarks (BlazeHand) index
    # the wrist at 0 and the middle-finger MCP (base knuckle) at 9. These use the
    # same NEUTRAL names as the COCO skeleton (normalizer.base.REQUIRED_ANCHORS) so
    # one normalizer serves both topologies; here they map to the hand block's points.
    anchors = tuple(
        (name, _POSE_OFFSET + landmark)
        for name, landmark in _POSE_ANCHOR_LANDMARKS.items()
    ) + (
        ("left_hand_wrist", _LEFT_HAND_OFFSET + 0),
        ("left_hand_middle_mcp", _LEFT_HAND_OFFSET + 9),
        ("right_hand_wrist", _RIGHT_HAND_OFFSET + 0),
        ("right_hand_middle_mcp", _RIGHT_HAND_OFFSET + 9),
    )

    regions = (
        ("body_upper", tuple(_POSE_OFFSET + i for i in _POSE_BODY_UPPER_LANDMARKS)),
        ("arms", tuple(_POSE_OFFSET + i for i in _POSE_ARM_LANDMARKS)),
        ("left_hand", tuple(range(_LEFT_HAND_OFFSET, _LEFT_HAND_OFFSET + HAND_LM_COUNT))),
        ("right_hand", tuple(range(_RIGHT_HAND_OFFSET, _RIGHT_HAND_OFFSET + HAND_LM_COUNT))),
        ("face", tuple(_POSE_OFFSET + i for i in _POSE_FACE_FINE_LANDMARKS)
                 + tuple(range(_FACE_OFFSET, _FACE_OFFSET + FACE_LM_COUNT))),
        ("legs_feet", tuple(_POSE_OFFSET + i for i in _POSE_LEG_LANDMARKS)),
    )

    return Skeleton(names=tuple(names), edges=tuple(edges), anchors=anchors, regions=regions)


MEDIAPIPE_HOLISTIC = _build_skeleton()


def _fill_segment(
    keypoints,
    scores,
    offset,
    landmarks,
    width,
    height,
    use_visibility=False,
    det_score=1.0,
):
    """Write one landmark group into the combined arrays.

    Presence is carried by `scores`: undetected groups are never passed here, so
    they keep the 0.0 the array was initialised with. `det_score` is the group's
    detection confidence, stored on every point when there is no per-point score
    (hands: the handedness score; face: 1.0, since the face landmarker exposes
    none). Pose overrides per point via `use_visibility`.
    """
    for i, landmark in enumerate(landmarks):
        keypoints[offset + i] = (landmark.x * width, landmark.y * height)
        visibility = landmark.visibility if use_visibility else None
        scores[offset + i] = visibility if visibility is not None else det_score


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

    Delegate (CPU/GPU)
    -------------------
    `delegate="cpu"` (default) matches every clip in the existing cache --
    kept as the default so nothing about the 1,874-clip cache's provenance or
    a caller's expectations changes unless GPU is asked for explicitly.
    `delegate="gpu"` requests `BaseOptions.Delegate.GPU` on all three
    landmarkers for a measured ~3.3x speedup (see the module-level comment on
    `_maybe_force_nvidia_egl` for how and why), with an automatic, logged
    fallback to CPU if GPU landmarker construction raises (e.g. no compatible
    GPU/driver on this machine) -- never a hard failure just because GPU
    wasn't available. `self.delegate_used` records which one actually ran,
    since "GPU was requested" and "GPU was used" are not the same claim.
    """

    def __init__(
        self,
        mirrored: bool,
        running_mode: RunningMode = RunningMode.IMAGE,
        num_faces: int = 1,
        num_hands: int = 2,
        delegate: str = "cpu",
    ):
        super().__init__(running_mode)
        self._mirrored = mirrored
        if delegate not in ("cpu", "gpu"):
            raise ValueError(f"delegate must be 'cpu' or 'gpu', got {delegate!r}")

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

        def build(mp_delegate):
            pose_options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH, delegate=mp_delegate),
                running_mode=mp_mode,
            )
            face_options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH, delegate=mp_delegate),
                running_mode=mp_mode,
                num_faces=num_faces,
                output_face_blendshapes=True,  # 52 semantic coeffs for NMM / facial grammar
            )
            hand_options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH, delegate=mp_delegate),
                running_mode=mp_mode,
                num_hands=num_hands,
            )
            if is_live:
                pose_options.result_callback = self._on_pose_result
                face_options.result_callback = self._on_face_result
                hand_options.result_callback = self._on_hands_result
            return (
                vision.PoseLandmarker.create_from_options(pose_options),
                vision.FaceLandmarker.create_from_options(face_options),
                vision.HandLandmarker.create_from_options(hand_options),
            )

        self.delegate_used = delegate
        if delegate == "gpu":
            _maybe_force_nvidia_egl()
            try:
                self._pose, self._face, self._hands = build(BaseOptions.Delegate.GPU)
            except Exception as e:
                print(f"MediaPipePoseExtractor: GPU delegate failed ({e!r}), falling back to CPU",
                      flush=True)
                self.delegate_used = "cpu"
                self._pose, self._face, self._hands = build(BaseOptions.Delegate.CPU)
        else:
            self._pose, self._face, self._hands = build(BaseOptions.Delegate.CPU)

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
                keypoints,
                scores,
                _POSE_OFFSET,
                pose_result.pose_landmarks[0],
                width,
                height,
                use_visibility=True,
            )

        if has_face:
            _fill_segment(
                keypoints,
                scores,
                _FACE_OFFSET,
                face_result.face_landmarks[0],
                width,
                height,
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
            # Store the per-hand detection confidence (graded per hand, not per
            # point) instead of a blind 1.0, so a barely-detected hand reads lower
            # than a crisp one. Face has no such score and stays 1.0.
            _fill_segment(
                keypoints,
                scores,
                offset,
                landmarks,
                width,
                height,
                det_score=category.score,
            )

        # Face blendshapes: (52,) graded coefficients aligned to BLENDSHAPE_NAMES,
        # or None when no face was detected this frame.
        blendshapes = None
        if has_face and face_result.face_blendshapes:
            blendshapes = np.array(
                [c.score for c in face_result.face_blendshapes[0]], dtype=np.float32
            )

        return Pose(
            keypoints=keypoints,
            scores=scores,
            width=width,
            height=height,
            blendshapes=blendshapes,
        )

    def draw(self, frame: np.ndarray, pose: Pose, threshold: float = 0.0) -> np.ndarray:
        annotated = frame.copy()
        # Draw a point iff it is present AND at/above `threshold`. Presence is
        # `score > 0`: undetected groups keep the 0.0 from array init, while every
        # detected point -- even one landing at pixel (0,0) -- has a positive score
        # (do NOT test coords==(0,0); a real origin point would be dropped).
        # `threshold` additionally hides low-visibility pose points; 0.0 shows all
        # detected points, raise it (e.g. 0.5) to declutter.
        confident = (pose.scores > 0.0) & pose.confident(threshold)

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
