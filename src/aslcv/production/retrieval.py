"""Phase 5a — fetch real reference clips. Retrieval, never generation.

Two entry points:
    fetch_reference(id_gloss, extractor)        -> ReferenceClip   (one sign)
    fetch_sequence(gloss_sequence, extractor)    -> ComposedReference (a whole
        Phase 5b GlossSequence's worth, concatenated into one playable video)

Deliberately builds VIDEO concatenation only, not a concatenated POSE-SEQUENCE
grading target. Grading a live continuous attempt against such a target would
require knowing where one sign ends and the next begins in that attempt -- an
unsolved problem (Phase 7's continuous-recognition segmentation; today's live
pipeline only has Phase 2's manual fixed-window + keypress reset). A pose-
sequence target built now would have no consumer until Phase 7 exists, so it
is left out rather than built as unusable scaffolding. Per-sign grading
targets already exist (Phase 4's EmbeddingGrader/DTWGrader reference banks).

Concatenation never uses a generative model to smooth the join between clips
-- CLAUDE.md's non-negotiable: "Retrieve reference video, never generate it.
No synthesized/avatar signing shown as a model of correct form. Ever." A model
interpolating frames between two real clips would fabricate motion no signer
produced and present it as correct form. Instead each clip is trimmed to its
motion-active span (hand_motion_energy()/motion_active_span(), the same
signal Phase 2's trim_to_motion and Phase 6's live segmenter use) and clips
are hard-cut together -- visibly stitched, honestly so, same framing
scripts/compose_sentence.py's banner already uses.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..extractor.coco_wholebody import COCO_WHOLEBODY
from ..extractor.mediapipe import MEDIAPIPE_HOLISTIC
from ..features import hand_motion_energy, motion_active_span
from ..grading.embedding_dataset import _load_poses_npz
from ..normalizer.shoulder import ShoulderNormalizer
from .gloss_rules import GlossSequence

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "data" / "manifest.csv"
CACHE = REPO / "data" / "cache"

# Trimming needs only the local-hand blocks hand_motion_energy() reads --
# independent of whatever face/legs_feet/confidence/velocity toggles a
# grading FeaturePipeline carries, so this is a fixed, minimal normalizer,
# not something callers configure.
_TRIM_NORMALIZER = ShoulderNormalizer(local_hand=True)
# Same defaults as FeaturePipeline's own trim_to_motion toggle (features.py) --
# reused, not retuned, so a clip trims identically here and in training/eval.
_MOTION_THRESHOLD = 0.02
_MOTION_PAD_FRAMES = 3


def skeleton_for(extractor: str):
    return MEDIAPIPE_HOLISTIC if extractor == "mediapipe" else COCO_WHOLEBODY


def _manifest_rows() -> list[dict]:
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_row(rows: list[dict]) -> dict:
    """Deterministic default clip: prefer a train-split clip (what a
    reference bank actually uses), else the first match. THE one selection
    rule -- previously the same logic independently copy-pasted in
    render_clip.py's pick_row, compose_sentence.py's pick_clip, and
    diagnose_demo.py's reference_row."""
    return next((r for r in rows if r["split"] == "train"), rows[0])


@dataclass
class ReferenceClip:
    """One gloss's resolved reference clip -- the retrieval unit."""

    id_gloss: str
    asllex_code: str
    video_id: str
    video_path: Path
    npz_path: "Path | None"  # None if this extractor hasn't cached this clip
    signer_id: str
    split: str
    n_available: int


def fetch_reference(id_gloss: str, extractor: str = "mediapipe") -> ReferenceClip:
    """The real reference clip for one curriculum sign.

    Raises KeyError if `id_gloss` has no manifest rows at all (unknown sign).
    A known sign whose npz_path comes back None (no cache for THIS extractor)
    is NOT an error here -- retrieval is agnostic to what's graded against;
    callers that need a cached pose sequence (e.g. fetch_sequence's trimming,
    or a live demo that must show something to imitate) are responsible for
    refusing on that, the same fail-closed convention diagnose_demo.py's
    resolve_targets already uses.
    """
    rows = [r for r in _manifest_rows() if r["id_gloss"] == id_gloss]
    if not rows:
        raise KeyError(f"{id_gloss!r} has no manifest rows in {MANIFEST}")
    row = _pick_row(rows)
    npz = CACHE / extractor / f"{row['video_id']}.npz"
    return ReferenceClip(
        id_gloss=id_gloss,
        asllex_code=row["asllex_code"],
        video_id=row["video_id"],
        video_path=REPO / row["video_path"],
        npz_path=npz if npz.exists() else None,
        signer_id=row["signer_id"],
        split=row["split"],
        n_available=len(rows),
    )


@dataclass
class ComposedReference:
    """A whole GlossSequence's worth of reference clips, concatenated."""

    gloss_sequence: GlossSequence
    clips: list[ReferenceClip]
    frames: list[np.ndarray]  # concatenated, trimmed, ready to write/show
    fps: float
    clip_frame_ranges: list[tuple[int, int]] = field(default_factory=list)  # [start, stop) into frames, per clip


def _read_all_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames, fps


def _trimmed_frames(clip: ReferenceClip, skeleton) -> tuple[list[np.ndarray], float]:
    """This clip's video frames, cut to its motion-active span. Requires a
    cached pose sequence to find that span from (hand_motion_energy needs
    poses, not pixels) -- callers must have already fail-closed on
    clip.npz_path being None before calling this."""
    frames, fps = _read_all_frames(clip.video_path)
    poses = _load_poses_npz(clip.npz_path)
    if len(frames) != len(poses):
        raise RuntimeError(
            f"{clip.video_id}: video has {len(frames)} frames but the cached pose "
            f"sequence has {len(poses)} -- extraction is expected to be 1:1 with the "
            f"source video (CLAUDE.md known issue #1); can't trust frame alignment.")
    energy = hand_motion_energy(_TRIM_NORMALIZER, skeleton, poses)
    start, stop = motion_active_span(energy, _MOTION_THRESHOLD, _MOTION_PAD_FRAMES)
    return frames[start:stop], fps


def fetch_sequence(gloss_sequence: GlossSequence, extractor: str = "mediapipe") -> ComposedReference:
    """Resolve every gloss in an in-scope GlossSequence to a reference clip,
    trim each to its motion-active span, and hard-cut them together into one
    ordered frame sequence -- retrieval + concatenation, never generation
    (see module docstring). Fail-closed: refuses (raises) rather than return
    a partial composition if the sequence itself is out of scope, or if any
    gloss has no cached reference clip for `extractor`.
    """
    if not gloss_sequence.in_scope:
        raise ValueError(
            f"refusing to compose an out-of-scope GlossSequence: {gloss_sequence.reason}")
    if not gloss_sequence.glosses:
        raise ValueError("refusing to compose an empty GlossSequence")

    skeleton = skeleton_for(extractor)
    clips: list[ReferenceClip] = []
    missing: list[str] = []
    for id_gloss in gloss_sequence.gloss_ids:
        try:
            clip = fetch_reference(id_gloss, extractor=extractor)
        except KeyError:
            missing.append(f"{id_gloss}: not a known sign")
            continue
        if clip.npz_path is None:
            missing.append(f"{id_gloss}: no cached reference for extractor={extractor!r}")
            continue
        clips.append(clip)
    if missing:
        raise ValueError(
            "refusing to compose -- missing reference clip(s) for: " + "; ".join(missing))

    all_frames: list[np.ndarray] = []
    clip_frame_ranges: list[tuple[int, int]] = []
    fps = 30.0
    for clip in clips:
        trimmed, clip_fps = _trimmed_frames(clip, skeleton)
        fps = clip_fps  # last clip's fps wins; ASL Citizen clips share fps in practice
        start = len(all_frames)
        all_frames.extend(trimmed)
        clip_frame_ranges.append((start, len(all_frames)))

    return ComposedReference(
        gloss_sequence=gloss_sequence, clips=clips, frames=all_frames,
        fps=fps, clip_frame_ranges=clip_frame_ranges,
    )


def write_composed_video(composed: ComposedReference, out_path: Path) -> Path:
    """Write a ComposedReference's frames to an mp4. Separate from
    fetch_sequence deliberately -- composing is cheap and callers (e.g. a
    future Phase 6 presenter) may want the frames in memory without always
    paying for a disk write."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not composed.frames:
        raise ValueError("refusing to write an empty composed video")
    h, w = composed.frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), composed.fps, (w, h))
    try:
        for frame in composed.frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        writer.release()
    return out_path
