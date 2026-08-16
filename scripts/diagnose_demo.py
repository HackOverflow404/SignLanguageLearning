#!/usr/bin/env python3
"""diagnose_demo.py -- Phase 6: the adaptive loop, live, over Phase 4's
grade_against(attempt, target).

Prompts a target sign, plays that sign's real reference clip on loop next to the live
webcam view, captures your attempt, and shows the PER-PARAMETER diagnosis from
EmbeddingGrader.grade_against_poses -- handshape / major_location / minor_location /
movement / repeated_movement, each MATCH/OFF/insufficient-data with the model's
confidence, plus overall fidelity. `c` clears the capture and re-tries the SAME target
(the last verdict stays on screen, dimmed, so you can see what changed). `n` is the
adaptive step: it records the current (non-stale) verdict into a persisted per-sign/
per-parameter mastery model (aslcv.learner.mastery), prints one line of coaching
(aslcv.generator.feedback, or an LLM phrasing under --llm-feedback) naming the
parameter most confidently wrong, and picks the next target itself
(aslcv.learner.scheduler) -- gap-targeting + a light recency bias, or an automatic
contrastive minimal-pair drill if the mistake has a real partner sign in the current
pool (e.g. missing minor_location on `father` queues `mother` next).

CAPTURE, not a fixed sliding window (aslcv.capture.CaptureBuffer): a fixed trailing
window silently evicts its oldest frames as new ones arrive, so a signer slower than
the window truncates their own attempt -- exactly the kind of corruption that hits
`repeated_movement` hardest (it needs the full cyclic pattern) and can catch
`handshape` mid-transition. CaptureBuffer instead grows from the start of motion to a
natural rest boundary (the SAME hand_motion_energy signal features.py's trim_to_motion
already uses for cached clips), capped for safety -- READY -> CAPTURING -> CAPTURED,
shown on screen so you know whether a verdict reflects a complete attempt.

Two always-on, fully grounded (no LLM) text descriptions, sourced from ASL-LEX/
curriculum phonology data, never invented: `aslcv.generator.sign_description` shows
what the TARGET sign's handshape/location/movement actually are, at the bottom of the
reference video; `aslcv.generator.feedback`/`handshape_descriptions` show what a WRONG
attempt's parameter actually looked like vs. the target, in the coaching line.

HONEST LIMITS (also shown on screen, every frame): this confirms the plumbing and
lets you practice imitating a real reference clip. It does NOT independently verify
ASL correctness -- "all correct" means "matched the reference," not "fluent," and the
underlying model's own measured accuracy (PHASE4_REPORT.md) is well short of perfect
(handshape 80.7%, repeated_movement 82.1% on held-out val clips) -- a wrong verdict is
often the model, not you. A target with no cached reference clip is refused outright
(fail-closed), never graded without something on screen to imitate.

Two optional HuggingFace-hosted-API upgrades, both opt-in and fail-open (need
HF_TOKEN -- see .env.example -- and fall back to nothing/templated-text silently
on any missing token/network/API failure): `--llm-feedback` phrases the coaching
line more naturally; `--sentence-prompts` shows an LLM-written example sentence
using each target's word, gloss-composed and accepted by the fail-closed rule
engine before ever being displayed (aslcv.generator.sentence_prompts) --
presentational only, still not graded (continuous-sentence grading is Phase 7).

    .venv/bin/python scripts/diagnose_demo.py                       # mediapipe, adaptive over the default pool
    .venv/bin/python scripts/diagnose_demo.py --target father        # start on father, [n] still adapts onward
    .venv/bin/python scripts/diagnose_demo.py --targets you,me,water
    .venv/bin/python scripts/diagnose_demo.py --sentence-prompts     # + LLM example sentences per target
    .venv/bin/python scripts/diagnose_demo.py --selftest             # no camera: verify the whole path offline

Phase 7 -- SENTENCE MODE (`--sentence`): grades a full sentence via forced
alignment (aslcv.grading.alignment.align_and_grade) instead of one isolated
sign -- the target gloss sequence is always known (Phase 5b's rule engine
glossed it), so this is forced alignment against a known reference, not open
continuous recognition (see project_workflow.md's Phase 7 section for the
full reasoning and validation). ONE continuous capture per attempt (no
background re-grading -- alignment needs the complete attempt); `[n]` grades
it once CAPTURED, updates mastery per word, and coaches the first word with a
real mistake. Not adaptive across sentences (grades exactly the one sentence
given; picking what to practice next is out of this scope).

    .venv/bin/python scripts/diagnose_demo.py --sentence "I want water."
    .venv/bin/python scripts/diagnose_demo.py --selftest --sentence "I want water."  # no camera
"""
import argparse
import csv
import functools
import textwrap
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from aslcv.capture import CaptureBuffer
from aslcv.extractor.base import Pose, RunningMode
from aslcv.extractor.coco_wholebody import COCO_WHOLEBODY
from aslcv.extractor.mediapipe import MEDIAPIPE_HOLISTIC
from aslcv.features import hand_motion_energy
from aslcv.generator.feedback import focus_parameter, readable_value
from aslcv.generator.llm_feedback import coach_text_maybe_llm
from aslcv.generator.sentence_prompts import sentence_prompt_maybe_llm
from aslcv.generator.sign_description import describe_sign
from aslcv.grading.alignment import align_and_grade
from aslcv.grading.embedding_grader import EmbeddingGrader
from aslcv.grading.phonology_labels import ALL_PARAMETERS, PhonologyLabels
from aslcv.learner.mastery import MasteryState
from aslcv.learner.scheduler import find_minimal_pairs, pick_next
from aslcv.pipeline_config import add_pipeline_args, build_pipeline
from aslcv.production import GlossRuleEngine, fetch_sequence
from live_demo import _poses_from_npz, build_extractor  # reuse, don't rebuild

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifest.csv"
DEFAULT_CHECKPOINT = REPO / "models" / "embedding_grader"
DEFAULT_MASTERY_PATH = REPO / "data" / "learner_state.json"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# A small, phonologically clear starting set -- includes the curriculum's built-in
# mother/father minimal pair (differ ONLY in minor_location). [n] cycles this list;
# --target/--targets override it (see resolve_targets).
DEFAULT_TARGETS = ["mother", "father", "you", "me", "water", "thank_you", "yes", "good"]

DISCLAIMER = ("PRACTICE AID -- confirms plumbing + match to a real reference clip, "
              "does NOT independently verify ASL correctness. The model itself is imperfect "
              "(see --help): a wrong verdict is often the model, not you.")

PARAM_LABEL = {
    "handshape": "Handshape", "major_location": "Major location", "minor_location": "Minor location",
    "movement": "Movement", "repeated_movement": "Repeated",
}

# ---------------------------------------------------------------------- theme ----
# One consistent palette/type scale for every drawn panel, instead of each function
# picking its own colors/sizes -- BGR (cv2 convention), a soft charcoal rather than
# pure black so long viewing is easier on the eyes. Text that sits over the LIVE
# camera feed always gets a near-opaque panel behind it first (see _panel_bg calls
# below) -- colored text directly over a busy, brightly-lit camera image is close to
# unreadable regardless of how saturated the color is.
BG_HEADER = (22, 20, 18)
BG_PANEL = (36, 33, 30)
DIVIDER = (64, 60, 55)
ACCENT = (50, 172, 250)           # warm amber-gold header/title accent
TEXT_PRIMARY = (245, 245, 245)
TEXT_SECONDARY = (195, 195, 195)
TEXT_MUTED = (145, 145, 145)
MATCH_COLOR = (70, 220, 90)        # saturated green
OFF_COLOR = (60, 75, 255)          # saturated red
INSUFFICIENT_COLOR = (190, 190, 190)
SENTENCE_COLOR = (150, 205, 255)
GLOSS_COLOR = (110, 220, 150)
COACH_COLOR = (110, 190, 255)
READY_COLOR = (165, 165, 165)
CAPTURING_COLOR = (50, 190, 255)
CAPTURED_COLOR = (70, 220, 90)

# Bumped up across the board -- the previous scale (0.4-0.72) was too small to read
# comfortably at a normal viewing distance. Bold (thickness=2) is used for anything
# that needs to pop against a busy background, not just a bigger scale.
F_TITLE, F_HEADER, F_BODY, F_SMALL, F_TINY = 1.0, 0.8, 0.68, 0.58, 0.52


def _text(canvas, s, x, y, scale, color, thickness=1):
    cv2.putText(canvas, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _panel_bg(canvas, y0, y1, color=BG_PANEL, alpha=0.93):
    y0, y1 = max(0, int(y0)), min(canvas.shape[0], int(y1))
    if y1 <= y0:
        return
    band = canvas.copy()
    cv2.rectangle(band, (0, y0), (canvas.shape[1], y1), color, -1)
    cv2.addWeighted(band, alpha, canvas, 1 - alpha, 0, canvas)


def _divider(canvas, y):
    cv2.line(canvas, (0, y), (canvas.shape[1], y), DIVIDER, 1, cv2.LINE_AA)


def _wrap_width(canvas, x, scale):
    """Character budget for HERSHEY_SIMPLEX at `scale`, given the available
    width from `x` to the canvas edge. cv2.putText does not wrap OR clip --
    an overlong line just silently runs off the canvas -- so this constant
    (~14.2px/char at scale=1, measured empirically via cv2.getTextSize; do
    not guess this number) has to be a real upper bound, not an estimate."""
    return max(10, int((canvas.shape[1] - x - 10) / (14.2 * scale)))


def _wrapped(canvas, lines_and_colors, x, y_top, scale, line_h, thickness=1):
    """Draws (text, color) pairs, wrapping each text to the canvas width at
    `scale`, returning the y just past the last drawn line."""
    max_chars = _wrap_width(canvas, x, scale)
    y = y_top
    for text, color in lines_and_colors:
        for line in (textwrap.wrap(text, width=max_chars) or [""]):
            _text(canvas, line, x, y, scale, color, thickness)
            y += line_h
    return y


def _wrapped_line_count(canvas, texts, x, scale):
    max_chars = _wrap_width(canvas, x, scale)
    return sum(len(textwrap.wrap(t, width=max_chars) or [""]) for t in texts)


# ------------------------------------------------------------- manifest / clips ----

def manifest_rows():
    return list(csv.DictReader(open(MANIFEST)))


def reference_row(rows, sign):
    """Manifest row to show as the reference clip for `sign` -- prefer a train-split
    clip (same convention as render_clip.py / compose_sentence.py), else the first."""
    matches = [r for r in rows if r["id_gloss"] == sign]
    if not matches:
        return None
    return next((r for r in matches if r["split"] == "train"), matches[0])


def load_reference_frames(video_path, max_width=440):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(max_width * h / w)))
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"reference video decoded 0 frames: {video_path}")
    return frames


class ReferenceLoop:
    """The current target's reference clip, decoded ONCE and looped continuously.

    Mandatory alongside the live view, never optional: a learner can only correct
    toward a reference they can see, and this demo must never imply it can judge
    correctness without one on screen (see resolve_targets' fail-closed check)."""

    def __init__(self):
        self.sign = None
        self.video_id = None
        self.frames: list = []
        self.idx = 0

    def set_target(self, sign, video_id, frames):
        self.sign, self.video_id, self.frames, self.idx = sign, video_id, frames, 0

    def next_frame(self):
        if not self.frames:
            return None
        f = self.frames[self.idx % len(self.frames)]
        self.idx += 1
        return f


def resolve_targets(args, rows, grader):
    """The ordered [n]-cycle list, each entry pre-validated to have a real cached
    reference clip. Refuses to start rather than let a target reach the live loop
    with nothing to show on screen (fail-closed, same spirit as the rest of the
    project's out-of-scope refusals)."""
    targets = args.targets.split(",") if args.targets else list(DEFAULT_TARGETS)
    if args.target:
        targets = [args.target] + [t for t in targets if t != args.target]

    resolved = []
    for sign in targets:
        if sign not in grader.signs:
            raise SystemExit(f"target {sign!r} is not a sign this checkpoint knows; have {grader.signs}")
        row = reference_row(rows, sign)
        video_path = REPO / row["video_path"] if row else None
        if row is None or not video_path.exists():
            raise SystemExit(
                f"no cached reference VIDEO for target {sign!r} -- refusing to grade against it "
                f"without a reference to imitate on screen. Drop it from --targets/--target, or "
                f"fix data/manifest.csv.")
        resolved.append((sign, row))
    return resolved


def resolve_sentence(args, grader):
    """Phase 7: --sentence's gloss sequence + composed reference video,
    pre-validated the same fail-closed way resolve_targets() validates the
    single-sign cycle -- refuses to reach the live loop with an out-of-scope
    sentence or a gloss missing a cached reference clip."""
    engine = GlossRuleEngine()
    seq = engine.gloss(args.sentence)
    if not seq.in_scope:
        raise SystemExit(f"sentence refused by the gloss engine: {seq.reason}")
    try:
        composed_video = fetch_sequence(seq, extractor=grader.extractor)
    except ValueError as exc:
        raise SystemExit(f"cannot resolve reference clips for this sentence: {exc}")
    return seq, composed_video


# ------------------------------------------------------------- config verification --

_PIPELINE_FIELDS = ("face", "legs_feet", "confidence", "binary_threshold", "velocity",
                    "depth_proxies", "trim_to_motion", "motion_threshold", "motion_pad_frames")


def verify_pipeline_matches_checkpoint(args, grader, skeleton):
    """Build the live-side FeaturePipeline from CLI flags (prints the resolved config,
    same as every other script routes through pipeline_config.py) and refuse to start
    if it differs from the checkpoint's own TRAINING-TIME config in any field. A silent
    mismatch here would corrupt every verdict without any visible symptom -- fail
    closed instead of guessing."""
    live_pipeline = build_pipeline(args, skeleton, extractor_name=args.extractor)
    saved = grader.pipeline

    mismatches = []
    if args.extractor != grader.extractor:
        mismatches.append(f"extractor: live={args.extractor!r} checkpoint={grader.extractor!r}")
    for field in _PIPELINE_FIELDS:
        lv, sv = getattr(live_pipeline, field), getattr(saved, field)
        if lv != sv:
            mismatches.append(f"{field}: live={lv} checkpoint={sv}")
    if live_pipeline.normalizer.local_hand != saved.normalizer.local_hand:
        mismatches.append(f"local_hand: live={live_pipeline.normalizer.local_hand} "
                           f"checkpoint={saved.normalizer.local_hand}")

    if mismatches:
        raise SystemExit(
            "REFUSING TO START: live feature pipeline does not match this checkpoint's "
            "training-time config -- every verdict would silently be wrong.\n  "
            + "\n  ".join(mismatches)
            + "\nDrop the overriding flag(s), or point --checkpoint at a model trained with this config.")
    print("verified: live pipeline config matches the checkpoint's training-time config.")


# ---------------------------------------------------------------------- overlay ----

def _tag_and_color(verdict):
    if verdict.correct is True:
        return "MATCH", MATCH_COLOR
    if verdict.correct is False:
        return "OFF", OFF_COLOR
    return "insufficient data", INSUFFICIENT_COLOR


_CAPTURE_STATE_DISPLAY = {
    "idle": ("READY", READY_COLOR, "position yourself, then start signing"),
    "active": ("CAPTURING...", CAPTURING_COLOR, "keep signing -- pause briefly when done"),
    "settled": ("CAPTURED", CAPTURED_COLOR, "attempt complete -- [n] to record, [c] to retry"),
}


HEADER_H = 58
BADGE_H = 76
ROW_H = 78  # per-parameter: label+tag line (34) + detail line (34) + gap (10)


def draw_verdict(canvas, result, target_sign, stale, capture_state, mastery=None):
    """Draws the target header, capture-state badge, and (if `result`) the
    fidelity + per-parameter verdict list, each behind its own solid panel --
    this all sits on top of the LIVE camera feed, and colored text directly
    over a busy, brightly-lit image is close to unreadable no matter the
    color, so nothing here is drawn without an opaque backing first. Returns
    content_bottom -- everything below this point is drawn TOP-DOWN from
    there by the caller (coach text, then the status bar), never at a fixed
    offset from the canvas bottom, so a taller verdict block can never
    silently overlap what comes after it (a real bug an earlier version of
    this file had: fixed offsets from both ends of the canvas can overlap in
    the middle if the content in between is taller than assumed)."""
    w = canvas.shape[1]

    _panel_bg(canvas, 0, HEADER_H, color=BG_HEADER, alpha=0.96)
    title = f"TARGET   {target_sign}"
    if mastery is not None:
        title += f"     mastery {mastery.sign_mastery(target_sign):.0%}"
    _text(canvas, title, 18, 40, F_HEADER, ACCENT, 2)
    _divider(canvas, HEADER_H)

    badge_top = HEADER_H
    _panel_bg(canvas, badge_top, badge_top + BADGE_H, color=BG_HEADER, alpha=0.93)
    label, color, hint = _CAPTURE_STATE_DISPLAY[capture_state]
    cv2.circle(canvas, (32, badge_top + 30), 9, color, -1, cv2.LINE_AA)
    _text(canvas, label, 54, badge_top + 38, F_BODY, color, 2)
    _text(canvas, hint, 54, badge_top + 62, F_TINY, TEXT_MUTED)
    _divider(canvas, badge_top + BADGE_H)

    content_top = badge_top + BADGE_H
    if result is None:
        return content_top

    content_h = 56 + ROW_H * len(ALL_PARAMETERS)
    _panel_bg(canvas, content_top, content_top + content_h, color=BG_PANEL, alpha=0.94)

    y = content_top + 44
    fid_color = TEXT_MUTED if stale else TEXT_PRIMARY
    suffix = "  (previous attempt -- re-signing)" if stale else ""
    _text(canvas, f"Fidelity {result.fidelity:.3f}{suffix}", 20, y, F_BODY, fid_color, 2)
    y += 48
    for p in ALL_PARAMETERS:
        v = result.parameters[p]
        tag, color = _tag_and_color(v)
        if stale:
            color = tuple(c // 2 for c in color)
        predicted = readable_value(p, v.predicted)
        cv2.circle(canvas, (26, y - 8), 6, color, -1, cv2.LINE_AA)
        _text(canvas, PARAM_LABEL[p], 46, y, F_BODY, TEXT_PRIMARY if not stale else TEXT_MUTED, 2)
        tag_text = f"{tag}   {v.confidence:.0%}"
        (tag_w, _), _ = cv2.getTextSize(tag_text, FONT, F_SMALL, 2)
        _text(canvas, tag_text, w - 22 - tag_w, y, F_SMALL, color, 2)
        y += 32
        if v.correct is False:
            detail = f"you: {predicted}    target: {readable_value(p, v.target)}"
        else:
            detail = f"signed: {predicted}"
        _text(canvas, detail, 46, y, F_SMALL, color, 1)
        y += ROW_H - 32
    return content_top + content_h


def draw_status_bar(canvas, y_top, fps, grade_ms):
    """Status/disclaimer bar, drawn starting exactly at `y_top` (wherever the
    content above it actually ended) and extending to the canvas bottom --
    never pinned to a fixed offset from the bottom, so it can't be squeezed
    or overlapped if the content above it turned out taller than expected."""
    h = canvas.shape[0]
    disclaimer_lines = textwrap.wrap(DISCLAIMER, width=_wrap_width(canvas, 18, F_TINY))
    _panel_bg(canvas, y_top, h, color=BG_HEADER, alpha=0.93)
    _divider(canvas, y_top)
    status = f"{fps:.0f} fps   grade {grade_ms:.0f} ms   [q]uit   [c]lear   [n]ext"
    _text(canvas, status, 18, y_top + 28, F_TINY, TEXT_SECONDARY)
    for i, line in enumerate(disclaimer_lines):
        _text(canvas, line, 18, y_top + 28 + (i + 1) * 26, F_TINY, TEXT_MUTED)


def _reference_footer_height(canvas, description, sentence_lines):
    """Pre-measures draw_reference_footer's content height WITHOUT drawing --
    lets compose_canvas anchor the footer's top from the actual content size
    instead of a guessed fixed offset (a guess too small would clip the
    panel background while the text still drew past it, right back onto the
    raw video -- the exact bug this whole redesign exists to fix)."""
    desc_lines_n = _wrapped_line_count(canvas, [description], 18, F_SMALL)
    band_h = 24 + 34 + desc_lines_n * 30
    if sentence_lines:
        sentence_lines_n = _wrapped_line_count(canvas, sentence_lines, 18, F_SMALL)
        band_h += 10 + sentence_lines_n * 30 + 20
    return band_h


def draw_reference_footer(canvas, y_top, band_h, description, sentence_lines):
    """Bottom-of-reference-video panel, drawn as ONE continuous solid card (no
    gaps of raw video peeking through between sub-sections): the ALWAYS-ON
    grounded, natural-language description of what the target sign's
    phonology actually is (sign_description.describe_sign -- no LLM, no
    network, works even with --sentence-prompts off), plus the optional LLM
    sentence-prompt example above it when available."""
    desc_line_h, heading_h = 30, 34
    _panel_bg(canvas, y_top, y_top + band_h, color=BG_PANEL, alpha=0.94)

    y = y_top + 18
    if sentence_lines:
        y = _wrapped(canvas, [(sentence_lines[0], SENTENCE_COLOR), (sentence_lines[1], GLOSS_COLOR)],
                     18, y + desc_line_h - 10, F_SMALL, desc_line_h, thickness=2) + 4
        _divider(canvas, y)
        y += 18

    _text(canvas, "WHAT THE CORRECT SIGN LOOKS LIKE", 18, y + 6, F_TINY, TEXT_MUTED, 1)
    _wrapped(canvas, [(description, TEXT_PRIMARY)], 18, y + heading_h, F_SMALL, desc_line_h, thickness=2)


def draw_coach_text(canvas, text, sign, y_top):
    """Draws the coaching line from the attempt just recorded on the
    PREVIOUS [n] press (templated coach_text, or an LLM phrasing of the same
    facts under --llm-feedback), starting exactly at `y_top`. Labeled with
    which sign it was about, since the target has usually already moved on
    by the time this draws (mirrors the same "show what changed" spirit as
    the dimmed stale verdict). Returns `y_top` unchanged if there's no
    coaching text yet (a fresh session, or before the first attempt has been
    submitted with [n]) -- never draws a band for nothing."""
    if text is None:
        return y_top
    full = f"Coach ({sign}): {text}"
    lines_n = _wrapped_line_count(canvas, [full], 18, F_SMALL)
    line_h = 30
    band_h = 20 + 14 + lines_n * line_h
    _panel_bg(canvas, y_top, y_top + band_h, color=BG_PANEL, alpha=0.94)
    _wrapped(canvas, [(full, COACH_COLOR)], 18, y_top + 14 + line_h, F_SMALL, line_h, thickness=2)
    _divider(canvas, y_top)
    return y_top + band_h


def compose_canvas(ref_frame, live_frame, target_sign, phon_labels, result, stale, capture_state,
                    fps, ms, mastery=None, sentence_prompt=None, coach_text=None, coach_text_for=None,
                    height=920):
    def fit(frame, fallback_w=460):
        if frame is None:
            return np.full((height, fallback_w, 3), 18, np.uint8)
        h, w = frame.shape[:2]
        scale = height / h
        return cv2.resize(frame, (max(1, int(w * scale)), height))

    left = fit(ref_frame)
    right = fit(live_frame)

    _panel_bg(left, 0, HEADER_H, color=BG_HEADER, alpha=0.96)
    _text(left, f"REFERENCE   {target_sign}", 18, 40, F_HEADER, ACCENT, 2)
    _divider(left, HEADER_H)
    description = describe_sign(phon_labels, target_sign)
    sentence_lines = []
    if sentence_prompt is not None:
        sentence_lines = [f'"{sentence_prompt.english}"',
                           " ".join(g.text for g in sentence_prompt.glosses)]
    footer_h = _reference_footer_height(left, description, sentence_lines)
    footer_top = max(HEADER_H + 160, left.shape[0] - footer_h)
    draw_reference_footer(left, footer_top, left.shape[0] - footer_top, description, sentence_lines)

    # Top-down: verdict content, then coach text right after it, then the
    # status bar right after THAT -- each section starts EXACTLY where the
    # previous one ended (no gap: a gap here is raw, un-paneled video peeking
    # through between two opaque sections, which is its own readability bug,
    # not just a cosmetic one) and a taller-than-usual section can never
    # overlap what comes after it (see draw_verdict's docstring).
    content_bottom = draw_verdict(right, result, target_sign, stale, capture_state, mastery=mastery)
    coach_bottom = draw_coach_text(right, coach_text, coach_text_for, y_top=content_bottom)
    draw_status_bar(right, coach_bottom, fps, ms)

    divider_canvas = np.hstack([left, right])
    cv2.line(divider_canvas, (left.shape[1], 0), (left.shape[1], height), DIVIDER, 2, cv2.LINE_AA)
    return divider_canvas


SENT_ROW_H = 44


def draw_sentence_header(canvas, seq, capture_state):
    """Sentence-mode analogue of draw_verdict's header + capture badge --
    same layout convention (opaque panels, top-down), just with the target
    text/gloss line instead of a single sign name."""
    _panel_bg(canvas, 0, HEADER_H, color=BG_HEADER, alpha=0.96)
    _text(canvas, f'TARGET SENTENCE   "{seq.english}"', 18, 40, F_HEADER, ACCENT, 2)
    _divider(canvas, HEADER_H)

    badge_top = HEADER_H
    _panel_bg(canvas, badge_top, badge_top + BADGE_H, color=BG_HEADER, alpha=0.93)
    label, color, hint = _CAPTURE_STATE_DISPLAY[capture_state]
    cv2.circle(canvas, (32, badge_top + 30), 9, color, -1, cv2.LINE_AA)
    _text(canvas, f"{label}   glosses: {' '.join(seq.gloss_ids)}", 54, badge_top + 38, F_BODY, color, 2)
    _text(canvas, hint, 54, badge_top + 62, F_TINY, TEXT_MUTED)
    _divider(canvas, badge_top + BADGE_H)
    return badge_top + BADGE_H


def draw_sentence_results(canvas, content_top, graded):
    """One compact row per gloss: how many of its judged parameters MATCHed,
    plus fidelity. Deliberately a scoreboard, not draw_verdict's full
    per-parameter detail view -- a multi-word sentence's 5-parameters-per-word
    breakdown wouldn't fit on screen at once (see project_workflow.md's Phase
    7 step 6 scoping); the coach-text band below still names the single most
    useful correction, same "one thing at a time" philosophy as single-sign
    mode."""
    if graded is None:
        _panel_bg(canvas, content_top, content_top + 50, color=BG_PANEL, alpha=0.94)
        _text(canvas, "sign the whole sentence, then press [n] to grade", 20, content_top + 32,
              F_SMALL, TEXT_MUTED)
        return content_top + 50

    content_h = 20 + SENT_ROW_H * len(graded)
    _panel_bg(canvas, content_top, content_top + content_h, color=BG_PANEL, alpha=0.94)
    y = content_top + 16
    for g in graded:
        matched = sum(1 for v in g.result.parameters.values() if v.correct is True)
        judged = sum(1 for v in g.result.parameters.values() if v.correct is not None)
        color = MATCH_COLOR if judged and matched == judged else OFF_COLOR
        row = f"{g.target_sign:<14} {matched}/{judged} matched    fidelity {g.result.fidelity:.3f}"
        _text(canvas, row, 20, y + 28, F_SMALL, color, 2)
        y += SENT_ROW_H
    return content_top + content_h


def compose_sentence_canvas(ref_frame, live_frame, seq, graded, capture_state, fps, ms,
                             coach_text=None, coach_text_for=None, height=920):
    def fit(frame, fallback_w=460):
        if frame is None:
            return np.full((height, fallback_w, 3), 18, np.uint8)
        h, w = frame.shape[:2]
        scale = height / h
        return cv2.resize(frame, (max(1, int(w * scale)), height))

    left = fit(ref_frame)
    right = fit(live_frame)

    _panel_bg(left, 0, HEADER_H, color=BG_HEADER, alpha=0.96)
    _text(left, "REFERENCE (stitched clips)", 18, 40, F_HEADER, ACCENT, 2)
    _divider(left, HEADER_H)

    badge_bottom = draw_sentence_header(right, seq, capture_state)
    content_bottom = draw_sentence_results(right, badge_bottom, graded)
    coach_bottom = draw_coach_text(right, coach_text, coach_text_for, y_top=content_bottom)
    draw_status_bar(right, coach_bottom, fps, ms)

    divider_canvas = np.hstack([left, right])
    cv2.line(divider_canvas, (left.shape[1], 0), (left.shape[1], height), DIVIDER, 2, cv2.LINE_AA)
    return divider_canvas


# ------------------------------------------------------------------------ live ----

def run_live(args):
    grader = EmbeddingGrader.build(args.checkpoint, which=args.which)
    args.extractor = args.extractor or grader.extractor
    skeleton = MEDIAPIPE_HOLISTIC if args.extractor == "mediapipe" else COCO_WHOLEBODY
    verify_pipeline_matches_checkpoint(args, grader, skeleton)

    rows = manifest_rows()
    cycle = resolve_targets(args, rows, grader)
    pool_signs = [sign for sign, _ in cycle]
    rows_by_sign = dict(cycle)
    print(f"targets ({len(cycle)}, [n] picks adaptively): {', '.join(pool_signs)}")
    print(DISCLAIMER)

    # Phase 6: per-sign/per-parameter mastery persisted across sessions, and the
    # minimal pairs available for contrastive drills WITHIN this pool -- both
    # scoped to `pool_signs` since a contrastive pick must itself be a resolvable
    # target (already fail-closed-validated above to have a reference clip).
    mastery = MasteryState.load(args.mastery_path)
    phon_labels = PhonologyLabels()
    minimal_pairs = find_minimal_pairs(phon_labels, pool_signs)

    ref = ReferenceLoop()
    state = {"target": None, "sentence_prompt": None, "coach_text": None, "coach_text_for": None}

    def switch_target(sign):
        row = rows_by_sign[sign]
        frames = load_reference_frames(REPO / row["video_path"])
        ref.set_target(sign, row["video_id"], frames)
        state["target"] = sign
        state["sentence_prompt"] = None
        print(f"target -> {sign}  (mastery {mastery.sign_mastery(sign):.0%}, "
              f"reference clip: {row['video_id']}, {len(frames)} frames)")
        print(f"  {describe_sign(phon_labels, sign)}")
        if args.sentence_prompts:
            # Blocking (same tradeoff already accepted for --llm-feedback):
            # this is an opt-in flag on a dev-machine demo script, not a
            # production UI -- see project_workflow.md's Phase 8 for what
            # "production" would actually require.
            seq = sentence_prompt_maybe_llm(sign)
            if seq is not None:
                state["sentence_prompt"] = seq
                print(f"  example: \"{seq.english}\" -> {seq.render()}")

    switch_target(pool_signs[0])

    # Motion-aware capture (aslcv.capture.CaptureBuffer) replaces a fixed sliding
    # window: it grows from the start of motion to a natural rest boundary instead
    # of silently truncating a slow signer's attempt -- see module docstring.
    energy_fn = functools.partial(hand_motion_energy, grader.pipeline.normalizer, skeleton)
    capture = CaptureBuffer(energy_fn, motion_threshold=grader.pipeline.motion_threshold,
                             settle_frames=args.settle_frames, preroll=args.preroll,
                             max_frames=args.capture_max)
    win_lock = threading.Lock()
    shared = {"result": None, "ms": 0.0, "stale": False}
    res_lock = threading.Lock()
    stop = threading.Event()

    def grade_loop():
        while not stop.is_set():
            with win_lock:
                snap = list(capture.frames)
                capture_state = capture.state
            if capture_state == "idle" or len(snap) < args.min_frames:
                time.sleep(0.03)
                continue
            with res_lock:
                target = state["target"]
            t0 = time.time()
            try:
                result = grader.grade_against_poses(snap, target)
            except Exception as exc:  # keep the demo alive on a bad window
                print("grade error:", exc)
                result = None
            with res_lock:
                if result is not None:
                    shared["result"] = result
                    shared["stale"] = False
                shared["ms"] = (time.time() - t0) * 1000.0

    print(f"opening camera {args.camera} in LIVE mode (extractor {args.extractor}"
          f"{', gpu' if args.gpu else ''}) ...")
    extractor, _ = build_extractor(args.extractor, RunningMode.LIVE, gpu=args.gpu)
    k = len(skeleton.names)
    zero_pose = lambda: Pose(np.zeros((k, 2), np.float32), np.zeros(k, np.float32))

    worker = threading.Thread(target=grade_loop, name="grader", daemon=True)
    worker.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        stop.set()
        extractor.close()
        raise SystemExit(f"cannot open camera {args.camera}")

    print("controls: [q]/ESC quit   [c] clear capture (keeps last verdict)   "
          "[n] record + adaptively pick next target")
    fps_t, fps_n, fps = time.time(), 0, 0.0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            pose = extractor.extract(frame)  # RAW frame, mirrored=False -> matches refs (issue #6)
            with win_lock:
                capture.append(pose if pose is not None else zero_pose())
                capture_state = capture.state

            live_canvas = extractor.draw(frame, pose) if pose is not None else frame.copy()
            if args.mirror:
                live_canvas = cv2.flip(live_canvas, 1)  # display-only selfie flip

            ref_frame = ref.next_frame()

            with res_lock:
                result, ms, stale = shared["result"], shared["ms"], shared["stale"]

            canvas = compose_canvas(ref_frame, live_canvas, state["target"], phon_labels, result, stale,
                                     capture_state, fps, ms, mastery=mastery,
                                     sentence_prompt=state["sentence_prompt"],
                                     coach_text=state["coach_text"], coach_text_for=state["coach_text_for"])
            cv2.imshow("ASL diagnose demo", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                with win_lock:
                    capture.reset()
                with res_lock:
                    shared["stale"] = True  # last verdict stays visible, just marked stale
            if key == ord("n"):
                # Phase 6: record the CURRENT (non-stale) verdict into mastery
                # before moving on -- a stale one is a re-sign in progress, not
                # a completed attempt, so it must not be scored twice or scored
                # as this attempt's result.
                wrong_parameter = None
                if result is not None and not stale:
                    correct_by_parameter = {p: v.correct for p, v in result.parameters.items()}
                    mastery.update(state["target"], correct_by_parameter)
                    mastery.save(args.mastery_path)
                    text = coach_text_maybe_llm(state["target"], result.parameters, use_llm=args.llm_feedback)
                    print(f"  [{state['target']}] {text}")
                    state["coach_text"] = text
                    state["coach_text_for"] = state["target"]
                    wrong_parameter = focus_parameter(result.parameters)
                next_sign = pick_next(mastery, pool_signs, last_sign=state["target"],
                                       last_wrong_parameter=wrong_parameter, minimal_pairs=minimal_pairs)
                if wrong_parameter is not None and next_sign != state["target"] \
                        and any(next_sign in (a, b) for a, b in minimal_pairs.get(wrong_parameter, [])):
                    print(f"  -> contrastive drill: {next_sign} (differs from "
                          f"{state['target']} only in {wrong_parameter})")
                switch_target(next_sign)
                with win_lock:
                    capture.reset()
                with res_lock:
                    shared["result"] = None
                    shared["stale"] = False

            fps_n += 1
            if time.time() - fps_t >= 0.5:
                fps = fps_n / (time.time() - fps_t)
                fps_t, fps_n = time.time(), 0
    finally:
        stop.set()
        worker.join(timeout=1.0)
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


def run_live_sentence(args):
    """Phase 7 step 6: sentence mode (`--sentence "..."`). One fixed target
    sentence, ONE continuous capture bounded by CaptureBuffer tuned for a
    multi-sign attempt (`--sentence-settle-frames`/`--sentence-capture-max`,
    larger than single-sign mode's defaults -- see step 5's writeup and
    tests/test_capture.py's sentence-mode tests for why), then forced-aligned
    and graded segment-by-segment via `align_and_grade` on `[n]`.

    Deliberately NO background grade_loop thread the way single-sign mode
    has: alignment needs the COMPLETE attempt against the COMPLETE
    reference, not a growing partial window graded continuously -- there is
    no meaningful "live partial verdict" for a forced alignment the way there
    is for a single isolated sign.

    Deliberately NOT adaptive (no next-sentence picker): this grades ONE
    sentence per run. Sequencing which sentence to practice next is a bigger
    feature than Phase 7's 6-step scope covers (see project_workflow.md) --
    single-sign mode's scheduler is unaffected and untouched by this."""
    grader = EmbeddingGrader.build(args.checkpoint, which=args.which)
    args.extractor = args.extractor or grader.extractor
    skeleton = MEDIAPIPE_HOLISTIC if args.extractor == "mediapipe" else COCO_WHOLEBODY
    verify_pipeline_matches_checkpoint(args, grader, skeleton)

    seq, composed_video = resolve_sentence(args, grader)
    print(f'target sentence: "{seq.english}" -> {" ".join(seq.gloss_ids)}')
    print(DISCLAIMER)

    mastery = MasteryState.load(args.mastery_path)

    ref = ReferenceLoop()
    ref.set_target(seq.english, "composed", composed_video.frames)

    energy_fn = functools.partial(hand_motion_energy, grader.pipeline.normalizer, skeleton)
    capture = CaptureBuffer(energy_fn, motion_threshold=grader.pipeline.motion_threshold,
                             settle_frames=args.sentence_settle_frames, preroll=args.preroll,
                             max_frames=args.sentence_capture_max)
    win_lock = threading.Lock()
    state = {"graded": None, "coach_text": None, "coach_text_for": None}

    print(f"opening camera {args.camera} in LIVE mode (extractor {args.extractor}"
          f"{', gpu' if args.gpu else ''}) ...")
    extractor, _ = build_extractor(args.extractor, RunningMode.LIVE, gpu=args.gpu)
    k = len(skeleton.names)
    zero_pose = lambda: Pose(np.zeros((k, 2), np.float32), np.zeros(k, np.float32))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        extractor.close()
        raise SystemExit(f"cannot open camera {args.camera}")

    print("controls: [q]/ESC quit   [c] clear capture and retry   "
          "[n] grade the completed attempt (only once CAPTURED)")
    fps_t, fps_n, fps = time.time(), 0, 0.0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            pose = extractor.extract(frame)  # RAW frame, mirrored=False -> matches refs (issue #6)
            with win_lock:
                capture.append(pose if pose is not None else zero_pose())
                capture_state = capture.state
                snap = list(capture.frames)

            live_canvas = extractor.draw(frame, pose) if pose is not None else frame.copy()
            if args.mirror:
                live_canvas = cv2.flip(live_canvas, 1)  # display-only selfie flip

            ref_frame = ref.next_frame()
            canvas = compose_sentence_canvas(ref_frame, live_canvas, seq, state["graded"], capture_state,
                                              fps, 0.0, coach_text=state["coach_text"],
                                              coach_text_for=state["coach_text_for"])
            cv2.imshow("ASL diagnose demo -- sentence mode", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                with win_lock:
                    capture.reset()
                state["graded"] = None
            if key == ord("n") and capture_state == "settled":
                try:
                    _, graded = align_and_grade(grader, snap, seq)
                except Exception as exc:  # noqa: BLE001 -- keep the demo alive on a bad attempt
                    print("alignment/grading error:", exc)
                    graded = None
                if graded is not None:
                    state["graded"] = graded
                    for g in graded:
                        correct_by_parameter = {p: v.correct for p, v in g.result.parameters.items()}
                        mastery.update(g.target_sign, correct_by_parameter)
                    mastery.save(args.mastery_path)

                    # coach the FIRST segment with a real mistake -- one correction
                    # at a time, same philosophy as single-sign mode's coach_text
                    worst = next((g for g in graded if focus_parameter(g.result.parameters) is not None), None)
                    if worst is not None:
                        text = coach_text_maybe_llm(worst.target_sign, worst.result.parameters,
                                                     use_llm=args.llm_feedback)
                        state["coach_text"] = text
                        state["coach_text_for"] = worst.target_sign
                        print(f"  [{worst.target_sign}] {text}")
                    else:
                        state["coach_text"] = "Great job -- every judged parameter matched across the sentence."
                        state["coach_text_for"] = seq.english
                    for g in graded:
                        print(f"  {g.target_sign:<14} frames {g.frame_range}  fidelity {g.result.fidelity:.3f}")

            fps_n += 1
            if time.time() - fps_t >= 0.5:
                fps = fps_n / (time.time() - fps_t)
                fps_t, fps_n = time.time(), 0
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------- selftest ----

def run_selftest(args):
    """No camera: push cached val clips' frames through the SAME grade_against_poses
    -> verdict path the live loop uses, so the wiring (and the headline mother/father
    head-independence behavior) is verifiable offline. Uses a plain trailing window
    (args.window) over the already-complete cached clip -- CaptureBuffer's motion
    state machine is a live-capture concern, exercised only by the real camera path."""
    grader = EmbeddingGrader.build(args.checkpoint, which=args.which)
    args.extractor = args.extractor or grader.extractor
    skeleton = MEDIAPIPE_HOLISTIC if args.extractor == "mediapipe" else COCO_WHOLEBODY
    verify_pipeline_matches_checkpoint(args, grader, skeleton)

    rows = manifest_rows()
    val_by_sign = defaultdict(list)
    for r in rows:
        if r["split"] == "val":
            val_by_sign[r["id_gloss"]].append(r)
    cache_dir = REPO / "data" / "cache" / grader.extractor

    def emulate(row):
        npz = cache_dir / f"{row['video_id']}.npz"
        return list(deque(_poses_from_npz(npz), maxlen=args.window))

    def print_result(true_sign, target_sign, result):
        parts = []
        for p in ALL_PARAMETERS:
            v = result.parameters[p]
            tag, _ = _tag_and_color(v)
            parts.append(f"{p}={tag}({v.confidence:.0%})")
        print(f"  attempt={true_sign:<10} target={target_sign:<10} "
              f"fidelity={result.fidelity:.3f}  {' '.join(parts)}")

    print(f"\nselftest: prompt -> grade_against_poses -> verdict path, no camera "
          f"(trailing window = last {args.window} frames)\n")

    print("self-check (attempt graded against its OWN true sign):")
    for sign in ("mother", "father", "you", "me", "water", "thank_you"):
        clips = val_by_sign.get(sign)
        if not clips:
            print(f"  {sign:<10} (no val clip, skipped)")
            continue
        result = grader.grade_against_poses(emulate(clips[0]), sign)
        print_result(sign, sign, result)

    print("\nmother/father minimal-pair cross-check (differ ONLY in minor_location):")
    for true_sign, target_sign in (("father", "mother"), ("mother", "father")):
        clips = val_by_sign.get(true_sign)
        if not clips:
            print(f"  no val clip for {true_sign}, skipped")
            continue
        result = grader.grade_against_poses(emulate(clips[0]), target_sign)
        print_result(true_sign, target_sign, result)

    if args.sentence:
        print(f'\nsentence selftest: "{args.sentence}"')
        seq, _composed_video = resolve_sentence(args, grader)
        attempt_poses = []
        missing = [g for g in seq.gloss_ids if not val_by_sign.get(g)]
        if missing:
            raise SystemExit(f"sentence selftest needs a val clip for each of {missing}, none cached")
        for id_gloss in seq.gloss_ids:
            npz = cache_dir / f"{val_by_sign[id_gloss][0]['video_id']}.npz"
            attempt_poses.extend(_poses_from_npz(npz))
        _, graded = align_and_grade(grader, attempt_poses, seq)
        for g in graded:
            print(f"  {g.target_sign:<14} frames {g.frame_range}  fidelity {g.result.fidelity:.3f}  "
                  f"focus={focus_parameter(g.result.parameters)}")
        print("sentence selftest complete -- same align_and_grade/compose_sentence_canvas "
              "path run_live_sentence uses.")

    print("\nselftest complete -- this is the offline path; a live failure is camera/lighting, not grading.")
    print(DISCLAIMER)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--target", default=None, help="starting target sign; [n] still cycles onward from here")
    ap.add_argument("--targets", default=None, help="comma-separated cycle list (default: a curated set incl. mother/father)")
    ap.add_argument("--camera", type=int, default=0, help="cv2 VideoCapture index")
    ap.add_argument("--window", type=int, default=60,
                    help="--selftest only: trailing-window length (frames) over a cached clip; "
                         "the live path uses --settle-frames/--preroll/--capture-max instead")
    ap.add_argument("--min-frames", type=int, default=20, help="frames needed before grading starts")
    ap.add_argument("--settle-frames", type=int, default=8,
                    help="live capture: consecutive low-motion frames that mark an attempt CAPTURED")
    ap.add_argument("--preroll", type=int, default=12,
                    help="live capture: rest frames kept before motion starts (natural clip lead-in)")
    ap.add_argument("--capture-max", type=int, default=150,
                    help="live capture: safety cap (frames) so a stuck/very slow attempt can't grow forever")
    ap.add_argument("--sentence", default=None,
                    help="Phase 7: switch to sentence mode -- grade a full sentence (e.g. "
                         "'I want water.') via forced alignment instead of one isolated sign. "
                         "Refused (SystemExit) if the gloss engine rejects it as out of scope or "
                         "any word lacks a cached reference clip.")
    ap.add_argument("--sentence-settle-frames", type=int, default=30,
                    help="--sentence only: larger than --settle-frames' single-sign default since "
                         "a multi-sign attempt has several motion rises with brief pauses between "
                         "words that must NOT be misread as the end of the attempt (see "
                         "project_workflow.md's Phase 7 step 5 -- tuned conservatively, not "
                         "validated against a real camera in this environment)")
    ap.add_argument("--sentence-capture-max", type=int, default=400,
                    help="--sentence only: safety cap sized for several signs' worth of frames, "
                         "not one (same untuned-conservative caveat as --sentence-settle-frames)")
    ap.add_argument("--mirror", dest="mirror", action="store_true", default=True,
                    help="selfie-mirror the DISPLAY only (default on; grading uses the raw frame)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--extractor", default=None,
                    help="override for verification only -- must match the checkpoint's own "
                         "extractor or the demo refuses to start (default: the checkpoint's own)")
    ap.add_argument("--gpu", action="store_true",
                    help="mediapipe only: request the GPU delegate (~3.3x faster, measured; "
                         "falls back to CPU automatically if unavailable on this machine)")
    ap.add_argument("--mastery-path", type=Path, default=DEFAULT_MASTERY_PATH,
                    help="Phase 6 learner-state JSON (persists across sessions; missing file = fresh learner)")
    ap.add_argument("--llm-feedback", action="store_true",
                    help="phrase [n]'s coaching line via HuggingFace's hosted Inference API "
                         "(needs HF_TOKEN set) instead of the templated text; falls back to "
                         "templated text automatically on any missing token/network/API failure")
    ap.add_argument("--sentence-prompts", action="store_true",
                    help="show an LLM-written example sentence using each target's word, "
                         "gloss-composed and validated by the fail-closed rule engine before "
                         "display (needs HF_TOKEN set); presentational only, not graded; "
                         "silently skipped on any missing token/network/API failure or if the "
                         "rule engine refuses every attempt")
    ap.add_argument("--selftest", action="store_true", help="no camera: verify the whole path on cached val clips")
    add_pipeline_args(ap)
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
    elif args.sentence:
        run_live_sentence(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
