"""COCO-WholeBody 133-keypoint layout as a `Skeleton`, plus region slicing.

The DWPose / RTMPose whole-body models (ucoco) emit keypoints in the exact
index order below, so `NAMES[i]` labels row `i` of a `Pose.keypoints` array.

Index blocks:
    0-16    body   (COCO 17)
    17-22   feet   (3 per foot)
    23-90   face   (68 landmarks, iBUG/300W order)
    91-111  left hand   (wrist root + 5 fingers x 4 joints)
    112-132 right hand

Region tuples (BODY, LEFT_HAND, FACE_REGIONS, ...) are derived from the names
so the +23 face offset can't be mistyped. Slice with them by meaning:
    kp[LEFT_HAND]                     -> just the left-hand points
    kp[FACE_REGIONS["inner_mouth"]]   -> mouth-morpheme points for ASL

These module-level tuples also feed `COCO_WHOLEBODY.regions`, the canonical
cross-topology vocabulary (normalizer/base.py:REQUIRED_REGIONS) -- so downstream
code that wants "the legs" or "the left hand" can ask `skeleton.region(name)`
instead of re-deriving name-prefix matches per topology.

Note: left/right in the face regions follow the iBUG convention (image space),
which mirrors the subject. Verify once against a real frame (see module docs).
"""

from .base import Skeleton

# --- names, in model output order -------------------------------------------

_BODY = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_FEET = (
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)
_FACE = tuple(f"face_{i}" for i in range(68))
_FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")


def _hand_names(side: str) -> tuple[str, ...]:
    names = [f"{side}_hand_root"]
    for finger in _FINGERS:
        for joint in range(1, 5):
            names.append(f"{side}_{finger}{joint}")
    return tuple(names)


_NAMES = _BODY + _FEET + _FACE + _hand_names("left") + _hand_names("right")
_idx = {name: i for i, name in enumerate(_NAMES)}

# --- edges, declared by name then resolved to indices -----------------------
# Face is intentionally unconnected (drawn as points, not a mesh).

_BODY_LINKS = (
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
)
_FEET_LINKS = (
    ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"),
    ("right_ankle", "right_heel"),
)
_WRIST_LINKS = (
    ("left_wrist", "left_hand_root"), ("right_wrist", "right_hand_root"),
)


def _hand_links(side: str) -> list[tuple[str, str]]:
    links = []
    for finger in _FINGERS:
        links.append((f"{side}_hand_root", f"{side}_{finger}1"))
        for joint in range(1, 4):
            links.append((f"{side}_{finger}{joint}", f"{side}_{finger}{joint + 1}"))
    return links


_NAME_LINKS = (
    list(_BODY_LINKS) + list(_FEET_LINKS) + list(_WRIST_LINKS)
    + _hand_links("left") + _hand_links("right")
)
_EDGES = tuple((_idx[a], _idx[b]) for a, b in _NAME_LINKS)

# --- region slicing (derived from names; offset-safe) -----------------------


def _where(pred) -> tuple[int, ...]:
    return tuple(i for i, n in enumerate(_NAMES) if pred(n))


def _is_hand(name: str, side: str) -> bool:
    return name.startswith(side + "_") and (
        "hand_root" in name or "finger" in name or "thumb" in name
    )


BODY = tuple(range(0, 17))
FEET = _where(lambda n: n.endswith("_toe") or n.endswith("_heel"))
FACE = _where(lambda n: n.startswith("face_"))
LEFT_HAND = _where(lambda n: _is_hand(n, "left"))
RIGHT_HAND = _where(lambda n: _is_hand(n, "right"))

# --- canonical cross-topology REGIONS (normalizer.base.REQUIRED_REGIONS) ----
# Splits the COCO-17 `BODY` block by meaning: nose/eyes/ears + shoulders
# (body_upper, coarse head+torso reference) vs the elbow->wrist chain (arms);
# hips/knees/ankles join `legs_feet` alongside the foot block. `face` reuses the
# existing 68-point `FACE` constant AS-IS -- unlike MediaPipe, COCO's coarse
# body-block head points were never folded into "face" by the code this region
# replaces (features.py's old `name.startswith("face_")` face toggle), so
# keeping them out of the region here preserves that toggle's exact behavior.
_BODY_UPPER_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                     "left_shoulder", "right_shoulder")
_ARMS_NAMES = ("left_elbow", "right_elbow", "left_wrist", "right_wrist")
_LEGS_NAMES = (
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)

_REGIONS = (
    ("body_upper", tuple(_idx[n] for n in _BODY_UPPER_NAMES)),
    ("arms", tuple(_idx[n] for n in _ARMS_NAMES)),
    ("left_hand", LEFT_HAND),
    ("right_hand", RIGHT_HAND),
    ("face", FACE),
    ("legs_feet", tuple(_idx[n] for n in _LEGS_NAMES) + FEET),
)


def _face(a: int, b: int) -> tuple[int, ...]:
    # resolve face-local numbers to absolute indices via the name map,
    # so the +23 offset is handled for free
    return tuple(_idx[f"face_{i}"] for i in range(a, b))


FACE_REGIONS = {
    "jaw": _face(0, 17),
    "right_eyebrow": _face(17, 22),
    "left_eyebrow": _face(22, 27),
    "nose_bridge": _face(27, 31),
    "lower_nose": _face(31, 36),
    "right_eye": _face(36, 42),
    "left_eye": _face(42, 48),
    "outer_mouth": _face(48, 60),
    "inner_mouth": _face(60, 68),
}

# --- named anchors (resolved from names, so indices can't be mistyped) ------
# Semantic keypoints downstream normalization needs by meaning. Looked up by
# name so the block offsets above are handled for free.

# Canonical cross-topology anchor NAME -> this skeleton's own point name. Body
# anchors reuse their point name; the hand anchors use NEUTRAL names (shared with the
# MediaPipe skeleton; see normalizer.base.REQUIRED_ANCHORS) that map to COCO's joints:
#   *_hand_wrist       -> *_hand_root       the HAND-block root, NOT the arm-chain
#                                           "*_wrist" (idx 9/10); the local hand frame
#                                           normalizes finger points, so its origin
#                                           must come from the same hand-block estimate.
#   *_hand_middle_mcp  -> *_middle_finger1  base knuckle (MCP); COCO numbers finger
#                                           joints 1..4 from the base, so *_finger1 is
#                                           the MCP -- verified by tests/test_anchors.py.
#   *_body_wrist       -> *_wrist           the ARM-CHAIN wrist (idx 9/10), NOT the
#                                           hand-block root -- deliberately named
#                                           differently from *_hand_wrist so the two
#                                           are never confused. Used with *_elbow for
#                                           arm-foreshortening scale (a monocular depth
#                                           proxy), not for any hand-shape frame.
_ANCHOR_TO_POINT = {
    "nose": "nose",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_hand_wrist": "left_hand_root",
    "right_hand_wrist": "right_hand_root",
    "left_hand_middle_mcp": "left_middle_finger1",
    "right_hand_middle_mcp": "right_middle_finger1",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_body_wrist": "left_wrist",
    "right_body_wrist": "right_wrist",
}
_ANCHORS = tuple((anchor, _idx[point]) for anchor, point in _ANCHOR_TO_POINT.items())

# --- the public artifact ----------------------------------------------------

COCO_WHOLEBODY = Skeleton(names=_NAMES, edges=_EDGES, anchors=_ANCHORS, regions=_REGIONS)
