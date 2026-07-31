"""Phonological label vocabulary for the Phase 4 diagnosis heads.

Five parameters are diagnosed: ``handshape``, ``major_location``, ``minor_location``,
``movement`` (categorical), and ``repeated_movement`` (binary). Class vocabularies are
derived from the ACTUAL 60-sign ``data/phonology.csv`` -- not curriculum.yaml's
``parameter_space``, which declares a few label values (e.g. movement's
"BackAndForth"/"Z-shaped"/"X-shaped") that occur in zero curriculum signs today; a head
sized to include them would carry output classes that can never be trained or
evaluated. Same convention as ``dataset.py``'s ``LabelEncoder``: sorted, deterministic,
fit once over every sign so train/val/test agree on the same class indices.

**The minimum-support gate.** Several label values are carried by only 1-2 of the 60
signs (9 of 20 handshape classes, major_location's ``Arm``, 4 of 14 minor_location
classes) -- for those, "the head got it right" is indistinguishable from "the head
recognized that one specific sign," so it cannot be shown to generalize. Rather than
drop those signs or merge classes (both lose real diagnostic specificity), every
parameter's classes are trained and predicted as-is, but ``support()`` reports how many
DISTINCT SIGNS carry each label value, and callers (``EmbeddingGrader.grade_against``)
use it to gate what's SHOWN to the user: a verdict for a label value backed by fewer
than ``MIN_SUPPORT`` signs is reported as "insufficient data," not a confident
right/wrong -- the same fail-closed instinct already applied in
``production/gloss_rules.py``, here at the level of one phonological verdict rather
than a whole gloss.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PHONOLOGY_CSV = REPO / "data" / "phonology.csv"

# The categorical parameters get a class-index head; repeated_movement is binary and
# handled separately (no class list needed -- see PhonologyLabels.repeated below).
CATEGORICAL_PARAMETERS = ("handshape", "major_location", "minor_location", "movement")
ALL_PARAMETERS = CATEGORICAL_PARAMETERS + ("repeated_movement",)

# A verdict for a label value backed by fewer than this many DISTINCT signs is
# reported as "insufficient data" rather than a confident correct/incorrect --
# the decision made explicitly for this phase (see project_workflow.md, Phase 4).
MIN_SUPPORT = 3


def _read_phonology_rows() -> list[dict]:
    if not PHONOLOGY_CSV.exists():
        raise FileNotFoundError(
            f"{PHONOLOGY_CSV} not found -- run `tools/join_phonology.py` first")
    with open(PHONOLOGY_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class PhonologyLabelEncoder:
    """Deterministic label <-> int mapping for ONE categorical parameter.

    Mirrors ``dataset.py.LabelEncoder`` (sorted classes, encode/decode) but for a
    phonological parameter's label values rather than sign identity.
    """

    def __init__(self, values):
        self.classes_ = sorted(set(values))
        self._to_idx = {c: i for i, c in enumerate(self.classes_)}

    def encode(self, value: str) -> int:
        return self._to_idx[value]

    def decode(self, idx: int) -> str:
        return self.classes_[idx]

    def __len__(self) -> int:
        return len(self.classes_)


class PhonologyLabels:
    """All five parameters' label encoders + per-class support, fit once over the
    full 60-sign phonology table (a property of the curriculum, not of one split)."""

    def __init__(self, rows: "list[dict] | None" = None):
        if rows is None:
            rows = _read_phonology_rows()
        self.rows = rows
        self.by_gloss: dict[str, dict] = {r["id_gloss"]: r for r in rows}

        self.encoders: dict[str, PhonologyLabelEncoder] = {
            p: PhonologyLabelEncoder(r[p] for r in rows) for p in CATEGORICAL_PARAMETERS
        }
        # support[param][label_value] = number of DISTINCT SIGNS carrying that value.
        self._support: dict[str, Counter] = {
            p: Counter(r[p] for r in rows) for p in CATEGORICAL_PARAMETERS
        }
        # repeated_movement: binary, no encoder needed, but still gated the same way --
        # support here means "signs with this 0/1 value," same semantics as the others.
        self._support["repeated_movement"] = Counter(r["repeated_movement"] for r in rows)

    def label_for(self, id_gloss: str, parameter: str) -> str:
        """The raw phonology.csv string value for one sign/parameter."""
        return self.by_gloss[id_gloss][parameter]

    def support(self, parameter: str, value: str) -> int:
        """How many distinct curriculum signs carry this exact label value."""
        return self._support[parameter][value]

    def well_supported(self, parameter: str, value: str) -> bool:
        return self.support(parameter, value) >= MIN_SUPPORT

    def repeated_bool(self, id_gloss: str) -> bool:
        return self.by_gloss[id_gloss]["repeated_movement"] == "1"
