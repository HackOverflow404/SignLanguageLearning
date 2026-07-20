#!/usr/bin/env python3
"""Join ASL-LEX 2.0 phonology + frequency onto the curriculum signs -> data/phonology.csv.

Reads every curriculum sign's `asllex_code` and looks it up in ASL-LEX's master
table (data/ASL_LEX/Data Files/signdata.csv) by its `Code` column, then writes
data/phonology.csv keyed by asllex_code with every phonological parameter ASL-LEX
records for the sign -- including the ones curriculum.yaml lacks (selected fingers,
flexion, non-dominant handshape, orientation/wrist-twist, thumb position, contact,
second minor location) plus sign frequency and percent-unknown.

Only FIRST-morpheme (`.2.0`) parameters are pulled: ASL-LEX codes phonology for the
first morpheme only (the M2..M6 columns describe later morphemes). Signs with more
than one morpheme are flagged `is_multimorphemic` -- their parameters describe only
the first morpheme, so parameter-level diagnosis of the rest would be invalid.

Inspects the real CSV header and reports which requested columns were found vs.
missing; it never invents a column.

    python tools/join_phonology.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CURRICULUM = REPO / "curriculum.yaml"
SIGNDATA = REPO / "data" / "ASL_LEX" / "Data Files" / "signdata.csv"
OUTPUT = REPO / "data" / "phonology.csv"

KEY_COLUMN = "Code"  # ASL-LEX 'Code' holds the D_02_065-style asllex_code

# Requested parameter -> ASL-LEX column. Checked against the real header; only the
# ones that exist are emitted. All first-morpheme (.2.0) values.
PARAM_COLUMNS = {
    # parameters curriculum.yaml already carries (kept so the answer key is complete)
    "handshape": "Handshape.2.0",
    "marked_handshape": "MarkedHandshape.2.0",
    "sign_type": "SignType.2.0",
    "movement": "Movement.2.0",
    "repeated_movement": "RepeatedMovement.2.0",
    "major_location": "MajorLocation.2.0",
    "minor_location": "MinorLocation.2.0",
    "lexical_class": "LexicalClass",
    # parameters curriculum.yaml LACKS (the reason for this join)
    "selected_fingers": "SelectedFingers.2.0",
    "flexion": "Flexion.2.0",
    "flexion_change": "FlexionChange.2.0",
    "spread": "Spread.2.0",
    "spread_change": "SpreadChange.2.0",
    "nondominant_handshape": "NonDominantHandshape.2.0",
    "ulnar_rotation": "UlnarRotation.2.0",  # orientation / wrist-twist
    "thumb_position": "ThumbPosition.2.0",
    "thumb_contact": "ThumbContact.2.0",
    "contact": "Contact.2.0",
    "second_minor_location": "SecondMinorLocation.2.0",
    # frequency / familiarity
    "sign_frequency": "SignFrequency(M)",
    "sign_frequency_z": "SignFrequency(Z)",
    "percent_unknown": "Unknown",
    # morphology (used for the multimorphemic flag; kept as columns too)
    "num_morphemes": "NumberOfMorphemes.2.0",
    "is_compound": "Compound.2.0",
    "initialized": "Initialized.2.0",
    "fingerspelled_loan": "FingerspelledLoanSign.2.0",
}

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # signdata.csv is not clean UTF-8


def read_rows(path):
    for enc in ENCODINGS:
        try:
            with open(path, newline="", encoding=enc) as f:
                rd = csv.DictReader(f)
                return list(rd), (rd.fieldnames or [])
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"could not decode {path} with any of {ENCODINGS}")


def load_curriculum():
    doc = yaml.safe_load(open(CURRICULUM, encoding="utf-8"))
    return [(s["asllex_code"], s["gloss"]) for u in doc["units"] for s in u["signs"]]


def truthy(v):
    return str(v).strip().lower() in ("1", "yes", "true")


def is_multimorphemic(row):
    n = str(row.get(PARAM_COLUMNS["num_morphemes"], "")).strip()
    try:
        multi = int(float(n)) > 1
    except ValueError:
        multi = False
    return multi or truthy(row.get(PARAM_COLUMNS["is_compound"], ""))


def main():
    curriculum = load_curriculum()
    rows, header = read_rows(SIGNDATA)
    print(f"ASL-LEX signdata: {len(rows)} signs, {len(header)} columns")

    hdr = set(header)
    resolved = {name: col for name, col in PARAM_COLUMNS.items() if col in hdr}
    missing = {name: col for name, col in PARAM_COLUMNS.items() if col not in hdr}

    print(f"\nrequested parameter columns FOUND ({len(resolved)}):")
    for name, col in resolved.items():
        print(f"  {name:<24} <- {col}")
    if missing:
        print(f"\nrequested columns MISSING from ASL-LEX header ({len(missing)}) -- skipped, not invented:")
        for name, col in missing.items():
            print(f"  {name:<24} (wanted {col!r})")
    else:
        print("\nall requested columns present in ASL-LEX.")

    by_code = {r[KEY_COLUMN]: r for r in rows}
    out_cols = ["asllex_code", "id_gloss"] + list(resolved) + ["is_multimorphemic"]

    unresolved, multimorph, written = [], [], 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for code, gloss in curriculum:
            src = by_code.get(code)
            if src is None:
                unresolved.append((code, gloss))
                continue
            rec = {"asllex_code": code, "id_gloss": gloss}
            for name, col in resolved.items():
                rec[name] = (src.get(col, "") or "").strip()
            mm = is_multimorphemic(src)
            rec["is_multimorphemic"] = int(mm)
            if mm:
                multimorph.append((code, gloss, rec.get("num_morphemes"), rec.get("is_compound")))
            w.writerow(rec)
            written += 1

    print(f"\nwrote {OUTPUT.relative_to(REPO)}: {written}/{len(curriculum)} signs, "
          f"{len(out_cols)} columns")
    if unresolved:
        print(f"!! {len(unresolved)} curriculum signs NOT found in ASL-LEX by {KEY_COLUMN}:")
        for code, gloss in unresolved:
            print(f"   {code}  {gloss}")

    print(f"\nMULTI-MORPHEMIC / COMPOUND signs ({len(multimorph)}) -- ASL-LEX phonology "
          f"describes only their FIRST morpheme, so parameter-level diagnosis of the\n"
          f"remaining morphemes would be invalid; exclude or annotate before Phase 4:")
    for code, gloss, n, comp in multimorph:
        print(f"   {code}  {gloss:<12} morphemes={n} compound={comp}")

    ok = written == len(curriculum) and not unresolved
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
