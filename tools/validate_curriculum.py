"""Validate curriculum.yaml, and regenerate its derived sections.

    python tools/validate_curriculum.py            # check only
    python tools/validate_curriculum.py --pairs    # print regenerated pairs

Checks the invariants that hand-editing breaks:
  * count matches the actual number of signs
  * coverage stats match the vectors (the v0.1 movement counts were wrong)
  * every contrastive pair is TRULY minimal (v0.1 had 6 of 16 that weren't)
  * every pair / construction example resolves to a real ID-gloss
  * glosses are unique, and YAML didn't coerce "yes"/"no"/handshapes to bools
  * every sign carries english_lemmas (the 5b lexicon depends on it)
  * every sign carries an asllex_code, codes are unique, and each one resolves
    in the ASL Citizen splits (the join key from tools/resolve_keys.py)

Run after ANY vocabulary change, and paste the regenerated blocks back in.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

FIELDS = ["handshape", "major_location", "minor_location", "movement",
          "repeated", "sign_type"]

REPO = Path(__file__).resolve().parents[1]
CURRICULUM = REPO / "curriculum.yaml"
CITIZEN_SPLITS = REPO / "data" / "ASL_Citizen" / "splits"


def load(path: Path):
    d = yaml.safe_load(open(path))
    signs = [s for u in d["units"] for s in u["signs"]]
    return d, signs


def load_citizen_codes() -> set[str] | None:
    """Set of ASL-LEX Codes present in the Citizen splits, or None if absent.

    Column name is matched case-insensitively (never hard-coded); encoding is
    auto-detected. Returns None when the splits are not downloaded so the
    validator can degrade to a warning rather than a hard failure.
    """
    if not CITIZEN_SPLITS.is_dir():
        return None
    files = sorted(CITIZEN_SPLITS.glob("*.csv"))
    if not files:
        return None
    codes: set[str] = set()
    for path in files:
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                with open(path, newline="", encoding=enc) as f:
                    rd = csv.DictReader(f)
                    ccol = next((c for c in (rd.fieldnames or [])
                                 if "code" in c.lower()), None)
                    if not ccol:
                        break
                    for r in rd:
                        code = (r[ccol] or "").strip()
                        if code:
                            codes.add(code)
                break
            except UnicodeDecodeError:
                continue
    return codes


def minimal_on(a, b, f) -> bool:
    """True iff a and b differ on f and agree on every other comparison field.

    Exception: minor_location is nested inside major_location, so a
    major_location pair is exempt from matching minor_location (changing the
    major region necessarily changes the minor one).
    """
    if a[f] == b[f]:
        return False
    for g in FIELDS:
        if g == f or (f == "major_location" and g == "minor_location"):
            continue
        if a[g] != b[g]:
            return False
    return True


def derive_pairs(signs):
    by_gloss = {s["gloss"]: s for s in signs}
    pairs = {f: [] for f in FIELDS}
    for x, y in itertools.combinations(sorted(by_gloss), 2):
        for f in FIELDS:
            if minimal_on(by_gloss[x], by_gloss[y], f):
                pairs[f].append((x, y, by_gloss[x][f], by_gloss[y][f]))
    return {f: p for f, p in pairs.items() if p}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(CURRICULUM))
    ap.add_argument("--pairs", action="store_true",
                    help="print the regenerated contrastive_pairs block")
    args = ap.parse_args()

    d, signs = load(Path(args.path))
    errors, warnings = [], []

    # --- counts -----------------------------------------------------------
    if d["count"] != len(signs):
        errors.append(f"count says {d['count']}, found {len(signs)} signs")

    glosses = [s["gloss"] for s in signs]
    dupes = [g for g, n in Counter(glosses).items() if n > 1]
    if dupes:
        errors.append(f"duplicate glosses: {dupes}")

    # --- YAML type safety -------------------------------------------------
    # "yes"/"no" and single-letter handshapes are bool-coercible in YAML 1.1.
    for s in signs:
        for f in ("gloss", "asllex_id", "handshape", "movement"):
            if not isinstance(s[f], str):
                errors.append(
                    f"{s['gloss']}.{f} parsed as {type(s[f]).__name__}, not str "
                    f"(quote it)")

    # --- fields the downstream phases depend on ---------------------------
    for s in signs:
        if not s.get("english_lemmas"):
            errors.append(f"{s['gloss']}: missing english_lemmas (5b lexicon)")
        if s.get("lexical_class") == "Verb" and not s.get("verb_class"):
            warnings.append(f"{s['gloss']}: Verb without verb_class tag")

    # --- ASL-LEX / ASL Citizen join key (tools/resolve_keys.py) -----------
    # Every sign must carry an asllex_code, codes must be unique, and each must
    # resolve in the Citizen splits -- otherwise the Phase 1 video join breaks.
    codes = []
    for s in signs:
        code = s.get("asllex_code")
        if not code:
            errors.append(f"{s['gloss']}: missing asllex_code "
                          f"(run tools/resolve_keys.py)")
        else:
            codes.append(code)
    dup_codes = [c for c, n in Counter(codes).items() if n > 1]
    if dup_codes:
        errors.append(f"asllex_code not unique: {dup_codes}")

    citizen_codes = load_citizen_codes()
    if citizen_codes is None:
        warnings.append("ASL Citizen splits not found -- skipped asllex_code "
                        "resolution check (download data/ASL_Citizen)")
    else:
        for s in signs:
            code = s.get("asllex_code")
            if code and code not in citizen_codes:
                errors.append(f"{s['gloss']}: asllex_code {code} does not resolve "
                              f"in the ASL Citizen splits")

    # auto-selected ambiguous variants still awaiting a human eyeball
    for s in signs:
        if s.get("variant_auto_selected") and not s.get("variant_reviewed"):
            warnings.append(f"{s['gloss']}: variant auto-selected "
                            f"({s.get('asl_citizen_gloss')}) but variant_reviewed "
                            f"is false -- eyeball the reference video")

    # --- coverage ---------------------------------------------------------
    cov = d.get("coverage", {})
    real_major = dict(Counter(s["major_location"] for s in signs))
    real_move = dict(Counter(s["movement"] for s in signs))
    real_hs = len({s["handshape"] for s in signs})
    if cov.get("major_location") != real_major:
        errors.append(f"coverage.major_location stale: {cov.get('major_location')} "
                      f"!= {real_major}")
    if cov.get("movement") != real_move:
        errors.append(f"coverage.movement stale: {cov.get('movement')} != {real_move}")
    if cov.get("distinct_handshapes") != real_hs:
        errors.append(f"coverage.distinct_handshapes stale: "
                      f"{cov.get('distinct_handshapes')} != {real_hs}")

    # --- contrastive pairs -------------------------------------------------
    derived = derive_pairs(signs)
    by_gloss = {s["gloss"]: s for s in signs}
    for f, plist in (d.get("contrastive_pairs") or {}).items():
        for entry in plist:
            x, y = (str(v) for v in entry["pair"])
            if x not in by_gloss or y not in by_gloss:
                errors.append(f"pair [{x}, {y}] references an unknown gloss")
                continue
            if not minimal_on(by_gloss[x], by_gloss[y], f):
                diffs = [g for g in FIELDS if by_gloss[x][g] != by_gloss[y][g]]
                errors.append(
                    f"pair [{x}, {y}] claimed minimal on {f} but differs on {diffs}")

    n_listed = sum(len(v) for v in (d.get("contrastive_pairs") or {}).values())
    n_derived = sum(len(v) for v in derived.values())
    if n_listed != n_derived:
        warnings.append(f"contrastive_pairs has {n_listed} pairs; {n_derived} exist "
                        f"-- regenerate with --pairs")

    for f, plist in derived.items():
        if len(plist) <= 1:
            warnings.append(f"only {len(plist)} minimal pair(s) for '{f}' -- thin "
                            f"drill material for that parameter")

    # --- construction examples --------------------------------------------
    ids = {g.upper() for g in glosses}
    allowed = ids | {"NO", "OR"}
    for c in d.get("constructions_v2", []):
        for tok in re.findall(r"[A-Z_0-9]{2,}", c["example_gloss"]):
            if tok not in allowed:
                errors.append(f"construction '{c['id']}': example gloss token "
                              f"'{tok}' is not an ID-gloss in this curriculum")

    # --- report ------------------------------------------------------------
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    if args.pairs:
        print("\ncontrastive_pairs:")
        for f, plist in derived.items():
            print(f"  {f}:   # {len(plist)}")
            for x, y, av, bv in plist:
                print(f'    - pair: ["{x}", "{y}]"'.replace('"]"', '"]'))
                print(f'      differs: "{av} vs {bv}"')

    if errors:
        print(f"\nFAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"\nOK: {len(signs)} signs, {n_derived} minimal pairs, "
          f"{len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
