#!/usr/bin/env python3
"""Filter ASL Citizen's official splits down to the curriculum and emit a manifest.

Reads data/ASL_Citizen/splits/{train,val,test}.csv, keeps only the rows whose
`ASL-LEX Code` is one of the 60 codes in curriculum.yaml (the join key written by
tools/resolve_keys.py), and writes data/manifest.csv -- one row per reference
video for the extraction pipeline (Phase 1).

The official ASL Citizen splits are used AS-IS: they are already
signer-independent (35 / 6 / 11 signers, no signer in more than one split), so we
never re-split.

manifest.csv columns:
    video_id       stem of the Citizen `Video file` (unique per clip)
    video_path     data/ASL_Citizen/videos/<file>  (repo-relative)
    asllex_code    ASL-LEX Code (the stable join key)
    id_gloss       curriculum gloss (e.g. "dog", "how")
    citizen_gloss  ASL Citizen's Gloss label (e.g. "DOG1", "HOW2")
    signer_id      Citizen `Participant ID`
    split          train | val | test

On run it asserts and prints:
    (a) no signer appears in more than one split (signer-independence),
    (b) kept-row counts match the expected per-split totals,
    (c) every video_path exists under data/ASL_Citizen/videos/ (lists any missing).
Any failed assertion prints a clear diagnosis and exits non-zero.

    python tools/build_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CURRICULUM = REPO / "curriculum.yaml"
SPLITS_DIR = REPO / "data" / "ASL_Citizen" / "splits"
VIDEOS_DIR = REPO / "data" / "ASL_Citizen" / "videos"
OUTPUT = REPO / "data" / "manifest.csv"
SPLITS = ("train", "val", "test")

# Expected kept-row counts for the current 60-sign curriculum (all codes,
# including the 5 human-reviewed ambiguous variants). Update these if the
# curriculum's sign set changes -- a mismatch means the filter drifted.
EXPECTED = {"train": 895, "val": 229, "test": 750}

MANIFEST_COLUMNS = ["video_id", "video_path", "asllex_code", "id_gloss",
                    "citizen_gloss", "signer_id", "split"]

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"\nFATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    """Read a CSV, auto-detecting encoding (Citizen splits are utf-8/cp1252)."""
    for enc in ENCODINGS:
        try:
            with open(path, newline="", encoding=enc) as f:
                rd = csv.DictReader(f)
                return list(rd), (rd.fieldnames or [])
        except UnicodeDecodeError:
            continue
    die(f"could not decode {path} with any of {ENCODINGS}")


def find_col(header: list[str], *, equals=None, contains=None, startswith=None) -> str:
    """Find a column case-insensitively; fail loudly if absent."""
    for h in header:
        hl = h.strip().lower()
        if equals and hl == equals.lower():
            return h
        if contains and contains.lower() in hl:
            return h
        if startswith and hl.startswith(startswith.lower()):
            return h
    die(f"no column matching {equals or contains or startswith!r} in {header}")


def load_curriculum_codes() -> dict[str, str]:
    """Return {asllex_code: id_gloss} for the 60 curriculum signs."""
    d = yaml.safe_load(open(CURRICULUM, encoding="utf-8"))
    signs = [s for u in d["units"] for s in u["signs"]]
    code_to_gloss: dict[str, str] = {}
    missing = [s["gloss"] for s in signs if not s.get("asllex_code")]
    if missing:
        die(f"{len(missing)} sign(s) lack asllex_code "
            f"(run tools/resolve_keys.py first): {missing}")
    for s in signs:
        code = str(s["asllex_code"])
        if code in code_to_gloss:
            die(f"asllex_code {code} is used by >1 sign "
                f"({code_to_gloss[code]!r} and {s['gloss']!r})")
        code_to_gloss[code] = str(s["gloss"])
    print(f"curriculum: {len(code_to_gloss)} signs, {len(code_to_gloss)} unique codes")
    return code_to_gloss


def build() -> tuple[list[dict], dict[str, set]]:
    """Filter the splits to the curriculum; return (manifest rows, signers/split)."""
    code_to_gloss = load_curriculum_codes()
    rows_out: list[dict] = []
    signers_by_split: dict[str, set] = {sp: set() for sp in SPLITS}

    for sp in SPLITS:
        path = SPLITS_DIR / f"{sp}.csv"
        if not path.is_file():
            die(f"missing split file: {path}")
        rows, header = read_rows(path)
        c_part = find_col(header, startswith="participant")
        c_file = find_col(header, startswith="video")
        c_gloss = find_col(header, equals="gloss")
        c_code = find_col(header, contains="code")
        for r in rows:
            signers_by_split[sp].add(r[c_part].strip())  # full-split signer set
            code = r[c_code].strip()
            if code not in code_to_gloss:
                continue
            vfile = r[c_file].strip()
            rows_out.append({
                "video_id": Path(vfile).stem,
                "video_path": f"data/ASL_Citizen/videos/{vfile}",
                "asllex_code": code,
                "id_gloss": code_to_gloss[code],
                "citizen_gloss": r[c_gloss].strip(),
                "signer_id": r[c_part].strip(),
                "split": sp,
            })
    return rows_out, signers_by_split


def check_signer_independence(signers_by_split: dict[str, set]) -> bool:
    print("\n(a) signer independence (official splits used as-is)")
    for sp in SPLITS:
        print(f"    {sp:<5} signers: {len(signers_by_split[sp])}")
    ok = True
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = signers_by_split[a] & signers_by_split[b]
            if overlap:
                ok = False
                print(f"    !! OVERLAP {a} & {b}: {sorted(overlap)}")
    print("    -> PASS: no signer in more than one split" if ok
          else "    -> FAIL: signer(s) span multiple splits")
    return ok


def check_counts(rows: list[dict]) -> bool:
    got = Counter(r["split"] for r in rows)
    print("\n(b) kept-row counts")
    print(f"    {'split':<7}{'got':>6}{'expected':>10}")
    ok = True
    for sp in SPLITS:
        exp = EXPECTED[sp]
        flag = "" if got[sp] == exp else "  <-- MISMATCH"
        if got[sp] != exp:
            ok = False
        print(f"    {sp:<7}{got[sp]:>6}{exp:>10}{flag}")
    print(f"    {'total':<7}{sum(got.values()):>6}{sum(EXPECTED.values()):>10}")
    print("    -> PASS: counts match expected" if ok
          else "    -> FAIL: counts drifted from EXPECTED (update EXPECTED after "
               "a curriculum change, or investigate the filter)")
    return ok


def check_videos_exist(rows: list[dict]) -> bool:
    print("\n(c) video files present under", VIDEOS_DIR.relative_to(REPO))
    if not VIDEOS_DIR.is_dir():
        print(f"    !! videos directory not found: {VIDEOS_DIR}")
        return False
    present = set(p.name for p in VIDEOS_DIR.iterdir())
    missing = [r for r in rows if Path(r["video_path"]).name not in present]
    print(f"    referenced: {len(rows)}  present: {len(rows) - len(missing)}  "
          f"missing: {len(missing)}")
    for r in missing:
        print(f"    !! MISSING {r['video_path']}  ({r['id_gloss']}/{r['citizen_gloss']})")
    print("    -> PASS: every video_path exists" if not missing
          else f"    -> FAIL: {len(missing)} video(s) missing")
    return not missing


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUTPUT), help="manifest output path")
    ap.add_argument("--dry-run", action="store_true",
                    help="run checks but do not write the manifest")
    args = ap.parse_args()

    rows, signers_by_split = build()
    print(f"\nkept {len(rows)} rows across {len(SPLITS)} splits")

    ok_a = check_signer_independence(signers_by_split)
    ok_b = check_counts(rows)
    ok_c = check_videos_exist(rows)

    if args.dry_run:
        print("\n[--dry-run] manifest NOT written")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out.relative_to(REPO)}  ({len(rows)} rows)")

    if not (ok_a and ok_b and ok_c):
        print("\nFAILED: one or more assertions did not hold")
        return 1
    print("\nOK: all assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
