#!/usr/bin/env python3
"""Give curriculum.yaml a stable join key onto ASL-LEX and ASL Citizen.

Every curriculum sign carries an ``asllex_id`` -- an ASL-LEX 2.0 *EntryID*
string such as ``want_2`` or ``dog``. That string is NOT a durable join key:

  * ASL Citizen keys its videos on the ASL-LEX *Code* (e.g. ``A_01_056``), not
    on the EntryID, and
  * for a handful of English words ASL-LEX / Citizen split one gloss into
    several distinct signs (Citizen has DOG1..DOG4, BABY1/2, ...), so an
    ``asllex_id`` like ``dog`` does not name exactly one Citizen video set.

This script resolves the join and writes it back into curriculum.yaml:

  1. resolve each ``asllex_id`` -> ASL-LEX ``Code`` via the ASL-LEX sign table,
     writing ``asllex_code`` onto every sign;
  2. verify the Code appears in the ASL Citizen splits and record the Citizen
     ``Gloss`` as ``asl_citizen_gloss``;
  3. for the AMBIGUOUS glosses (baby, dog, drink, how, milk) enumerate every
     Citizen variant with its Code / SignFrequency / PercentUnknown / video
     count / ASL-LEX example-video path, rank by SignFrequency descending,
     auto-select the most frequent variant whose PercentUnknown <= 0.10, and
     flag it (``variant_auto_selected: true``, ``variant_reviewed: false``) so a
     human eyeballs the reference video before Phase 1;
  4. write curriculum.yaml back in place, comment-preserving, via ruamel.yaml;
  5. flip ``status.asl_citizen_confirmed`` to true IFF all 60 signs resolve to a
     unique Code that is present in the Citizen splits.

Column names are NOT hard-coded. The ASL-LEX sign table is discovered by
scanning ``data/ASL_LEX`` and its header is matched case-insensitively against
known aliases; the same is done for the Citizen ``Gloss`` / ``ASL-LEX Code``
columns. If a required column cannot be found the script fails loudly.

    python tools/resolve_keys.py            # resolve and write in place
    python tools/resolve_keys.py --dry-run  # print everything, write nothing
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import CommentMark
from ruamel.yaml.tokens import CommentToken

REPO = Path(__file__).resolve().parents[1]
CURRICULUM = REPO / "curriculum.yaml"
ASLLEX_DIR = REPO / "data" / "ASL_LEX"
ASLLEX_EXAMPLES = ASLLEX_DIR / "ASL examples"
CITIZEN_SPLITS = REPO / "data" / "ASL_Citizen" / "splits"

# The curated set of glosses whose English word maps to several distinct signs.
# The *candidates* for each are discovered from the Citizen splits, not listed
# here -- only membership in this set is a curation decision.
AMBIGUOUS = {"baby", "dog", "drink", "how", "milk"}

# A variant is only auto-selectable if this fraction (or fewer) of ASL signers
# reported not knowing the sign. ASL-LEX's PercentUnknown is a proportion [0, 1].
UNKNOWN_THRESHOLD = 0.10

# Case-insensitive EXACT header aliases (exact so "SignFrequency(M)" never
# matches "SignFrequency(M-Native)", nor "Unknown" match "Unknown(Native)").
# Ordered by preference.
ENTRYID_ALIASES = ["EntryID", "Entry_ID", "EntryId", "Entry ID"]
CODE_ALIASES = ["Code", "SurveyCode", "Code (survey)"]
FREQ_ALIASES = ["SignFrequency(M)", "SignFrequency", "SignFreq(M)", "SignFrequencyM"]
UNKNOWN_ALIASES = ["PercentUnknown", "Percent_Unknown", "Unknown", "%Unknown"]
GLOSS_ALIASES = ["Gloss", "ASL Citizen Gloss", "Sign"]
CITIZEN_CODE_ALIASES = ["ASL-LEX Code", "ASL_LEX_Code", "Code", "ASLLEXCode"]

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"\nFATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def to_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_header(path: Path) -> list[str]:
    """Read only the first row -- safe even for multi-GB files.

    Opened with newline='' so the csv module handles \\n / \\r / \\r\\n and
    quoted newlines itself; encoding is auto-detected (ASL-LEX files are cp1252).
    """
    for enc in ENCODINGS:
        try:
            with open(path, newline="", encoding=enc) as f:
                return next(csv.reader(f))
        except UnicodeDecodeError:
            continue
    die(f"could not decode {path} header with any of {ENCODINGS}")


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    """Read a whole (small) CSV. Never call this on the multi-GB neighbor file."""
    for enc in ENCODINGS:
        try:
            with open(path, newline="", encoding=enc) as f:
                rd = csv.DictReader(f)
                return list(rd), (rd.fieldnames or [])
        except UnicodeDecodeError:
            continue
    die(f"could not decode {path} with any of {ENCODINGS}")


def find_col(header: list[str], aliases: list[str]) -> str | None:
    """Return the header entry matching an alias case-insensitively (exact)."""
    lower = {h.strip().lower(): h for h in header}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "?"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# --------------------------------------------------------------------------- #
# ASL-LEX sign table (discovered, not hard-coded)
# --------------------------------------------------------------------------- #
@dataclass
class SignRec:
    entry_id: str
    code: str
    freq: float | None
    unknown: float | None


def build_asllex_index() -> tuple[dict[str, SignRec], dict[str, str]]:
    """Scan data/ASL_LEX, print every CSV header, and build EntryID/Code indices.

    Returns (by_entry_id, code_to_entry_id). Fails loudly if the four required
    columns (EntryID, Code, SignFrequency, PercentUnknown) cannot be sourced.
    """
    if not ASLLEX_DIR.is_dir():
        die(f"ASL-LEX directory not found: {ASLLEX_DIR}")

    csvs = sorted(ASLLEX_DIR.rglob("*.csv"))
    if not csvs:
        die(f"no CSV files under {ASLLEX_DIR}")

    print("=" * 78)
    print("ASL-LEX: inspecting CSV headers under", ASLLEX_DIR.relative_to(REPO))
    print("=" * 78)

    # For each file, record which required columns it carries. EntryID is the
    # join column, so code/freq/unknown are only usable from a file that ALSO
    # carries EntryID.
    src = {"entry": None, "code": None, "freq": None, "unknown": None}
    col = {"entry": None, "code": None, "freq": None, "unknown": None}
    LABELS = {"entry": "EntryID", "code": "Code",
              "freq": "SignFrequency", "unknown": "PercentUnknown"}
    single = None  # a file carrying all four columns -> authoritative sign table
    for path in csvs:
        header = read_header(path)
        rel = path.relative_to(REPO)
        shown = ", ".join(header[:16]) + (
            f"  ... (+{len(header) - 16} more)" if len(header) > 16 else "")
        print(f"\n  {rel}  [{len(header)} cols]\n    {shown}")
        found = {"entry": find_col(header, ENTRYID_ALIASES),
                 "code": find_col(header, CODE_ALIASES),
                 "freq": find_col(header, FREQ_ALIASES),
                 "unknown": find_col(header, UNKNOWN_ALIASES)}
        matched = [f"{LABELS[k]}={v!r}" for k, v in found.items() if v]
        print(f"    matched: {', '.join(matched) if matched else '(none)'}")
        if found["entry"]:
            for k in ("entry", "code", "freq", "unknown"):
                if found[k] and src[k] is None and (k == "entry" or found["entry"]):
                    src[k], col[k] = path, found[k]
            if single is None and all(found[k] for k in found):
                single = (path, found)

    if single is not None:
        # Prefer one authoritative table (e.g. ASL-LEX signdata.csv) over
        # stitching aggregates from several files -- avoids reading a raw
        # trial-level file's per-trial Unknown flag as if it were a proportion.
        path, found = single
        for k in ("entry", "code", "freq", "unknown"):
            src[k], col[k] = path, found[k]
        print(f"\n  --> single authoritative table: {path.relative_to(REPO)}")

    missing = [k for k, v in src.items() if v is None]
    if missing:
        die("could not find a column for: "
            + ", ".join(LABELS[m] for m in missing)
            + f"\n  searched: {[str(p.relative_to(REPO)) for p in csvs]}")

    print("\n  --> column sources:")
    for k, label in (("code", "Code"), ("freq", "SignFrequency"),
                     ("unknown", "PercentUnknown")):
        print(f"      {label:<14} {col[k]!r:<18} from {src[k].relative_to(REPO)}")

    # Load each distinct source file once, join on EntryID.
    loaded: dict[Path, list[dict]] = {}
    for p in {src["code"], src["freq"], src["unknown"]}:
        loaded[p], _ = read_rows(p)

    by_eid: dict[str, SignRec] = {}
    dup_eids: set[str] = set()
    code_rows = loaded[src["code"]]
    freq_by_eid = {r[col["entry"]]: r for r in loaded[src["freq"]]}
    unk_by_eid = {r[col["entry"]]: r for r in loaded[src["unknown"]]}
    for r in code_rows:
        eid = r[col["entry"]]
        code = r[col["code"]]
        freq = to_float(freq_by_eid.get(eid, {}).get(col["freq"]))
        unk = to_float(unk_by_eid.get(eid, {}).get(col["unknown"]))
        if eid in by_eid and by_eid[eid].code != code:
            dup_eids.add(eid)  # EntryID -> multiple Codes; ambiguous at source
        by_eid[eid] = SignRec(eid, code, freq, unk)

    code_to_eid: dict[str, str] = {}
    for r in code_rows:
        code_to_eid.setdefault(r[col["code"]], r[col["entry"]])

    print(f"\n  loaded {len(code_rows)} ASL-LEX rows "
          f"({len(by_eid)} unique EntryID, {len(code_to_eid)} unique Code)")
    if dup_eids:
        print(f"  note: {len(dup_eids)} EntryID(s) map to >1 Code "
              f"(will error only if a curriculum sign hits one)")

    build_asllex_index.dup_eids = dup_eids  # type: ignore[attr-defined]
    return by_eid, code_to_eid


# --------------------------------------------------------------------------- #
# ASL Citizen splits
# --------------------------------------------------------------------------- #
def build_citizen_index() -> tuple[dict[str, Counter], dict[str, str], set[str]]:
    """Return (code -> Counter(gloss:videos), gloss -> code, all glosses)."""
    if not CITIZEN_SPLITS.is_dir():
        die(f"ASL Citizen splits directory not found: {CITIZEN_SPLITS}")
    files = sorted(CITIZEN_SPLITS.glob("*.csv"))
    if not files:
        die(f"no split CSVs under {CITIZEN_SPLITS}")

    print("\n" + "=" * 78)
    print("ASL Citizen: inspecting split headers under",
          CITIZEN_SPLITS.relative_to(REPO))
    print("=" * 78)

    code_to_gloss: dict[str, Counter] = {}
    gloss_to_codes: dict[str, Counter] = {}
    total = 0
    for path in files:
        rows, header = read_rows(path)
        gcol = find_col(header, GLOSS_ALIASES)
        ccol = find_col(header, CITIZEN_CODE_ALIASES)
        print(f"\n  {path.name}  [{len(rows)} rows]  {header}")
        if not gcol or not ccol:
            die(f"{path.name}: could not find Gloss/Code columns "
                f"(gloss={gcol!r}, code={ccol!r})")
        print(f"    matched: Gloss={gcol!r}, Code={ccol!r}")
        for r in rows:
            gloss = (r[gcol] or "").strip()
            code = (r[ccol] or "").strip()
            if not gloss or not code:
                continue
            code_to_gloss.setdefault(code, Counter())[gloss] += 1
            gloss_to_codes.setdefault(gloss, Counter())[code] += 1
            total += 1

    gloss_to_code = {g: c.most_common(1)[0][0] for g, c in gloss_to_codes.items()}
    print(f"\n  loaded {total} Citizen videos across {len(files)} split(s): "
          f"{len(code_to_gloss)} codes, {len(gloss_to_code)} glosses")
    return code_to_gloss, gloss_to_code, set(gloss_to_code)


# --------------------------------------------------------------------------- #
# example-video lookup
# --------------------------------------------------------------------------- #
def build_example_index() -> dict[str, str]:
    """{EntryID.upper(): 'data/.../FILE.webm'} for ASL-LEX example clips."""
    if not ASLLEX_EXAMPLES.is_dir():
        print(f"  note: no ASL example dir at {ASLLEX_EXAMPLES.relative_to(REPO)}")
        return {}
    out = {}
    for p in ASLLEX_EXAMPLES.glob("*.webm"):
        out[p.stem.upper()] = str(p.relative_to(REPO))
    return out


def example_for(entry_id: str, examples: dict[str, str]) -> str | None:
    return examples.get((entry_id or "").upper())


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
@dataclass
class Resolved:
    gloss: str
    asllex_id: str
    ambiguous: bool
    code: str | None = None
    citizen_gloss: str | None = None
    error: str | None = None
    candidates: list[dict] = field(default_factory=list)  # ambiguous only
    base_code: str | None = None  # code of asllex_id itself (ambiguous only)


def resolve_sign(sign, by_eid, code_to_eid, code_to_gloss, gloss_to_code,
                 all_glosses, examples) -> Resolved:
    aid = str(sign["asllex_id"])
    gloss = str(sign["gloss"])
    dup_eids = getattr(build_asllex_index, "dup_eids", set())

    if aid in AMBIGUOUS:
        return resolve_ambiguous(gloss, aid, by_eid, code_to_eid, code_to_gloss,
                                 gloss_to_code, all_glosses, examples)

    r = Resolved(gloss, aid, ambiguous=False)
    if aid not in by_eid:
        r.error = f"asllex_id {aid!r} not found in ASL-LEX EntryID column"
        return r
    if aid in dup_eids:
        r.error = f"asllex_id {aid!r} maps to multiple ASL-LEX Codes (ambiguous)"
        return r
    r.code = by_eid[aid].code
    glosses = code_to_gloss.get(r.code)
    if not glosses:
        r.error = f"Code {r.code} (from {aid!r}) not present in ASL Citizen splits"
        return r
    if len(glosses) > 1:
        # one Code -> several Citizen gloss strings; take the most common
        print(f"  WARN {gloss}: Code {r.code} has multiple Citizen glosses "
              f"{dict(glosses)}; using most common")
    r.citizen_gloss = glosses.most_common(1)[0][0]
    return r


def resolve_ambiguous(gloss, aid, by_eid, code_to_eid, code_to_gloss,
                      gloss_to_code, all_glosses, examples) -> Resolved:
    r = Resolved(gloss, aid, ambiguous=True)
    r.base_code = by_eid[aid].code if aid in by_eid else None

    base = aid.upper()
    variants = sorted(g for g in all_glosses if re.fullmatch(base + r"\d*", g))
    if not variants:
        r.error = f"no ASL Citizen variants matching {base!r}*"
        return r

    cands = []
    for vg in variants:
        code = gloss_to_code[vg]
        eid = code_to_eid.get(code)
        rec = by_eid.get(eid) if eid else None
        vids = int(code_to_gloss.get(code, Counter()).get(vg, 0))
        cands.append({
            "citizen_gloss": vg,
            "asllex_code": code,
            "asllex_entry_id": eid,
            "sign_frequency": rec.freq if rec else None,
            "percent_unknown": rec.unknown if rec else None,
            "citizen_video_count": vids,
            "asllex_example_video": example_for(eid, examples) if eid else None,
            "selected": False,
        })

    # rank by SignFrequency descending (missing frequency ranks last)
    cands.sort(key=lambda c: (c["sign_frequency"] is None,
                              -(c["sign_frequency"] or 0.0)))
    r.candidates = cands

    eligible = [c for c in cands
                if c["percent_unknown"] is not None
                and c["percent_unknown"] <= UNKNOWN_THRESHOLD]
    if not eligible:
        r.error = (f"no {base} variant has PercentUnknown <= {UNKNOWN_THRESHOLD} "
                   f"-- nothing auto-selectable")
        return r
    chosen = max(eligible, key=lambda c: (c["sign_frequency"] or 0.0))
    chosen["selected"] = True
    r.code = chosen["asllex_code"]
    r.citizen_gloss = chosen["citizen_gloss"]
    return r


# --------------------------------------------------------------------------- #
# writing back into the YAML tree (comment-preserving)
# --------------------------------------------------------------------------- #
def set_after(cm: CommentedMap, anchor: str, key: str, value) -> None:
    """Set cm[key]=value, inserting it directly after `anchor` if new."""
    if key in cm:
        cm[key] = value
        return
    keys = list(cm.keys())
    pos = keys.index(anchor) + 1 if anchor in keys else len(keys)
    cm.insert(pos, key, value)


def candidate_block(cands: list[dict]) -> CommentedSeq:
    seq = CommentedSeq()
    for c in cands:
        m = CommentedMap()
        m["citizen_gloss"] = c["citizen_gloss"]
        m["asllex_code"] = c["asllex_code"]
        m["asllex_entry_id"] = c["asllex_entry_id"]
        m["sign_frequency"] = (round(c["sign_frequency"], 3)
                               if c["sign_frequency"] is not None else None)
        m["percent_unknown"] = (round(c["percent_unknown"], 3)
                                if c["percent_unknown"] is not None else None)
        m["citizen_video_count"] = c["citizen_video_count"]
        m["asllex_example_video"] = c["asllex_example_video"]
        m["selected"] = c["selected"]
        seq.append(m)
    return seq


def set_eol_comment(cm: CommentedMap, key: str, text: str) -> None:
    """Replace the end-of-line comment on `key` with a single-line comment."""
    tok = CommentToken("# " + text + "\n", CommentMark(0), None)
    cm.ca.items[key] = [None, None, tok, None]


def write_back(sign, r: Resolved) -> None:
    set_after(sign, "asllex_id", "asllex_code", r.code)
    set_after(sign, "asllex_code", "asl_citizen_gloss", r.citizen_gloss)
    if r.ambiguous:
        set_after(sign, "asl_citizen_gloss", "variant_auto_selected", True)
        set_after(sign, "variant_auto_selected", "variant_reviewed", False)
        set_after(sign, "variant_reviewed", "variant_candidates",
                  candidate_block(r.candidates))


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def print_ambiguous_tables(results: list[Resolved]) -> None:
    print("\n" + "=" * 78)
    print("AMBIGUOUS SIGNS  (auto-select = highest SignFrequency with "
          f"PercentUnknown <= {UNKNOWN_THRESHOLD})")
    print("=" * 78)
    for r in results:
        if not r.ambiguous:
            continue
        n = len(r.candidates)
        print(f"\n  {r.asllex_id.upper()}  (curriculum asllex_id: {r.asllex_id!r}) "
              f"-- {n} Citizen variant(s)")
        print(f"    {'#':<3}{'citizen':<9}{'code':<11}{'entry_id':<12}"
              f"{'freq':>7} {'%unk':>7} {'vids':>6}  {'example':<38}sel")
        for i, c in enumerate(r.candidates, 1):
            ex = c["asllex_example_video"] or "-"
            if len(ex) > 37:
                ex = "..." + ex[-34:]
            sel = "<== AUTO" if c["selected"] else ""
            print(f"    {i:<3}{c['citizen_gloss']:<9}{str(c['asllex_code']):<11}"
                  f"{str(c['asllex_entry_id']):<12}{fmt(c['sign_frequency']):>7} "
                  f"{fmt(c['percent_unknown']):>7} {c['citizen_video_count']:>6}  "
                  f"{ex:<38}{sel}")
        if r.error:
            print(f"    !! UNRESOLVED: {r.error}")
        elif r.base_code and r.code != r.base_code:
            print(f"    !! NOTE: auto-selected Code {r.code} differs from the "
                  f"asllex_id's own Code {r.base_code}")
            print(f"       (curriculum said {r.asllex_id!r}; frequency prefers "
                  f"{next(c['asllex_entry_id'] for c in r.candidates if c['selected'])!r}) "
                  f"-- review the video.")


def print_summary(results: list[Resolved], confirmed: bool) -> None:
    print("\n" + "=" * 78)
    print("RESOLUTION SUMMARY  (all 60 signs)")
    print("=" * 78)
    print(f"  {'gloss':<12}{'asllex_id':<12}{'asllex_code':<12}"
          f"{'asl_citizen_gloss':<18}note")
    for r in results:
        note = ""
        if r.error:
            note = f"UNRESOLVED: {r.error}"
        elif r.ambiguous:
            note = "auto-selected variant (review)"
        print(f"  {r.gloss:<12}{r.asllex_id:<12}{str(r.code):<12}"
              f"{str(r.citizen_gloss):<18}{note}")

    resolved = [r for r in results if not r.error]
    codes = [r.code for r in resolved]
    dup = [c for c, n in Counter(codes).items() if n > 1]
    print("\n  signs total           :", len(results))
    print("  resolved              :", f"{len(resolved)}/{len(results)}")
    print("  duplicate codes       :", dup or "none")
    unresolved = [r.gloss for r in results if r.error]
    print("  unresolved            :", unresolved or "none")
    print("  ambiguous auto-select :",
          ", ".join(f"{r.asllex_id}->{r.citizen_gloss}"
                    for r in results if r.ambiguous and not r.error))
    print("  asl_citizen_confirmed :", confirmed)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(CURRICULUM))
    ap.add_argument("--dry-run", action="store_true",
                    help="print everything but do not write curriculum.yaml")
    args = ap.parse_args()

    by_eid, code_to_eid = build_asllex_index()
    code_to_gloss, gloss_to_code, all_glosses = build_citizen_index()
    examples = build_example_index()

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    path = Path(args.path)
    with open(path, encoding="utf-8") as f:
        doc = yaml.load(f)
    signs = [s for u in doc["units"] for s in u["signs"]]
    if len(signs) != doc.get("count"):
        print(f"  WARN: count={doc.get('count')} but found {len(signs)} signs")

    results = []
    for sign in signs:
        r = resolve_sign(sign, by_eid, code_to_eid, code_to_gloss,
                         gloss_to_code, all_glosses, examples)
        results.append((sign, r))

    print_ambiguous_tables([r for _, r in results])

    # all resolve + unique codes -> confirmed
    resolved = [r for _, r in results if not r.error]
    codes = [r.code for r in resolved]
    dup = [c for c, n in Counter(codes).items() if n > 1]
    confirmed = (len(resolved) == len(results) == 60) and not dup

    # write into the tree
    for sign, r in results:
        if not r.error:
            write_back(sign, r)

    status = doc.get("status")
    if isinstance(status, CommentedMap):
        status["asl_citizen_confirmed"] = confirmed
        if confirmed:
            set_eol_comment(
                status, "asl_citizen_confirmed",
                "confirmed by tools/resolve_keys.py: all 60 signs resolve to a "
                "unique ASL-LEX Code present in the ASL Citizen splits. "
                "5 signs auto-selected a Citizen variant -- see variant_reviewed.")

    print_summary([r for _, r in results], confirmed)

    if args.dry_run:
        print("\n[--dry-run] curriculum.yaml NOT written")
        return 0 if confirmed else 1

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f)
    print(f"\nwrote {path}")
    if not confirmed:
        print("status.asl_citizen_confirmed left/!set false "
              "(not all 60 signs resolved uniquely)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
