#!/usr/bin/env python3
"""export_review_sheet.py -- generate a self-contained HTML artifact so a
fluent Deaf signer can review the gloss rule engine's PENDING_CASES (and the
five constructions_v2 examples they cover) in one sitting, without touching
Python or the codebase.

For each case it shows: the English input, the produced gloss sequence with
non-manual marker SPANS rendered visually (a bar under the glosses it covers,
not a raw tag dump), the ordered reference clips each gloss resolves to
(embedded inline as playable video -- these are the actual ASL Citizen clips
`compose_sentence.py` would retrieve), a plain-language restatement of the
engine's own `:why` trace (which rule fired, in English), and a verdict form
(correct / wrong-order / wrong-NMM / other + a comment box).

This script does NOT judge ASL correctness -- it cannot. It only presents the
engine's output faithfully so a human who CAN judge it doesn't have to run
code to do so. The output HTML persists verdicts to the browser's
localStorage as the reviewer works, and has an "Export verdicts" button that
downloads a JSON file mapping each case's English text to its verdict +
comment -- that JSON is what turns a PendingCase.reviewed=False in
tests/test_gloss_rules_corpus.py into reviewed=True with expected_glosses
filled in.

    .venv/bin/python scripts/export_review_sheet.py
    .venv/bin/python scripts/export_review_sheet.py --out /tmp/review.html

Output defaults under data/ (gitignored) since the embedded video makes the
file itself ~15-20MB -- not something to check in.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from aslcv.production.gloss_rules import GlossRuleEngine, GlossSequence  # noqa: E402
from test_gloss_rules_corpus import CONSTRUCTIONS_V2, PENDING_CASES, PendingCase  # noqa: E402

MANIFEST = REPO / "data" / "manifest.csv"
OUT_DEFAULT = REPO / "data" / "review" / "gloss_review_sheet.html"

# topic_comment is opt-in (GlossRuleEngine.topicalize), never applied
# automatically -- plain engine.gloss("I want coffee.") does NOT front COFFEE
# or add brow_raise. This mirrors exactly what
# test_topic_comment_fronts_topic_and_brow_raises does in the corpus suite,
# so the review sheet shows the SAME transform the test suite already checks
# mechanically, not a different one. Keyed by english text since it's the
# only PENDING_CASES entry needing this special handling today.
TOPICALIZE_TARGET = {"I want coffee.": "coffee"}

# seq.trace entries look like "scope check: no unsupported dependency labels"
# -- restate the machine-facing prefix as a phrase a non-programmer reads
# comfortably, without changing or re-deriving the underlying claim.
_TRACE_LABELS = [
    ("scope check:", "Grammar-scope check"),
    ("lexicalize:", "Word-to-sign lookup"),
    ("classify:", "Sentence type"),
    ("reorder:", "Word-order rule"),
    ("tag:", "Non-manual marker rule"),
    ("confidence=", "Engine's own confidence heuristic"),
    ("refused:", "Refusal"),
    ("topicalize(", "Topic-comment transform"),
]


def _humanize_trace(seq: GlossSequence) -> list[str]:
    """Plain-language restatement of seq.trace -- the same log
    scripts/gloss_repl.py's `:why` prints -- skipping the raw input echo and
    the raw spaCy token dump (too technical for a reviewer who isn't reading
    code), relabeling the rest so a fluent signer sees WHAT RULE fired and
    WHY, in English."""
    notes = []
    for line in seq.trace:
        if line.startswith("input:") or line.startswith("spacy:"):
            continue
        label = next((lbl for prefix, lbl in _TRACE_LABELS if line.startswith(prefix)), None)
        if label is None:
            notes.append(line)
            continue
        rest = line.split(":", 1)[1].strip() if ":" in line else line
        notes.append(f"{label} -- {rest}")
    return notes


def _construction_for(english: str) -> dict | None:
    return next((c for c in CONSTRUCTIONS_V2 if c["example_english"] == english), None)


def _load_manifest_by_gloss() -> dict[str, list[dict]]:
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_gloss: dict[str, list[dict]] = {}
    for r in rows:
        by_gloss.setdefault(r["id_gloss"], []).append(r)
    return by_gloss


def _pick_clip(rows: list[dict]) -> dict:
    """Same deterministic default as compose_sentence.py/render_clip.py:
    prefer a train-split clip, else the first match."""
    return next((r for r in rows if r["split"] == "train"), rows[0])


def build_case(engine: GlossRuleEngine, case: PendingCase, by_gloss: dict,
                clip_rows: dict[str, dict]) -> dict:
    seq = engine.gloss(case.english)
    if case.english in TOPICALIZE_TARGET and seq.in_scope:
        target = TOPICALIZE_TARGET[case.english]
        seq = engine.topicalize(seq, seq.gloss_ids.index(target))

    clips: list[dict | None] = []
    if seq.in_scope:
        for g in seq.glosses:
            rows = by_gloss.get(g.asllex_id)
            if not rows:
                clips.append(None)
                continue
            row = _pick_clip(rows)
            clips.append({"asllex_id": g.asllex_id, "video_id": row["video_id"],
                           "signer_id": row["signer_id"], "split": row["split"]})
            clip_rows.setdefault(g.asllex_id, row)  # resolve+embed each gloss once

    return {
        "english": case.english,
        "note": case.note,
        "construction": _construction_for(case.english),
        "in_scope": seq.in_scope,
        "reason": seq.reason,
        "sentence_type": seq.sentence_type,
        "negated": seq.negated,
        "confidence": seq.confidence,
        "glosses": [{"text": g.text, "asllex_id": g.asllex_id} for g in seq.glosses],
        "nmm_tags": [{"marker": n.marker, "start": n.start, "end": n.end} for n in seq.nmm_tags],
        "notes": _humanize_trace(seq),
        "clips": clips,
    }


def _b64_video(path: Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_data() -> dict:
    engine = GlossRuleEngine()
    by_gloss = _load_manifest_by_gloss()
    clip_rows: dict[str, dict] = {}

    cases = [build_case(engine, c, by_gloss, clip_rows) for c in PENDING_CASES]

    declared_ids = {c["id"] for c in CONSTRUCTIONS_V2}
    covered_ids = {c["construction"]["id"] for c in cases if c["construction"]}
    missing = declared_ids - covered_ids
    if missing:
        print(f"WARNING: constructions_v2 ids not represented in PENDING_CASES: "
              f"{sorted(missing)} -- add a case or this review sheet is incomplete",
              file=sys.stderr)

    clips_b64: dict[str, str] = {}
    for asllex_id, row in clip_rows.items():
        video_path = REPO / row["video_path"]
        if not video_path.exists():
            print(f"WARNING: missing video for {asllex_id!r}: {video_path}", file=sys.stderr)
            continue
        clips_b64[asllex_id] = _b64_video(video_path)

    return {"cases": cases, "clips": clips_b64}


# ---------------------------------------------------------------------------
# HTML/CSS/JS template. Everything is inline -- no external requests, no
# server, opens as a plain file:// page. The Python side only computes DATA;
# all rendering (gloss/NMM grid, video playback queue, verdict persistence)
# happens client-side in the <script> below, driven by the embedded JSON.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gloss Rule Engine -- Deaf Review Sheet</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --card: #f7f7f9;
    --border: #ddd; --accent: #2a6df4; --good: #1a8a4a; --bad: #b3261e;
    --chip: #eef2ff; --chip-border: #c7d2fe;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --fg: #e8e8e8; --muted: #9aa0a6; --card: #1d2026;
      --border: #333; --accent: #6ea8fe; --good: #4fd07f; --bad: #ff6b6b;
      --chip: #232a44; --chip-border: #3a4a7a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0 0 4rem; background: var(--bg); color: var(--fg);
    font: 15px/1.5 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  header.top {
    position: sticky; top: 0; z-index: 10; background: var(--bg);
    border-bottom: 1px solid var(--border); padding: 1rem 1.5rem;
  }
  header.top h1 { margin: 0 0 .3rem; font-size: 1.25rem; }
  .banner {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: .75rem 1rem; margin: 1rem 1.5rem; font-size: .95rem; color: var(--muted);
  }
  .toolbar {
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-top: .5rem;
  }
  .progress { font-weight: 600; }
  button {
    font: inherit; padding: .5rem 1rem; border-radius: 6px; border: 1px solid var(--border);
    background: var(--accent); color: white; cursor: pointer;
  }
  button.secondary { background: transparent; color: var(--fg); }
  button:disabled { opacity: .4; cursor: not-allowed; }
  main { max-width: 980px; margin: 0 auto; padding: 0 1.5rem; }
  h2.section { margin-top: 2.5rem; border-bottom: 2px solid var(--border); padding-bottom: .4rem; }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.25rem; margin: 1.25rem 0;
  }
  .card.reviewed { border-color: var(--good); }
  .card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
  .english { font-size: 1.15rem; font-weight: 700; }
  .badge {
    display: inline-block; font-size: .75rem; padding: .15rem .5rem; border-radius: 999px;
    background: var(--chip); border: 1px solid var(--chip-border); color: var(--fg);
  }
  .refused-banner {
    background: color-mix(in srgb, var(--bad) 15%, transparent);
    border: 1px solid var(--bad); border-radius: 6px; padding: .6rem .8rem; margin: .6rem 0;
  }
  .construction-box {
    border-left: 3px solid var(--accent); padding: .4rem .8rem; margin: .6rem 0;
    background: color-mix(in srgb, var(--accent) 8%, transparent); font-size: .9rem;
  }
  .gloss-grid { display: grid; gap: .5rem; margin: 1rem 0 .25rem; overflow-x: auto; }
  .gloss-col { display: flex; flex-direction: column; align-items: center; gap: .35rem; min-width: 130px; }
  .gloss-chip {
    font-weight: 700; letter-spacing: .02em; padding: .3rem .6rem; border-radius: 6px;
    background: var(--chip); border: 1px solid var(--chip-border); transition: box-shadow .15s;
  }
  .gloss-chip.playing { box-shadow: 0 0 0 3px var(--accent); }
  video.clip { width: 130px; border-radius: 6px; background: black; }
  .clip-caption { font-size: .7rem; color: var(--muted); text-align: center; }
  .clip-missing {
    width: 130px; height: 90px; display: flex; align-items: center; justify-content: center;
    background: repeating-linear-gradient(45deg, var(--card), var(--card) 8px, var(--border) 8px, var(--border) 9px);
    border: 1px dashed var(--border); border-radius: 6px; font-size: .7rem; color: var(--muted); text-align: center;
  }
  .nmm-row { display: grid; gap: .5rem; margin-bottom: .3rem; }
  .nmm-bar {
    grid-row: 1; align-self: center; height: 22px; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: .7rem; font-weight: 600; color: white;
  }
  .nmm-brow_raise { background: #2a6df4; }
  .nmm-brow_furrow { background: #a24fd0; }
  .nmm-headshake { background: #d0794f; }
  .playbar { display: flex; align-items: center; gap: .75rem; margin: .75rem 0; }
  details.trace { margin: .75rem 0; }
  details.trace summary { cursor: pointer; color: var(--muted); font-size: .9rem; }
  details.trace ul { margin: .5rem 0 0; padding-left: 1.2rem; }
  details.trace li { margin: .2rem 0; font-size: .88rem; }
  .verdict-form { margin-top: 1rem; padding-top: .9rem; border-top: 1px dashed var(--border); }
  .verdict-options { display: flex; gap: 1.1rem; flex-wrap: wrap; margin: .5rem 0; }
  .verdict-options label { display: flex; align-items: center; gap: .35rem; cursor: pointer; }
  textarea.comment {
    width: 100%; min-height: 3.2rem; font: inherit; border-radius: 6px;
    border: 1px solid var(--border); background: var(--bg); color: var(--fg); padding: .5rem;
  }
  .saved-tag { font-size: .78rem; color: var(--good); margin-left: .5rem; visibility: hidden; }
  .saved-tag.show { visibility: visible; }
  footer { max-width: 980px; margin: 2rem auto; padding: 0 1.5rem; color: var(--muted); font-size: .85rem; }
</style>
</head>
<body>

<header class="top">
  <h1>Gloss Rule Engine -- Deaf Review Sheet</h1>
  <div class="toolbar">
    <span class="progress" id="progress">0 / 0 reviewed</span>
    <button id="export-btn" disabled>Export verdicts (JSON)</button>
    <button id="export-md-btn" class="secondary" disabled>Copy summary (Markdown)</button>
    <button id="clear-btn" class="secondary">Clear my answers</button>
  </div>
</header>

<div class="banner">
  <strong>What this page does NOT do:</strong> it does not judge ASL correctness --
  nobody who wrote this code is a fluent signer, so it can't. It only shows,
  faithfully, what the rule engine produced for each sentence: the gloss order,
  where each non-manual marker (brow-raise, brow-furrow, headshake) is claimed to
  apply, and the actual reference clip each sign resolves to. <strong>Your review is
  what turns this from "the engine's guess" into "checked."</strong> For each case:
  watch the clips (individually or "Play in order" for a rough preview -- this is
  NOT a real stitched sentence, just each clip played back to back), read the gloss
  and the engine's own explanation of what it did, and mark a verdict. When you're
  done, click "Export verdicts" and send the downloaded file back.
</div>

<main id="main"></main>

<footer>
  Generated by <code>scripts/export_review_sheet.py</code> from
  <code>tests/test_gloss_rules_corpus.py</code>'s <code>PENDING_CASES</code>. Verdicts
  are saved in this browser only (localStorage) until exported. Exporting produces a
  small JSON file mapping each sentence to your verdict + comment; a developer uses
  that to fill in <code>expected_glosses</code>/<code>expected_nmm</code> and set
  <code>reviewed=True</code> for cases you marked correct.
</footer>

<script id="review-data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById("review-data").textContent);
const STORAGE_KEY = "gloss_review_verdicts_v1";

function loadVerdicts() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
  catch (e) { return {}; }
}
function saveVerdicts(v) { localStorage.setItem(STORAGE_KEY, JSON.stringify(v)); }
let verdicts = loadVerdicts();

function updateProgress() {
  const total = DATA.cases.length;
  const done = DATA.cases.filter(c => verdicts[c.english] && verdicts[c.english].verdict).length;
  document.getElementById("progress").textContent = `${done} / ${total} reviewed`;
  document.getElementById("export-btn").disabled = done === 0;
  document.getElementById("export-md-btn").disabled = done === 0;
}

function glossRow(c) {
  const cols = c.glosses.map((g, i) => {
    const clip = c.clips[i];
    let clipHtml;
    if (clip && DATA.clips[clip.asllex_id]) {
      clipHtml = `<video class="clip" data-idx="${i}" muted playsinline controls
                    src="${DATA.clips[clip.asllex_id]}"></video>
                  <div class="clip-caption">${clip.video_id}<br>${clip.signer_id} / ${clip.split}</div>`;
    } else {
      clipHtml = `<div class="clip-missing">no cached clip</div>`;
    }
    return `<div class="gloss-col" style="grid-column:${i + 1}">
              <div class="gloss-chip" data-chip="${i}">${g.text}</div>
              ${clipHtml}
            </div>`;
  }).join("");
  return `<div class="gloss-grid" style="grid-template-columns:repeat(${c.glosses.length},auto)">${cols}</div>`;
}

function nmmRows(c) {
  if (!c.nmm_tags.length) return "";
  const n = c.glosses.length;
  return c.nmm_tags.map(nm => {
    const bar = `<div class="nmm-bar nmm-${nm.marker}"
                   style="grid-column:${nm.start + 1} / ${nm.end + 1}">${nm.marker}</div>`;
    return `<div class="nmm-row" style="grid-template-columns:repeat(${n},auto)">${bar}</div>`;
  }).join("");
}

function playInOrder(card, videos) {
  let i = 0;
  const chips = card.querySelectorAll(".gloss-chip");
  function playNext() {
    chips.forEach(ch => ch.classList.remove("playing"));
    if (i >= videos.length) return;
    videos[i].currentTime = 0;
    chips[i] && chips[i].classList.add("playing");
    videos[i].play();
    videos[i].onended = () => { i += 1; playNext(); };
  }
  videos.forEach(v => v.pause());
  i = 0;
  playNext();
}

function caseCard(c, idx) {
  const card = document.createElement("div");
  card.className = "card";

  let constructionHtml = "";
  if (c.construction) {
    constructionHtml = `<div class="construction-box">
      <strong>Documented construction: ${c.construction.name}</strong> (${c.construction.id})<br>
      ${c.construction.description}<br>
      Non-manual: ${c.construction.non_manual}<br>
      curriculum.yaml's own example gloss (for comparison, not ground truth):
      <code>${c.construction.example_gloss}</code>
    </div>`;
  }

  let bodyHtml;
  if (!c.in_scope) {
    bodyHtml = `<div class="refused-banner"><strong>REFUSED</strong> -- ${c.reason}</div>`;
  } else {
    bodyHtml = `
      ${glossRow(c)}
      ${nmmRows(c)}
      <div class="playbar">
        <button class="play-btn secondary">&#9654; Play sentence in order</button>
        <span style="color:var(--muted);font-size:.85rem">
          type=${c.sentence_type} negated=${c.negated} confidence=${c.confidence}
        </span>
      </div>`;
  }

  const notesHtml = c.notes.map(n => `<li>${n}</li>`).join("");

  card.innerHTML = `
    <div class="card-head">
      <span class="english">${idx + 1}. "${c.english}"</span>
      ${c.note ? `<span class="badge">${c.note}</span>` : ""}
    </div>
    ${constructionHtml}
    ${bodyHtml}
    <details class="trace">
      <summary>How the engine got here (plain-language trace)</summary>
      <ul>${notesHtml}</ul>
    </details>
    <div class="verdict-form">
      <div class="verdict-options">
        ${["correct", "wrong-order", "wrong-nmm", "other"].map(v => `
          <label>
            <input type="radio" name="verdict-${idx}" value="${v}">
            ${v}
          </label>`).join("")}
      </div>
      <textarea class="comment" placeholder="Comment (what's wrong, or why it's correct)"></textarea>
      <span class="saved-tag">saved</span>
    </div>
  `;

  if (c.in_scope) {
    const videos = Array.from(card.querySelectorAll("video.clip"));
    card.querySelector(".play-btn").addEventListener("click", () => playInOrder(card, videos));
  }

  // restore + persist verdict
  const saved = verdicts[c.english] || {};
  if (saved.verdict) {
    const radio = card.querySelector(`input[name="verdict-${idx}"][value="${saved.verdict}"]`);
    if (radio) radio.checked = true;
    card.classList.add("reviewed");
  }
  const commentBox = card.querySelector("textarea.comment");
  commentBox.value = saved.comment || "";

  function persist() {
    const checked = card.querySelector(`input[name="verdict-${idx}"]:checked`);
    verdicts[c.english] = { verdict: checked ? checked.value : null, comment: commentBox.value };
    saveVerdicts(verdicts);
    card.classList.toggle("reviewed", !!(checked));
    const tag = card.querySelector(".saved-tag");
    tag.classList.add("show");
    clearTimeout(tag._t);
    tag._t = setTimeout(() => tag.classList.remove("show"), 1200);
    updateProgress();
  }
  card.querySelectorAll(`input[name="verdict-${idx}"]`).forEach(r => r.addEventListener("change", persist));
  commentBox.addEventListener("input", persist);

  return card;
}

function render() {
  const main = document.getElementById("main");
  main.innerHTML = "";

  const constructionCases = DATA.cases.filter(c => c.construction);
  const otherCases = DATA.cases.filter(c => !c.construction);

  const h1 = document.createElement("h2");
  h1.className = "section";
  h1.textContent = "Part 1 -- the five documented ASL constructions (curriculum.yaml)";
  main.appendChild(h1);
  constructionCases.forEach((c, i) => main.appendChild(caseCard(c, DATA.cases.indexOf(c))));

  const h2 = document.createElement("h2");
  h2.className = "section";
  h2.textContent = "Part 2 -- additional vocabulary & stacked-construction spot checks";
  main.appendChild(h2);
  otherCases.forEach((c, i) => main.appendChild(caseCard(c, DATA.cases.indexOf(c))));

  updateProgress();
}

document.getElementById("export-btn").addEventListener("click", () => {
  const out = {};
  DATA.cases.forEach(c => { if (verdicts[c.english]) out[c.english] = verdicts[c.english]; });
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gloss_review_verdicts.json";
  a.click();
});

document.getElementById("export-md-btn").addEventListener("click", () => {
  let md = "| English | Verdict | Comment |\n|---|---|---|\n";
  DATA.cases.forEach(c => {
    const v = verdicts[c.english];
    if (!v) return;
    md += `| ${c.english} | ${v.verdict || ""} | ${(v.comment || "").replace(/\n/g, " ")} |\n`;
  });
  navigator.clipboard.writeText(md).then(() => alert("Markdown summary copied to clipboard."));
});

document.getElementById("clear-btn").addEventListener("click", () => {
  if (!confirm("Clear all saved verdicts in this browser?")) return;
  localStorage.removeItem(STORAGE_KEY);
  verdicts = {};
  render();
});

render();
</script>
</body>
</html>
"""


def render_html(data: dict) -> str:
    # JSON can legally contain "</script" inside a string; escape it so the
    # browser's HTML parser doesn't treat it as the end of our <script> tag.
    payload = json.dumps(data).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__DATA_JSON__", payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    print(f"resolving {len(PENDING_CASES)} pending cases + embedding reference clips ...")
    data = build_data()
    html = render_html(data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB, {len(data['clips'])} embedded clips)")
    print(f"open it directly in a browser: file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
