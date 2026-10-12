"""Sentence engine API for the phone app: the LLM writes English, the fail-closed
gloss rule engine decides whether it becomes ASL content.

    POST /sentence {"target": "want_2"}  -> an engine-accepted sentence using the target sign
    POST /gloss    {"english": "..."}     -> the engine's GlossSequence for a typed sentence
    GET  /health

Sentences are written by a small open model running in-process (llm.py), so
the server needs no inference credits or API token. Both return
GlossSequence.to_dict(). /sentence only ever returns a sequence the
rule engine accepted, that contains the target sign, AND that a larger model
judges sensible (llm.LocalLLM.judge >= JUDGE_MIN) -- the engine checks
structure, not meaning. Writing and scoring take ~10-20 s on
a small CPU, so a background thread keeps a small pool of finished sentences
per sign and /sentence serves from it, generating on demand only when a
sign's pool is empty. /gloss returns the
engine's refusal (in_scope false + reason) as an ordinary 200, since a refusal
is the engine's answer, not a server error. The LLM never sees or produces a
gloss -- same structural gate as diagnose_demo.py --sentence-prompts.

The engine code is the repo's own src/aslcv modules, copied unchanged into the
Docker build context by server/build.sh. Runs locally too:

    .venv/bin/uvicorn --app-dir server app:app --port 7860
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from aslcv.generator.sentence_prompts import _lemma_for, sentence_prompt_maybe_llm
from aslcv.production.gloss_rules import _CURRICULUM_GLOSSES, gloss_sentence
from llm import LocalLLM

# The model runs on the server's own CPU, so nothing is billed; the caps keep
# one client from monopolising a server that writes one sentence at a time.
PER_CLIENT_LIMIT = int(os.environ.get("PER_CLIENT_LIMIT", 30))   # /sentence calls per window
PER_CLIENT_WINDOW = 600.0                                       # seconds
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", 1000))         # /sentence calls per UTC day
MAX_TRIES = 3          # candidates per on-demand request, each with its own engine retries
MAX_ENGLISH_CHARS = 200
JUDGE_MIN = float(os.environ.get("JUDGE_MIN", 2.5))  # yes-over-no log-odds; see llm.py's docstring
POOL_SIZE = int(os.environ.get("POOL_SIZE", 3))           # ready sentences kept per sign
POOL_FILL = os.environ.get("POOL_FILL", "1") == "1"       # background filling (off in tests)


@asynccontextmanager
async def _lifespan(app):
    if POOL_FILL:
        threading.Thread(target=_fill_forever, daemon=True).start()
    yield


app = FastAPI(title="ASL tutor sentence engine", lifespan=_lifespan)
_lock = threading.Lock()
_pool: dict[str, deque] = defaultdict(deque)
_last_try: dict[str, float] = {}  # sign -> when the filler last tried it
_serving = threading.Event()  # an on-demand request is generating; the filler waits
_recent: dict[str, deque] = defaultdict(deque)
_day = [time.strftime("%Y-%m-%d", time.gmtime()), 0]
_llm = LocalLLM()


class SentenceRequest(BaseModel):
    target: str = Field(max_length=64)


class GlossRequest(BaseModel):
    english: str = Field(max_length=MAX_ENGLISH_CHARS)


def _client(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "?")


def _admit(client: str) -> None:
    now = time.time()
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    with _lock:
        if _day[0] != today:
            _day[:] = [today, 0]
        if _day[1] >= DAILY_LIMIT:
            raise HTTPException(429, "daily sentence limit reached; try again tomorrow")
        q = _recent[client]
        while q and now - q[0] > PER_CLIENT_WINDOW:
            q.popleft()
        if len(q) >= PER_CLIENT_LIMIT:
            raise HTTPException(429, "too many sentences requested; wait a few minutes")
        q.append(now)
        _day[1] += 1


def _log(kind: str, payload: dict) -> None:
    """One JSON line per engine output, so generated sentences can be pulled
    from the server logs (`docker compose logs app`) into the Deaf review sheet."""
    print(json.dumps({"kind": kind, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}),
          flush=True)


def _writer(target: str):
    """prompt -> English for sentence_prompt_maybe_llm. The local model gets
    its own short prompt and a target-forcing grammar (llm.py) instead of the
    hosted prompt's vocabulary list; the engine gate afterwards is unchanged."""
    lemma = _lemma_for(target)
    return lambda _prompt: _llm.write(target, lemma)


def _candidate(target: str) -> "dict | None":
    """One finished sentence for `target`, or None: engine-accepted (with its
    own retries), on target, and judged sensible."""
    seq = sentence_prompt_maybe_llm(target, generate=_writer(target))
    if seq is None:
        return None
    if target not in seq.gloss_ids:
        _log("off_target", {"target": target, **seq.to_dict()})
        return None
    score = _llm.judge(seq.english)
    result = {"target": target, **seq.to_dict(), "sense": round(score, 3)}
    if score < JUDGE_MIN:
        _log("nonsense", result)
        return None
    return result


def _next_to_fill() -> str:
    """The emptiest pool's sign, least recently tried first: a sign whose
    candidates keep failing gets its turn in rotation, never all of them."""
    return min(sorted(_CURRICULUM_GLOSSES), key=lambda s: (len(_pool[s]), _last_try.get(s, 0.0)))


def _fill_forever() -> None:
    """Top up the emptiest sign's pool, one sentence at a time, yielding to
    on-demand requests; idles once every pool is full."""
    while True:
        if _serving.is_set():
            time.sleep(0.5)
            continue
        target = _next_to_fill()
        _last_try[target] = time.monotonic()
        if len(_pool[target]) >= POOL_SIZE:
            time.sleep(30)
            continue
        try:
            result = _candidate(target)
        except Exception as exc:  # noqa: BLE001 -- the filler must never die
            print(f"pool filler: {exc!r}", flush=True)
            time.sleep(5)
            continue
        if result is not None:
            _pool[target].append(result)
            _log("pooled", result)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "signs": len(_CURRICULUM_GLOSSES),
            "pooled": sum(len(q) for q in _pool.values())}


@app.post("/sentence")
def sentence(body: SentenceRequest, request: Request) -> dict:
    if body.target not in _CURRICULUM_GLOSSES:
        raise HTTPException(404, f"{body.target!r} is not a curriculum sign")
    _admit(_client(request))
    try:
        result = _pool[body.target].popleft()
        _log("served_pooled", result)
        return result
    except IndexError:
        pass
    _serving.set()
    try:
        for _ in range(MAX_TRIES):
            result = _candidate(body.target)
            if result is not None:
                _log("sentence", result)
                return result
    finally:
        _serving.clear()
    raise HTTPException(503, "no accepted sentence this time (every try was refused, off target, or judged nonsense)")


@app.post("/gloss")
def gloss(body: GlossRequest) -> dict:
    english = body.english.strip()
    if not english:
        raise HTTPException(422, "empty sentence")
    result = gloss_sentence(english).to_dict()
    _log("gloss", result)
    return result
