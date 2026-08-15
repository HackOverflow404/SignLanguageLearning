"""Shared bootstrap for generator modules that call HuggingFace's HOSTED
Inference API (`llm_feedback.py`, `sentence_prompts.py`) -- both need the
identical repo-local `.env` + HF_TOKEN/HUGGINGFACE_HUB_TOKEN resolution, so it
lives in one place rather than risking the two copies silently drifting apart.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root: src/aslcv/generator/_hf_client.py -> generator -> aslcv -> src -> root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Llama-3.1-8B-Instruct: same size class as the alternatives (8B -- small
# enough to be cheap/fast for a short phrasing call, large enough for solid
# instruction-following), picked over Qwen2.5-7B-Instruct (used until this
# session) specifically for provider REDUNDANCY under provider="auto": HF's
# own model API (GET /api/models/<id>?expand[]=inferenceProviderMapping,
# checked directly, not assumed) showed Qwen2.5-7B-Instruct served by only 2
# providers (together, featherless-ai) vs. Llama-3.1-8B-Instruct's 4 (novita,
# nscale, deepinfra, featherless-ai) -- and Qwen was observed failing live
# with "not supported by any provider you have enabled" then succeeding
# minutes later with no code change, exactly the flakiness thin provider
# coverage predicts. More providers behind "auto" means one provider being
# down/rate-limited is far less likely to take the whole call out.
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PROVIDER = "auto"
DEFAULT_TIMEOUT = 6.0


def load_dotenv() -> None:
    """Best-effort load of a repo-root `.env` (KEY=VALUE per line, gitignored
    -- see `.env.example`) into os.environ. No `python-dotenv` dependency,
    deliberately: this project's `uv` tracking is already fragile (CLAUDE.md
    known issue #13) to not add one just for this. Real env vars always win --
    setdefault only fills in what isn't already set."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_token(token: "str | None" = None) -> "str | None":
    """`token` if given, else HF_TOKEN, else HUGGINGFACE_HUB_TOKEN -- loading
    the repo-local `.env` first so either env var can come from there too."""
    load_dotenv()
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
