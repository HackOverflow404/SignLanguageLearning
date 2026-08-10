"""Shared test-only helpers for isolating HF token/`.env` state across
test_llm_feedback.py, test_hf_client.py, and test_sentence_prompts.py. NOT a
test file itself (no test_ functions), so pytest never collects it directly.
"""
import contextlib
import os
import tempfile
from pathlib import Path

from aslcv.generator import _hf_client


@contextlib.contextmanager
def no_ambient_hf_token():
    """Guarantees resolve_token() sees NO token: pops HF_TOKEN/
    HUGGINGFACE_HUB_TOKEN from the real environment and points
    _hf_client.REPO_ROOT at an empty temp dir, so neither a developer's shell
    env nor a real repo-root `.env` (this project's own recommended place to
    put HF_TOKEN) can leak into a "no token" test and trigger a real network
    call."""
    saved = {k: os.environ.pop(k, None) for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")}
    original_root = _hf_client.REPO_ROOT
    with tempfile.TemporaryDirectory() as d:
        _hf_client.REPO_ROOT = Path(d)
        try:
            yield
        finally:
            _hf_client.REPO_ROOT = original_root
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value


@contextlib.contextmanager
def temp_repo_root(env_file_contents=None):
    """Points _hf_client.REPO_ROOT at a fresh temp dir, optionally containing
    a `.env` with the given contents, for testing load_dotenv()/resolve_token()
    directly without touching the real repo root."""
    original_root = _hf_client.REPO_ROOT
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        if env_file_contents is not None:
            (root / ".env").write_text(env_file_contents)
        _hf_client.REPO_ROOT = root
        try:
            yield root
        finally:
            _hf_client.REPO_ROOT = original_root
