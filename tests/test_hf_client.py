"""_hf_client: the repo-local `.env` loader and HF_TOKEN resolution shared by
llm_feedback.py and sentence_prompts.py. No real HF_TOKEN or network needed.

Runs under pytest OR as a plain script (`python tests/test_hf_client.py`).
"""
import os
import sys

from aslcv.generator import _hf_client
from aslcv.generator._hf_client import load_dotenv, resolve_token

from _hf_test_utils import no_ambient_hf_token, temp_repo_root


def test_load_dotenv_sets_unset_env_vars():
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("QUOTED", None)
    try:
        with temp_repo_root('HF_TOKEN=from-dotenv\n# a comment\nQUOTED="value"\n'):
            load_dotenv()
            assert os.environ["HF_TOKEN"] == "from-dotenv"
            assert os.environ["QUOTED"] == "value"
    finally:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("QUOTED", None)


def test_load_dotenv_never_overrides_a_real_env_var():
    os.environ["HF_TOKEN"] = "from-real-env"
    try:
        with temp_repo_root("HF_TOKEN=from-dotenv\n"):
            load_dotenv()
            assert os.environ["HF_TOKEN"] == "from-real-env"
    finally:
        os.environ.pop("HF_TOKEN", None)


def test_load_dotenv_missing_file_is_a_noop():
    with temp_repo_root(None):
        load_dotenv()  # must not raise


def test_resolve_token_prefers_explicit_token_over_dotenv():
    with no_ambient_hf_token():
        with temp_repo_root("HF_TOKEN=from-dotenv\n"):
            assert resolve_token("explicit-token") == "explicit-token"


def test_resolve_token_falls_back_to_dotenv():
    with no_ambient_hf_token():
        with temp_repo_root("HF_TOKEN=from-dotenv\n"):
            assert resolve_token(None) == "from-dotenv"


def test_resolve_token_none_when_nothing_set():
    with no_ambient_hf_token():
        assert resolve_token(None) is None


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
