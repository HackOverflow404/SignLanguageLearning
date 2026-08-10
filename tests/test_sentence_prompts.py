"""sentence_prompts: fail-open behavior, the LLM's English is never trusted
without the fail-closed gloss rule engine accepting it, and a real target's
lemma resolves against the real curriculum.yaml -- no real HF_TOKEN, network,
or huggingface_hub package needed for any of this.

Runs under pytest OR as a plain script (`python tests/test_sentence_prompts.py`).
"""
import sys
import types

from aslcv.generator.sentence_prompts import (
    _lemma_for, _prompt, _vocabulary, sentence_prompt_maybe_llm,
)

from _hf_test_utils import no_ambient_hf_token


# ---- pure lookups against the real curriculum ------------------------------------

def test_lemma_for_known_sign():
    assert _lemma_for("water") == "water"


def test_lemma_for_unknown_sign_is_none():
    assert _lemma_for("not_a_real_sign") is None


def test_vocabulary_includes_known_lemmas_and_has_no_duplicates():
    vocab = _vocabulary()
    assert "water" in vocab
    assert len(vocab) == len(set(vocab))


def test_prompt_names_the_lemma_and_constrains_vocabulary():
    prompt = _prompt("water", ["water", "want_2"])
    assert "'water'" in prompt
    assert "want_2" in prompt


# ---- fail-open behavior, no real token/network needed ----------------------------

def test_disabled_never_touches_the_llm_path():
    with no_ambient_hf_token():
        assert sentence_prompt_maybe_llm("water", enabled=False, token="would-be-used-if-enabled") is None


def test_unknown_target_sign_returns_none_without_calling_llm():
    with no_ambient_hf_token():
        assert sentence_prompt_maybe_llm("not_a_real_sign", token="fake-token-for-test") is None


def test_no_hf_token_returns_none():
    with no_ambient_hf_token():
        assert sentence_prompt_maybe_llm("water", token=None) is None


# ---- mocked SDK -------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeChatCompletionResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeInferenceClient:
    def __init__(self, replies, **kw):
        # sentence_prompt_maybe_llm constructs a NEW InferenceClient on every
        # retry attempt -- `replies` must be the SAME shared list across
        # those constructions (not copied) so popping actually advances.
        self._replies = replies

    def chat_completion(self, **kwargs):
        return _FakeChatCompletionResponse(self._replies.pop(0))


def _install_fake_hub_module(client_factory):
    fake_module = types.ModuleType("huggingface_hub")
    fake_module.InferenceClient = client_factory
    sys.modules["huggingface_hub"] = fake_module


def test_in_scope_reply_is_accepted():
    _install_fake_hub_module(lambda **kw: _FakeInferenceClient(["I want water."]))
    try:
        seq = sentence_prompt_maybe_llm("water", token="fake-token-for-test")
        assert seq is not None
        assert seq.in_scope
        assert "water" in seq.gloss_ids
    finally:
        sys.modules.pop("huggingface_hub", None)


def test_out_of_scope_reply_retries_then_gives_up():
    # both attempts use vocabulary the rule engine will refuse (unknown word).
    # `replies` must be the SAME list object across the two InferenceClient
    # constructions sentence_prompt_maybe_llm's retry loop makes -- a list
    # literal inside the lambda body would be recreated (full again) on every
    # call instead of being popped down.
    replies = ["I love pizza.", "The book that I read is good."]
    _install_fake_hub_module(lambda **kw: _FakeInferenceClient(replies))
    try:
        assert sentence_prompt_maybe_llm("water", token="fake-token-for-test", max_attempts=2) is None
    finally:
        sys.modules.pop("huggingface_hub", None)


def test_second_attempt_succeeds_after_first_is_refused():
    replies = ["I love pizza.", "I want water."]
    _install_fake_hub_module(lambda **kw: _FakeInferenceClient(replies))
    try:
        seq = sentence_prompt_maybe_llm("water", token="fake-token-for-test", max_attempts=2)
        assert seq is not None and seq.in_scope
    finally:
        sys.modules.pop("huggingface_hub", None)


def test_client_exception_falls_back_to_none_not_raise():
    def _raising_client(**kw):
        raise RuntimeError("simulated network failure")
    _install_fake_hub_module(_raising_client)
    try:
        assert sentence_prompt_maybe_llm("water", token="fake-token-for-test") is None
    finally:
        sys.modules.pop("huggingface_hub", None)


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
