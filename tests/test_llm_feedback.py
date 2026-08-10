"""llm_feedback: fail-open behavior (no token / bad import / request error all
fall back to coach_text), the fact-extraction stays pure-data (no ASL judgment
of its own), and a successful call is used when huggingface_hub is mocked --
no real HF_TOKEN or network access needed for any of this.

Runs under pytest OR as a plain script (`python tests/test_llm_feedback.py`).
"""
import sys
import types
from types import SimpleNamespace

from aslcv.generator.feedback import coach_text
from aslcv.generator.llm_feedback import _facts, _prompt, coach_text_maybe_llm, llm_coach_text

from _hf_test_utils import no_ambient_hf_token


def verdict(parameter, correct, confidence=0.8, predicted="A", target="B"):
    return SimpleNamespace(parameter=parameter, correct=correct, confidence=confidence,
                            predicted=predicted, target=target)


# ---- pure fact-extraction / prompt shape ---------------------------------------

def test_facts_all_correct():
    parameters = {"handshape": verdict("handshape", True)}
    facts = _facts("mother", parameters)
    assert facts["all_correct"] is True
    assert facts["wrong"] == []


def test_facts_records_wrong_with_readable_name_and_grounded_values():
    parameters = {
        "handshape": verdict("handshape", False, predicted="1", target="5"),
        "movement": verdict("movement", True),
    }
    facts = _facts("mother", parameters)
    assert facts["wrong"] == [{"name": "handshape", "you_signed": "1", "should_be": "5"}]
    assert facts["correct"] == ["movement path"] or facts["correct"] == ["movement"]  # PARAM_NAME value


def test_facts_skips_unjudged_none_verdicts():
    parameters = {"handshape": verdict("handshape", None)}
    facts = _facts("mother", parameters)
    assert facts["wrong"] == [] and facts["correct"] == []
    assert facts["all_correct"] is False


def test_prompt_never_asks_the_model_to_judge_correctness():
    facts = _facts("mother", {"handshape": verdict("handshape", False)})
    prompt = _prompt(facts)
    assert "phrase" in prompt.lower()
    assert "not an asl expert" in prompt.lower() or "not add any asl knowledge" in prompt.lower()


def test_prompt_carries_the_exact_signed_and_target_values():
    facts = _facts("mother", {"handshape": verdict("handshape", False, predicted="1", target="5")})
    prompt = _prompt(facts)
    assert "'1'" in prompt
    assert "'5'" in prompt


# ---- fail-open behavior, no real token/network needed ---------------------------
# Wrapped in no_ambient_hf_token(): these must hold even on a machine that has
# a real HF_TOKEN in its shell or in this project's own recommended repo-root
# `.env` (see .env.example) -- token=None means "pretend nothing is
# configured," not "whatever this machine happens to have lying around."

def test_no_hf_token_returns_none():
    parameters = {"handshape": verdict("handshape", False)}
    with no_ambient_hf_token():
        assert llm_coach_text("mother", parameters, token=None) is None


def test_coach_text_maybe_llm_false_never_touches_llm_path():
    parameters = {"handshape": verdict("handshape", False, confidence=0.9)}
    # use_llm=False must produce exactly the templated text, with no token at all
    with no_ambient_hf_token():
        assert coach_text_maybe_llm("mother", parameters, use_llm=False) == coach_text(parameters)


def test_coach_text_maybe_llm_falls_back_when_llm_unavailable():
    parameters = {"handshape": verdict("handshape", False, confidence=0.9)}
    # use_llm=True but no token available -- must still return the templated
    # text, never None, never raise
    with no_ambient_hf_token():
        result = coach_text_maybe_llm("mother", parameters, use_llm=True, token=None)
    assert result == coach_text(parameters)


# ---- mocked SDK: a successful call is actually used -----------------------------

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
    def __init__(self, reply="Great try! Focus on your handshape next time.", **kw):
        self._reply = reply

    def chat_completion(self, **kwargs):
        return _FakeChatCompletionResponse(self._reply)


def _install_fake_hub_module(client_factory):
    fake_module = types.ModuleType("huggingface_hub")
    fake_module.InferenceClient = client_factory
    sys.modules["huggingface_hub"] = fake_module


def test_successful_call_is_used_verbatim():
    _install_fake_hub_module(lambda **kw: _FakeInferenceClient())
    try:
        parameters = {"handshape": verdict("handshape", False)}
        result = llm_coach_text("mother", parameters, token="fake-token-for-test")
        assert result == "Great try! Focus on your handshape next time."
    finally:
        sys.modules.pop("huggingface_hub", None)


def test_client_exception_falls_back_to_none_not_raise():
    def _raising_client(**kw):
        raise RuntimeError("simulated network failure")
    _install_fake_hub_module(_raising_client)
    try:
        parameters = {"handshape": verdict("handshape", False)}
        assert llm_coach_text("mother", parameters, token="fake-token-for-test") is None
    finally:
        sys.modules.pop("huggingface_hub", None)


def test_coach_text_maybe_llm_uses_successful_llm_text():
    _install_fake_hub_module(lambda **kw: _FakeInferenceClient(reply="Nice work overall!"))
    try:
        parameters = {"handshape": verdict("handshape", True)}
        result = coach_text_maybe_llm("mother", parameters, use_llm=True, token="fake-token-for-test")
        assert result == "Nice work overall!"
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
