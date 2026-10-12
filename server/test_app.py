"""Server tests -- no network: the LLM step is replaced with the real rule
engine glossing fixed sentences. Run inside the server image
(`docker run --rm aslcv-sentence pytest -q`) or with the repo venv
(`.venv/bin/python -m pytest server/test_app.py`)."""
import os

os.environ["POOL_FILL"] = "0"  # no background generation in tests

import pytest
from fastapi.testclient import TestClient

import app as server
from aslcv.production.gloss_rules import gloss_sentence


class FakeScorer:
    """Stands in for the judge model: `unnatural` sentences score low."""

    def __init__(self, unnatural=()):
        self.unnatural = set(unnatural)

    def judge(self, english):
        return -3.0 if english in self.unnatural else 4.0


@pytest.fixture
def client(monkeypatch):
    server._recent.clear()
    server._pool.clear()
    server._last_try.clear()
    server._day[1] = 0
    monkeypatch.setattr(server, "_llm", FakeScorer())
    return TestClient(server.app)


def _llm_writes(monkeypatch, *sentences):
    """Each sentence_prompt_maybe_llm call glosses the next sentence, as the
    real function would after the LLM replied with it (None = no reply)."""
    replies = iter(sentences)

    def fake(target, **kw):
        english = next(replies)
        if english is None:
            return None
        seq = gloss_sentence(english)
        return seq if seq.in_scope else None

    monkeypatch.setattr(server, "sentence_prompt_maybe_llm", fake)


def test_health(client):
    assert client.get("/health").json() == {"ok": True, "signs": 60, "pooled": 0}


def test_gloss_accepts_and_orders(client):
    r = client.post("/gloss", json={"english": "I want water."}).json()
    assert r["in_scope"] is True
    assert [g["asllex_id"] for g in r["glosses"]] == ["me", "want_2", "water"]


def test_gloss_refusal_is_an_answer_not_an_error(client):
    r = client.post("/gloss", json={"english": "The xylophone sings."})
    assert r.status_code == 200
    assert r.json()["in_scope"] is False and r.json()["reason"]
    assert r.json()["glosses"] == []


def test_gloss_rejects_empty_and_overlong(client):
    assert client.post("/gloss", json={"english": "  "}).status_code == 422
    assert client.post("/gloss", json={"english": "a" * 500}).status_code == 422


def test_sentence_unknown_target(client):
    assert client.post("/sentence", json={"target": "xylophone"}).status_code == 404


def test_sentence_skips_off_target_and_refused(client, monkeypatch):
    _llm_writes(monkeypatch, None, "I want coffee.", "I want water.")
    r = client.post("/sentence", json={"target": "water"})
    assert r.status_code == 200
    assert r.json()["target"] == "water"
    assert "water" in [g["asllex_id"] for g in r.json()["glosses"]]


def test_sentence_gives_up_with_503(client, monkeypatch):
    _llm_writes(monkeypatch, None, None, None)
    assert client.post("/sentence", json={"target": "water"}).status_code == 503


def test_sentence_rate_limited_per_client(client, monkeypatch):
    monkeypatch.setattr(server, "PER_CLIENT_LIMIT", 2)
    _llm_writes(monkeypatch, *["I want water."] * 5)
    codes = [client.post("/sentence", json={"target": "water"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_every_sign_has_a_valid_grammar():
    from llama_cpp import LlamaGrammar

    from llm import grammar_text
    for sign in sorted(server._CURRICULUM_GLOSSES):
        LlamaGrammar.from_string(grammar_text(sign), verbose=False)
    assert '"thank you"' in grammar_text("thank_you")
    assert '"went"' in grammar_text("go")


def test_unnatural_sentences_are_never_served(client, monkeypatch):
    monkeypatch.setattr(server, "_llm", FakeScorer(unnatural={"I want water."}))
    _llm_writes(monkeypatch, "I want water.", "I want water.", "I want water.")
    assert client.post("/sentence", json={"target": "water"}).status_code == 503


def test_pooled_sentence_is_served_first(client, monkeypatch):
    _llm_writes(monkeypatch, "I want water.", "I drink water.")
    server._pool["water"].append(server._candidate("water"))
    r = client.post("/sentence", json={"target": "water"}).json()
    assert r["english"] == "I want water." and r["sense"] == 4.0
    assert client.post("/sentence", json={"target": "water"}).json()["english"] == "I drink water."


def test_filler_rotates_past_a_sign_that_keeps_failing(client):
    first = server._next_to_fill()
    server._last_try[first] = 1.0
    second = server._next_to_fill()
    assert second != first
    server._last_try[second] = 2.0
    assert server._next_to_fill() not in (first, second)
    server._pool[first].append({})
    assert server._next_to_fill() != first


def test_real_models_write_and_judge():
    """The baked-in models end to end (~20 s): a grammar-forced sentence
    contains the target's word, and the judge ranks sense above nonsense."""
    from llm import LocalLLM
    llm = LocalLLM()
    assert "water" in llm.write("water", "water").lower()
    assert llm.judge("I want to drink water.") > llm.judge("The water is my school father.")
