"""Optional LLM pass over `feedback.coach_text`'s diagnosis -- phrasing only,
via HuggingFace's HOSTED Inference API (huggingface_hub.InferenceClient), not a
locally-run model.

That's a deliberate choice, not a default: this project's eventual target is a
phone app, and a phone can't run even a "small" open LLM the way a desktop
Python process can (no PyTorch/transformers runtime in a shipped mobile app;
on-device LLM inference needs a mobile-native runtime -- GGUF+llama.cpp,
ExecuTorch, CoreML -- and its own model conversion work, none of which this
does). A phone client calling a hosted API for a short phrased sentence is
exactly how a real mobile app would do this, so building against the hosted
API now means this code is close to what ships, not a local-only prototype
that gets thrown away later. See project_workflow.md's Phase 8 for the full
mobile-porting picture this sits inside.

CLAUDE.md's rule: "An LLM only ever touches English... it never authors or
judges ASL." Enforced structurally here, not just by convention: `llm_coach_text`
hands the model a fixed, pre-computed set of facts (target sign, which
parameter(s) were wrong, confidence) and instructs it to phrase ONLY those
facts more naturally -- it is never asked "was this correct," never shown raw
attempt data, and has no path to add ASL knowledge of its own. The verdict
itself always comes from `EmbeddingGrader.grade_against`, never from this
module.

Opt-in and fail-open: this is a phrasing UPGRADE over `coach_text`'s templated
output, not a replacement it depends on. No token, no `huggingface_hub`
package, a network error, a timeout, or any malformed response all fall back
to `coach_text` silently (one warning printed once) rather than block or crash
a live session -- the live diagnostic loop must never hang on a network call.
"""
from __future__ import annotations

from ._hf_client import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TIMEOUT, resolve_token
from .feedback import PARAM_NAME, coach_text, readable_value

_warned_no_token = False


def _facts(target_sign: str, parameters) -> dict:
    """`wrong` carries not just WHICH parameters missed but the two
    pre-computed, grounded values behind that verdict: `you_signed` (the
    grader's own classification of the attempt) and `should_be` (the target
    sign's true ASL-LEX/curriculum label) -- both already computed by
    `EmbeddingGrader.grade_against`, never invented here or by the LLM this
    feeds into. Both go through `readable_value` first (formatting only --
    `closed_b` -> `Closed B` -- never a new claim), so the LLM is never asked
    to phrase a raw jargon code like the learner would otherwise see."""
    verdicts = list(parameters.values()) if hasattr(parameters, "values") else list(parameters)
    judged = [v for v in verdicts if v.correct is not None]
    return {
        "target_sign": target_sign,
        "all_correct": bool(judged) and all(v.correct for v in judged),
        "wrong": [
            {"name": PARAM_NAME[v.parameter],
             "you_signed": readable_value(v.parameter, v.predicted),
             "should_be": readable_value(v.parameter, v.target)}
            for v in judged if v.correct is False
        ],
        "correct": [PARAM_NAME[v.parameter] for v in judged if v.correct is True],
    }


def _prompt(facts: dict) -> str:
    if not facts["wrong"] and not facts["correct"]:
        body = "Nothing could be confidently diagnosed this attempt."
    elif facts["all_correct"]:
        body = f"Every diagnosed aspect matched the reference sign for '{facts['target_sign']}'."
    else:
        wrong_lines = "; ".join(
            f"{w['name']}: they signed '{w['you_signed']}', it should be '{w['should_be']}'"
            for w in facts["wrong"]
        )
        body = (
            f"Target sign: '{facts['target_sign']}'. "
            f"Matched: {', '.join(facts['correct']) or 'none'}. "
            f"Did NOT match -- {wrong_lines}."
        )
    return (
        "You are phrasing ASL practice feedback for a learner, in one short, "
        "SPECIFIC sentence (max ~35 words), plain English, no ASL jargon they "
        "wouldn't already know from the app. For anything that did NOT "
        "match, clearly state what they signed and what the target value "
        "actually is, using the exact values given -- do not soften this "
        "into vague praise like 'keep practicing.' You are NOT an ASL expert "
        "and must not add any ASL knowledge, corrections, or claims beyond "
        "the facts given below -- only phrase them naturally.\n\n"
        f"Facts: {body}\n\n"
        "Reply with ONLY the sentence, nothing else."
    )


def llm_coach_text(target_sign: str, parameters, token: "str | None" = None,
                    model: str = DEFAULT_MODEL, provider: str = DEFAULT_PROVIDER,
                    timeout: float = DEFAULT_TIMEOUT) -> "str | None":
    """The LLM-phrased coaching line, or None on ANY failure -- callers must
    treat None as "fall back to coach_text(parameters)", never as an error to
    surface to the learner."""
    global _warned_no_token
    token = resolve_token(token)
    if not token:
        if not _warned_no_token:
            print("llm_coach_text: no HF_TOKEN set -- falling back to templated feedback", flush=True)
            _warned_no_token = True
        return None

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("llm_coach_text: `huggingface_hub` not installed -- falling back to templated feedback", flush=True)
        return None

    try:
        client = InferenceClient(model=model, provider=provider, token=token, timeout=timeout)
        response = client.chat_completion(
            messages=[{"role": "user", "content": _prompt(_facts(target_sign, parameters))}],
            max_tokens=150,  # up from 100: naming multiple wrong parameters' you-signed/should-be
                              # values (not just their names) needs more room before truncating
        )
        text = response.choices[0].message.content.strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 -- any SDK/network failure must fall back, not crash the demo
        print(f"llm_coach_text: request failed ({exc!r}) -- falling back to templated feedback", flush=True)
        return None


def coach_text_maybe_llm(target_sign: str, parameters, use_llm: bool, **kwargs) -> str:
    """The single entry point callers should use: LLM phrasing if `use_llm`
    and it succeeds, else the templated fallback -- always returns something
    displayable, never None."""
    if use_llm:
        text = llm_coach_text(target_sign, parameters, **kwargs)
        if text is not None:
            return text
    return coach_text(parameters)
