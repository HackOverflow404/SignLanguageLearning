"""Optional LLM-generated sentence prompts -- Phase 6 v2's "sentence prompts",
deferred at v1 (project_workflow.md), now built on top of Phase 5a/5b's already-
finished pieces per that same scoping note.

The LLM writes an English sentence containing the current target sign's word
and NOTHING else ASL-related -- it never sees or produces a gloss, an NMM tag,
or a judgment of correctness. That sentence is then handed to Phase 5b's
ALREADY-BUILT, fail-closed `gloss_rules.gloss_sentence()` exactly as a
human-typed sentence would be. Only if the rule engine accepts it in scope
does it ever become a displayed prompt; an out-of-scope or unknown-vocabulary
sentence is silently discarded and retried, never shown. This is CLAUDE.md's
"an LLM only ever touches English" and "grammar is a rule engine, not a
trained model" enforced TOGETHER: the LLM supplies wording, the deterministic
rule engine is the only thing that decides whether that wording becomes ASL
content -- not a convention, a structural gate.

Presentational only, not graded: this shows a learner an example sentence
using their current target word, glossed and NMM-tagged, to read as an
English -> ASL composition example. Continuous-sentence grading is Phase 7,
unbuilt -- diagnose_demo.py still grades only the isolated target sign, same
as before this module existed.

Opt-in and fail-open, same pattern as llm_feedback.py: no token, no
`huggingface_hub`, a network error, a timeout, or every attempt getting
refused by the rule engine all mean "no sentence prompt this round" (`None`),
never a crash or a hang.
"""
from __future__ import annotations

from ._hf_client import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TIMEOUT, resolve_token
from ..production.gloss_rules import GlossSequence, _curriculum_signs, gloss_sentence

_warned_no_token = False
_vocabulary_cache: "list[str] | None" = None


def _vocabulary() -> list[str]:
    """Every english_lemmas word in curriculum.yaml, deduped -- handed to the
    LLM as an explicit allow-list so it has a real chance of writing something
    the fail-closed rule engine will accept, not so it can be TRUSTED to stay
    inside it (the rule engine is what actually enforces that)."""
    global _vocabulary_cache
    if _vocabulary_cache is None:
        words = {lemma for sign in _curriculum_signs() for lemma in sign.get("english_lemmas", [])}
        _vocabulary_cache = sorted(words)
    return _vocabulary_cache


def _lemma_for(target_gloss: str) -> "str | None":
    """The first english_lemmas entry curriculum.yaml lists for the sign whose
    `gloss` field is `target_gloss`, or None if it isn't a curriculum sign."""
    for sign in _curriculum_signs():
        if sign["gloss"] == target_gloss:
            lemmas = sign.get("english_lemmas") or []
            return lemmas[0] if lemmas else None
    return None


def _prompt(lemma: str, vocabulary: list[str]) -> str:
    return (
        "You are writing ONE short practice sentence for a beginner ASL "
        f"vocabulary app. It must naturally use the word '{lemma}'. Use ONLY "
        "words from this list, plus basic function words (a, an, the, am, "
        f"is, are, to): {', '.join(vocabulary)}. 4-8 words, no contractions, "
        "no idioms. Reply with ONLY the sentence, nothing else."
    )


def sentence_prompt_maybe_llm(target_gloss: str, enabled: bool = True, token: "str | None" = None,
                               model: str = DEFAULT_MODEL, provider: str = DEFAULT_PROVIDER,
                               timeout: float = DEFAULT_TIMEOUT, max_attempts: int = 2,
                               ) -> "GlossSequence | None":
    """An in-scope GlossSequence built from an LLM-written English sentence
    containing `target_gloss`'s English word, or None if disabled, unavailable,
    or every attempt was refused by the gloss rule engine. NEVER returns a
    sequence the fail-closed rule engine itself didn't accept -- callers may
    display the result directly."""
    global _warned_no_token
    if not enabled:
        return None

    lemma = _lemma_for(target_gloss)
    if lemma is None:
        return None

    token = resolve_token(token)
    if not token:
        if not _warned_no_token:
            print("sentence_prompt_maybe_llm: no HF_TOKEN set -- skipping sentence prompts", flush=True)
            _warned_no_token = True
        return None

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("sentence_prompt_maybe_llm: `huggingface_hub` not installed -- skipping", flush=True)
        return None

    vocabulary = _vocabulary()
    for _ in range(max_attempts):
        try:
            client = InferenceClient(model=model, provider=provider, token=token, timeout=timeout)
            response = client.chat_completion(
                messages=[{"role": "user", "content": _prompt(lemma, vocabulary)}],
                max_tokens=40,
            )
            english = response.choices[0].message.content.strip().strip('"')
        except Exception as exc:  # noqa: BLE001 -- any SDK/network failure must fall back, not crash the demo
            print(f"sentence_prompt_maybe_llm: request failed ({exc!r}) -- skipping", flush=True)
            return None
        if not english:
            continue
        seq = gloss_sentence(english)
        if seq.in_scope:
            return seq
        # refused (unknown word / unsupported construction) -- retry rather
        # than ever show a sentence the rule engine didn't accept.
    return None
