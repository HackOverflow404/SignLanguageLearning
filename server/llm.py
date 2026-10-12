"""The sentence writer and its sense check: small open instruct models
run in-process with llama.cpp on the server's CPU -- no inference
credits, no API token.

Decoding is constrained by a grammar built from the gloss engine's own
lexicon: every word is a curriculum word (or an inflection of one) or a
function word the engine drops, the target sign's word must appear, and the
sentence is one short line ending in "." or "?". The grammar is deliberately
looser than the engine -- it only stops the model wasting a try on words the
engine can't know. Whether a sentence becomes ASL content is still decided
solely by gloss_rules, after generation, exactly as for any other writer.

A grammar-forced small model also writes fluent nonsense the engine can't
catch ("My father is a school."), since the engine checks structure, not
meaning. A second, larger model judges each sentence: the log-odds of it
answering "yes" over "no" to "does this make sense as something a person
would really say?"; app.py shows only sentences at or above JUDGE_MIN.
The cutoff was picked on 121 hand-labelled outputs (AUC 0.95; mean token
log-probability only 0.85), then checked on 304 new ones from the 0.5B, 1.5B
and 3B writers, hand-labelled 152 sensible / 152 not: AUC 0.96, and at 2.5 it
shows 120, of which 115 are sensible (precision 0.96, recall 0.76). The 3B
judges worse (AUC 0.94): it never passes nonsense but rejects many sensible
questions outright."""
from __future__ import annotations

import os
import random
import threading
from functools import lru_cache
from pathlib import Path

from aslcv.production.gloss_rules import _PHRASES, _build_lexicon

# Writer: the 3B model. Writing 3 sentences for each of the 60 signs, the
# share that is on target AND judged sensible was 22% (0.5B), 35% (1.5B) and
# 54% (3B); 62% for the 3B once the grammar allowed "at" and 7 words a side.
MODEL_REPO = os.environ.get("LLM_REPO", "Qwen/Qwen2.5-3B-Instruct-GGUF")
MODEL_FILE = os.environ.get("LLM_FILE", "qwen2.5-3b-instruct-q4_k_m.gguf")
# Judge: the 1.5B model.
SCORER_REPO = os.environ.get("SCORER_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
SCORER_FILE = os.environ.get("SCORER_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
JUDGE_PROMPT = ("<|im_start|>system\nYou check practice sentences for a beginner English class.<|im_end|>\n"
                "<|im_start|>user\nDoes this sentence make sense as something a person would really say? "
                "\"{sentence}\" Answer yes or no.<|im_end|>\n<|im_start|>assistant\n")
MODEL_DIR = Path(os.environ.get("LLM_DIR", Path(__file__).resolve().parent / "models"))
MAX_TOKENS = 24
MAX_SIDE_WORDS = 7  # words allowed before and after the target; 5 cut sentences off at "to."
TEMPERATURE = 0.8  # retries need a different sentence, not the same refused one again
# Layers offloaded to a GPU, for evaluating on a CUDA build of llama.cpp
# (-1 = all). The server's VM has no GPU, so 0 there.
GPU_LAYERS = int(os.environ.get("LLM_GPU_LAYERS", 0))

# Words the engine drops (articles, copula, auxiliary "do", "to", "at") or
# reads as structure ("not" -> negation, "and" -> noun/adjective coordination).
# "at" is here because without it a model writing "at school" was forced into
# "ate school"; other prepositions stay out, since the engine drops them too
# and "on"/"with" carry meaning (spatial relations are signed in space).
FUNCTION_WORDS = ["a", "an", "the", "am", "is", "are", "was", "were", "be", "do", "does", "did",
                  "to", "at", "not", "and"]
# Lexicon words that never inflect.
_UNINFLECTED = {"i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her",
                "we", "us", "our", "ours", "what", "where", "who", "whom", "why", "how", "hello",
                "hi", "no", "yes", "yeah", "yep", "ok", "okay", "please", "now", "today",
                "tomorrow", "thanks", "thank_you"}
_IRREGULAR = {"go": ["went", "gone"], "eat": ["ate", "eaten"], "drink": ["drank", "drunk"],
              "come": ["came"], "sleep": ["slept"], "learn": ["learnt"]}

SYSTEM = ("You write one very short, simple English sentence for a beginner sign language class. "
          "One clause, 3 to 7 words, everyday words only, and it must make sense.")
# A few are shown per request, never one containing the target word: with fixed
# examples the model copied them ("My friend is happy today." for 6 targets).
EXAMPLES = ["I want water.", "Where is my mother?", "My friend is happy today.", "Do you want coffee?",
            "I am not tired.", "He is reading a book.", "We go to school tomorrow.", "What is your name?",
            "My sister is hungry now.", "The dog is sleeping.", "Please come home.", "I love my family."]


def _forms(word: str) -> set[str]:
    """English surface forms of a lexicon lemma: plural/3sg, -ing, -ed, plus
    a few irregular verbs. Over-generating is harmless (the engine judges)."""
    if word in _UNINFLECTED:
        return {word}
    forms = {word, word + "s", word + "ing", word + "ed", *_IRREGULAR.get(word, [])}
    if word.endswith(("s", "sh", "ch", "x", "o")):
        forms.add(word + "es")
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        forms |= {word[:-1] + "ies", word[:-1] + "ied"}
    if word.endswith("e"):
        forms |= {word[:-1] + "ing", word + "d"}
    return forms


def _alternatives(words) -> str:
    """A GBNF alternation of the words, each lowercase or Capitalized."""
    variants = sorted({v for w in words for v in (w, w[:1].upper() + w[1:])})
    return " | ".join(f'"{v}"' for v in variants)


@lru_cache(maxsize=None)
def grammar_text(target_gloss: str) -> str:
    lexicon = _build_lexicon()
    phrase_words = {combined: list(phrase) for phrase, combined in _PHRASES.items()}
    words = set(FUNCTION_WORDS)
    for lemma in lexicon:
        words |= set(phrase_words.get(lemma, [])) if lemma in phrase_words else _forms(lemma)
    targets = set()
    for lemma, gloss in lexicon.items():
        if gloss == target_gloss:
            targets |= {" ".join(phrase_words[lemma])} if lemma in phrase_words else _forms(lemma)
    if not targets:
        raise KeyError(f"{target_gloss!r} has no English word in the lexicon")
    return "\n".join([
        f'root ::= (w " "){{0,{MAX_SIDE_WORDS}}} t (" " w){{0,{MAX_SIDE_WORDS}}} end',
        f"w ::= {_alternatives(words)}",
        f"t ::= {_alternatives(targets)}",
        'end ::= "." | "?"',
    ])


def fetch(repo: str = MODEL_REPO, file: str = MODEL_FILE) -> Path:
    """The GGUF weights, downloaded once (public model, no token)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, file, local_dir=MODEL_DIR))


def fetch_all() -> None:
    fetch(MODEL_REPO, MODEL_FILE)
    fetch(SCORER_REPO, SCORER_FILE)


class LocalLLM:
    """Loaded on first use; one generation at a time (a llama.cpp context is
    not thread-safe, and two CPUs gain nothing from running two at once)."""

    def __init__(self, repo: str = MODEL_REPO, file: str = MODEL_FILE,
                 scorer_repo: str = SCORER_REPO, scorer_file: str = SCORER_FILE):
        self.repo, self.file = repo, file
        self.scorer_repo, self.scorer_file = scorer_repo, scorer_file
        self._llm = None
        self._scorer = None
        self._grammars = {}
        self._lock = threading.Lock()

    def _load(self):
        from llama_cpp import Llama
        # A random seed per process: the default is fixed, so every restart
        # would replay the same sentences in the same order.
        self._llm = Llama(model_path=str(fetch(self.repo, self.file)), n_ctx=512,
                          n_threads=os.cpu_count() or 2, n_gpu_layers=GPU_LAYERS, seed=random.randrange(2**31), verbose=False)

    @staticmethod
    def _examples(lemma: str, k: int = 3) -> list[str]:
        pool = [e for e in EXAMPLES if lemma.lower() not in e.lower().rstrip(".?").split()]
        return random.sample(pool, k)

    def write(self, target_gloss: str, lemma: str) -> str:
        """One English sentence using `target_gloss`'s word `lemma` (named in
        the prompt; the grammar accepts any of the sign's English forms)."""
        from llama_cpp import LlamaGrammar
        with self._lock:
            if self._llm is None:
                self._load()
            if target_gloss not in self._grammars:
                self._grammars[target_gloss] = LlamaGrammar.from_string(grammar_text(target_gloss), verbose=False)
            out = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": f"{SYSTEM} Examples: {' '.join(self._examples(lemma))}"},
                          {"role": "user", "content": f"Write one sentence using the word '{lemma}'."}],
                grammar=self._grammars[target_gloss], max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        return out["choices"][0]["message"]["content"]

    def judge(self, sentence: str) -> float:
        """Log-odds that the judge model answers yes (makes sense) rather than
        no, from its next-token distribution. Higher is more sensible."""
        import numpy as np
        from llama_cpp import Llama
        with self._lock:
            if self._scorer is None:
                self._scorer = Llama(model_path=str(fetch(self.scorer_repo, self.scorer_file)), n_ctx=512,
                                     n_threads=os.cpu_count() or 2, n_gpu_layers=GPU_LAYERS, logits_all=True, verbose=False)
            m = self._scorer
            first = lambda w: m.tokenize(w.encode(), add_bos=False, special=False)[0]
            yes = sorted({first(w) for w in ("yes", "Yes", " yes", " Yes")})
            no = sorted({first(w) for w in ("no", "No", " no", " No")})
            tokens = m.tokenize(JUDGE_PROMPT.format(sentence=sentence).encode(), special=True)
            m.reset()
            m.eval(tokens)
            logits = np.array(m.scores[len(tokens) - 1], dtype=np.float64)
        top = logits.max()
        log_sum = lambda ids: np.log(np.exp(logits[ids] - top).sum())
        return float(log_sum(yes) - log_sum(no))
