"""How often does a local model's sentence survive the rule engine, and how
fast is it on small hardware? Run in the image, limited to the server's cores:

    docker run --rm --cpus=2 -v aslcv-llm:/app/models aslcv-sentence \\
        python eval_generator.py --per-sign 3 [--repo ... --file ...]

Per generation: accepted (in scope), on target (contains the target sign),
and judged sensible (the judge model, app.JUDGE_MIN).
Per request: the chance that at least one of the server's tries is usable,
from each sign's own on-target rate (signs differ a lot)."""
import argparse
import statistics
import time

from app import JUDGE_MIN, MAX_TRIES
from aslcv.generator.sentence_prompts import _lemma_for, sentence_prompt_maybe_llm
from aslcv.production.gloss_rules import _CURRICULUM_GLOSSES, gloss_sentence
from llm import MODEL_FILE, MODEL_REPO, LocalLLM

ATTEMPTS_PER_CALL = 2  # sentence_prompt_maybe_llm's default max_attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=MODEL_REPO)
    ap.add_argument("--file", default=MODEL_FILE)
    ap.add_argument("--per-sign", type=int, default=3)
    ap.add_argument("--signs", type=int, default=0, help="only the first N signs (0 = all)")
    args = ap.parse_args()

    llm = LocalLLM(args.repo, args.file)
    t0 = time.time()
    llm.write("water", "water")
    print(f"{args.file}: loaded + first reply in {time.time() - t0:.1f}s")
    t0 = time.time()
    llm.judge("I want water.")
    print(f"judge: loaded + first verdict in {time.time() - t0:.1f}s")

    signs = sorted(_CURRICULUM_GLOSSES)[: args.signs or None]
    latencies, rows, request_ok = [], [], []
    for sign in signs:
        hits = 0
        for _ in range(args.per_sign):
            replies = []

            def generate(prompt):
                t = time.time()
                english = llm.write(sign, _lemma_for(sign))
                latencies.append(time.time() - t)
                replies.append(english)
                return english

            sentence_prompt_maybe_llm(sign, generate=generate, max_attempts=1)
            english = replies[0].strip().strip('"') if replies else ""
            seq = gloss_sentence(english) if english else None
            accepted = bool(seq and seq.in_scope)
            on_target = accepted and sign in seq.gloss_ids
            score = llm.judge(english) if on_target else None
            shown = on_target and score >= JUDGE_MIN
            hits += shown
            rows.append((sign, english, accepted, on_target, score, seq.reason if seq and not accepted else ""))
        p = hits / args.per_sign
        request_ok.append(1 - (1 - p) ** (MAX_TRIES * ATTEMPTS_PER_CALL))

    n = len(rows)
    acc = sum(r[2] for r in rows)
    on = sum(r[3] for r in rows)
    shown = sum(r[4] is not None and r[4] >= JUDGE_MIN for r in rows)
    lat = sorted(latencies)
    print(f"{n} generations over {len(signs)} signs")
    print(f"accepted by the engine: {acc}/{n} ({100 * acc / n:.0f}%), on target: {on}/{n} ({100 * on / n:.0f}%), "
          f"judged sensible (>= {JUDGE_MIN}): {shown}/{n} ({100 * shown / n:.0f}%)")
    print(f"latency per generation: median {statistics.median(lat):.1f}s, p90 {lat[int(0.9 * (len(lat) - 1))]:.1f}s")
    print(f"est. on-demand requests that get a sentence within {MAX_TRIES * ATTEMPTS_PER_CALL} generations: "
          f"{100 * statistics.mean(request_ok):.0f}%; signs that never hit: "
          f"{[s for s, ok in zip(signs, request_ok) if ok == 0]}")
    print("\nsamples:")
    for sign, english, accepted, on_target, score, reason in rows:
        mark = ("SHOW" if score >= JUDGE_MIN else "low ") if score is not None else ("off " if accepted else "REF ")
        print(f"  {mark} {'' if score is None else f'{score:6.2f}'} {sign:<10} {english!r}  {reason or ''}")


if __name__ == "__main__":
    main()
