# build_ablation_data.py — construct ARIA ablation datasets
# Reproduces the exact seed-42 sample from 6000_samples.ipynb, applies per-arm
# trace treatments, and emits train-ready jsonl in the shipped dataset's format.
# Gemini compressions are cached in cache/compressions.jsonl and resumable.
# Run: python build_ablation_data.py --arm s1-uniform          (see ABLATIONS.md)

import os
import json
import time
import random
import hashlib
import argparse

from datasets import load_dataset

# Same system prompt as convert_dataset.py — training format, ASCII pipes
TRAIN_SYSTEM_PROMPT = (
    "You are ARIA, a math reasoning assistant. "
    "Think through problems carefully but concisely inside <think> tags, "
    "then provide a clean final answer."
)
EOS = "<|end▁of▁sentence|>"

# Verbatim from compressed_reasoning.ipynb
COMPRESS_PROMPT = """You are a reasoning trace compressor.

Below is a LONG reasoning trace from a math problem. Your job is to rewrite it as a SHORT,
direct solution that keeps ONLY the essential logical steps needed to reach the answer.

REMOVE:
- Self-doubt ("let me think", "hmm", "wait")
- Redundant verification ("let me check", "let me verify")
- Repeated explanations of the same idea
- Dead ends and backtracking
- Obvious steps that add no value

KEEP:
- The core logical chain from problem to answer
- Key equations and calculations
- Critical insights that lead to the solution

RULES:
- The compressed trace MUST lead to the same final answer
- Target length: {target_words} words maximum
- Write in first person, natural reasoning style
- Do NOT include the final answer — only the thinking steps

ORIGINAL TRACE:
{trace}

Write the compressed reasoning trace:"""

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "cache", "compressions.jsonl")
DATA_DIR = os.path.join(HERE, "data")

# Per-tier targets used for the shipped dataset (paper §3.2)
GRADED_TARGETS = {"easy": 150, "medium": 400, "hard": None}  # None = keep original
UNIFORM_S_TARGET = 300     # token-budget-matched to shipped easy+medium avg (~317 words)
UNIFORM_F_HARD_TARGET = 1300  # 7,640 words / 5.9 — uniform ratio applied to hard

ARMS = [
    "s1-uniform", "s2-shuffled", "s3-uncompressed",
    "f0-graded", "f1-uniform", "f2-shuffled",
]


# ── Sampling (verbatim logic from 6000_samples.ipynb) ─────────────────────────
def load_seed42_sample():
    ds_math = load_dataset("open-r1/OpenThoughts-114k-math")
    all_examples = list(ds_math["train"])
    all_examples.sort(key=lambda x: x["generated_token_count"])
    n = len(all_examples)
    p25 = all_examples[n // 4]["generated_token_count"]       # ~2820
    p75 = all_examples[3 * n // 4]["generated_token_count"]   # ~8648

    easy = [ex for ex in all_examples if ex["generated_token_count"] < p25]
    medium = [ex for ex in all_examples if p25 <= ex["generated_token_count"] < p75]
    hard = [ex for ex in all_examples if ex["generated_token_count"] >= p75]

    random.seed(42)
    easy_sample = random.sample(easy, 2000)
    medium_sample = random.sample(medium, 2000)
    hard_sample = random.sample(hard, 2000)

    parsed = []
    for ex in easy_sample:
        parsed.append(parse_example(ex, "easy"))
    for ex in medium_sample:
        parsed.append(parse_example(ex, "medium"))
    for ex in hard_sample:
        parsed.append(parse_example(ex, "hard"))

    # Drop parse failures (empty thinking) — shipped dataset has 1,993 easy for
    # the same reason
    parsed = [p for p in parsed if p["thinking"].strip()]
    for diff in ["easy", "medium", "hard"]:
        subset = [p for p in parsed if p["difficulty"] == diff]
        avg = sum(p["thinking_word_count"] for p in subset) // len(subset)
        print(f"  {diff}: {len(subset)} examples, avg thinking {avg} words")
    return parsed


def parse_example(ex, difficulty):
    response = ex["conversations"][1]["value"]
    if "<|end_of_thought|>" in response:
        thinking = response.split("<|end_of_thought|>")[0].replace("<|begin_of_thought|>", "").strip()
    else:
        thinking = ""
    if "<|begin_of_solution|>" in response and "<|end_of_solution|>" in response:
        answer = response.split("<|begin_of_solution|>")[1].split("<|end_of_solution|>")[0].strip()
    elif "<|end_of_thought|>" in response:
        answer = response.split("<|end_of_thought|>")[1].strip()
    else:
        answer = response
    return {
        "problem": ex["problem"],
        "thinking": thinking,
        "answer": answer,
        "difficulty": difficulty,
        "thinking_word_count": len(thinking.split()),
    }


# ── Treatment assignment ───────────────────────────────────────────────────────
# A treatment is a target word count (int) or None (keep original trace).
def assign_treatments(parsed, arm):
    easy_medium = [p for p in parsed if p["difficulty"] in ("easy", "medium")]
    rng = random.Random(42)

    if arm == "s1-uniform":
        return [(p, UNIFORM_S_TARGET) for p in easy_medium]

    if arm == "s2-shuffled":
        # Same marginal target distribution as shipped (n_easy×150 + n_medium×400),
        # assignment decoupled from difficulty
        targets = [150 if p["difficulty"] == "easy" else 400 for p in easy_medium]
        rng.shuffle(targets)
        return list(zip(easy_medium, targets))

    if arm == "s3-uncompressed":
        return [(p, None) for p in easy_medium]

    if arm == "f0-graded":
        return [(p, GRADED_TARGETS[p["difficulty"]]) for p in parsed]

    if arm == "f1-uniform":
        targets = dict(GRADED_TARGETS, hard=UNIFORM_F_HARD_TARGET)
        return [(p, targets[p["difficulty"]]) for p in parsed]

    if arm == "f2-shuffled":
        targets = [GRADED_TARGETS[p["difficulty"]] for p in parsed]
        rng.shuffle(targets)
        return list(zip(parsed, targets))

    raise ValueError(f"Unknown arm: {arm}")


# ── Compression cache ──────────────────────────────────────────────────────────
def cache_key(problem, target_words):
    h = hashlib.sha1(problem.encode("utf-8")).hexdigest()
    return f"{h}|{target_words}"

def load_cache():
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            for line in f:
                row = json.loads(line)
                cache[row["key"]] = row["compressed"]
    return cache

def append_cache(key, compressed):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps({"key": key, "compressed": compressed}, ensure_ascii=False) + "\n")


def seed_cache_from_shipped(cache):
    """Pre-load cache with the published easy@150 / medium@400 compressions so
    f0-graded (and matching slots elsewhere) need no new Gemini calls."""
    ds = load_dataset("Eunice-Labs/aria-easy-medium", split="train")
    added = 0
    for row in ds:
        text = row["text"]
        try:
            problem = text.split("<|User|>")[1].split("<|Assistant|>")[0]
            thinking = text.split("<think>\n")[1].split("\n</think>")[0]
        except IndexError:
            continue
        target = 150 if row["difficulty"] == "easy" else 400
        key = cache_key(problem, target)
        if key not in cache:
            append_cache(key, thinking)
            cache[key] = thinking
            added += 1
    print(f"Seeded cache with {added} shipped compressions ({len(cache)} total)")


# ── Gemini compression ─────────────────────────────────────────────────────────
def compress_trace(client, model, trace, target_words, max_retries=3):
    prompt = COMPRESS_PROMPT.format(target_words=target_words, trace=trace)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"    Gemini error ({e}), retrying in {wait}s...")
            time.sleep(wait)
    return None


# ── Dataset assembly ───────────────────────────────────────────────────────────
def to_training_text(problem, thinking, answer):
    return (f"{TRAIN_SYSTEM_PROMPT}<|User|>{problem}<|Assistant|>"
            f"<think>\n{thinking}\n</think>\n\n{answer}{EOS}")


def build_arm(arm, parsed, args):
    pairs = assign_treatments(parsed, arm)
    cache = load_cache()
    to_compress = [(p, t) for p, t in pairs
                   if t is not None and cache_key(p["problem"], t) not in cache]
    print(f"\n[{arm}] {len(pairs)} examples | "
          f"{sum(1 for _, t in pairs if t is not None)} compressed slots | "
          f"{len(to_compress)} new Gemini calls needed")
    if args.stats_only:
        return

    if to_compress:
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY not set — required for compression calls")
        from google import genai
        client = genai.Client()
        delay = 60.0 / args.rpm
        for i, (p, target) in enumerate(to_compress):
            compressed = compress_trace(client, args.gemini_model, p["thinking"], target)
            if compressed is None:
                print(f"    [{i+1}/{len(to_compress)}] FAILED — skipping (re-run to retry)")
                continue
            key = cache_key(p["problem"], target)
            append_cache(key, compressed)
            cache[key] = compressed
            if (i + 1) % 25 == 0 or i == len(to_compress) - 1:
                print(f"    [{i+1}/{len(to_compress)}] compressed "
                      f"({p['thinking_word_count']} → {len(compressed.split())} words)")
            time.sleep(delay)

    rows, dropped = [], 0
    for p, target in pairs:
        if target is None:
            thinking = p["thinking"]
        else:
            thinking = cache.get(cache_key(p["problem"], target))
            if thinking is None:
                dropped += 1
                continue
        rows.append({
            "text": to_training_text(p["problem"], thinking, p["answer"]),
            "difficulty": p["difficulty"],
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{arm}.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} examples → {out_path}"
          + (f" ({dropped} dropped: compression missing)" if dropped else ""))

    # Sanity check: word-count distribution per tier
    for diff in ["easy", "medium", "hard"]:
        tier = [r for r in rows if r["difficulty"] == diff]
        if not tier:
            continue
        words = [len(r["text"].split("<think>\n")[1].split("\n</think>")[0].split())
                 for r in tier]
        print(f"    {diff}: n={len(tier)}, avg think {sum(words)//len(words)} words")

    if args.push:
        ds = load_dataset("json", data_files=out_path, split="train")
        repo = f"Eunice-Labs/aria-ablation-{arm}"
        ds.push_to_hub(repo)
        print(f"  Pushed → {repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=False, default="all",
                    help=f"one of {ARMS} or 'all'")
    ap.add_argument("--stats-only", action="store_true",
                    help="print counts and planned Gemini calls, build nothing")
    ap.add_argument("--seed-cache-from-shipped", action="store_true",
                    help="pre-load cache from Eunice-Labs/aria-easy-medium, then exit")
    ap.add_argument("--gemini-model", default="gemini-2.0-flash")
    ap.add_argument("--rpm", type=float, default=13, help="Gemini requests/minute")
    ap.add_argument("--push", action="store_true", help="push built dataset(s) to HF hub")
    args = ap.parse_args()

    if args.seed_cache_from_shipped:
        seed_cache_from_shipped(load_cache())
        return

    print("Loading + sampling OpenThoughts-114k-math (seed 42)...")
    parsed = load_seed42_sample()

    arms = ARMS if args.arm == "all" else [args.arm]
    for arm in arms:
        build_arm(arm, parsed, args)


if __name__ == "__main__":
    main()
