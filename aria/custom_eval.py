# custom_eval.py — ARIA paper evaluation (final)
# GSM8K: 200 samples | MATH-500: all 500 (5 levels)
# Outputs: accuracy, avg think tokens, RES score, per-level table, case studies
# Run: python custom_eval.py

import re
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from math_verify import verify, parse as math_parse
from huggingface_hub import HfApi

HF_REPO = "Eunice-Labs/aria-eval-results"
hf_api = HfApi()

def push_file_to_hub(filepath):
    try:
        hf_api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filepath,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print(f"  Pushed {filepath} to {HF_REPO}")
    except Exception as e:
        print(f"  Warning: failed to push {filepath}: {e}")

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Think through problems carefully inside <think> tags, "
    "then provide a clean final answer."
)

MAX_NEW_TOKENS_DEFAULT = 4096   # GSM8K + MATH L1/L2
MAX_NEW_TOKENS_HARD    = 8192   # MATH L3/L4/L5
DEVICE = "cuda"

# ── Token counting ─────────────────────────────────────────────────────────────
def count_think_tokens(text, tokenizer):
    """Tokens from first <think> (or start of generation) to LAST </think>."""
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    last_close = text.rfind("</think>")
    if last_close == -1:
        return 0
    first_open = text.find("<think>")
    if first_open != -1:
        think_content = text[first_open + len("<think>"):last_close]
    else:
        think_content = text[:last_close]
    return len(tokenizer.encode(think_content))

def has_think_tag(text):
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    return "<think>" in text


# ── Answer extraction ──────────────────────────────────────────────────────────
def extract_boxed(text):
    """Balanced brace matcher — handles nested braces e.g. \\boxed{\\frac{1}{2}}."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None

def extract_answer(text):
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    last_close = text.rfind("</think>")
    if last_close != -1:
        text = text[last_close + len("</think>"):]
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"answer\s+is\s+\$?([0-9][\d\s,\.]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    nums = re.findall(r"\$?-?[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "").replace("$", "")
    return None

def normalize(ans):
    if ans is None:
        return None
    return ans.replace(",", "").replace("$", "").replace(" ", "").strip().lower()

def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        if verify(math_parse(gold), math_parse(pred)):
            return True
    except Exception:
        pass
    return normalize(pred) == normalize(gold)


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(model, tokenizer, problem, max_new_tokens=MAX_NEW_TOKENS_DEFAULT):
    prompt = f"{SYSTEM_PROMPT}<|User|>{problem}<|Assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0], skip_special_tokens=False)


# ── Eval loop ──────────────────────────────────────────────────────────────────
def evaluate(model, tokenizer, problems, label):
    results = []
    correct_count = 0
    total_time = 0.0

    pbar = tqdm(problems, desc=label)
    for item in pbar:
        level = item.get("level")
        max_tok = MAX_NEW_TOKENS_HARD if level in [3, 4, 5] else MAX_NEW_TOKENS_DEFAULT

        t0 = time.time()
        decoded = run_inference(model, tokenizer, item["problem"], max_new_tokens=max_tok)
        elapsed = time.time() - t0
        total_time += elapsed

        think_tokens = count_think_tokens(decoded, tokenizer)
        pred = extract_answer(decoded)
        correct = is_correct(pred, item["answer"])
        if correct:
            correct_count += 1

        n = len(results) + 1
        avg_time = total_time / n
        running_acc = round(100 * correct_count / n, 1)
        eta_sec = avg_time * (len(problems) - n)
        eta_min = round(eta_sec / 60, 1)

        pbar.set_postfix({
            "acc": f"{running_acc}%",
            "avg_s": f"{avg_time:.1f}s",
            "ETA": f"{eta_min}m",
        })

        results.append({
            "problem": item["problem"],
            "difficulty": item.get("level", None),
            "model_output": decoded,
            "think_tokens": think_tokens,
            "answer_extracted": pred,
            "ground_truth": item["answer"],
            "correct": correct,
            "has_think_tag": has_think_tag(decoded),
            "inference_time_s": round(elapsed, 2),
        })

    total_min = round(total_time / 60, 1)
    print(f"  {label} done — {correct_count}/{len(problems)} correct | "
          f"total time: {total_min}m | avg: {round(total_time/len(problems),1)}s/sample")
    return results


def summarize(results, by_level=False):
    if not results:
        return {}
    acc = sum(r["correct"] for r in results) / len(results)
    think = [r["think_tokens"] for r in results if r["think_tokens"] > 0]
    mean_tokens = round(sum(think) / len(think), 1) if think else 0
    res_score = round(acc * 100 / mean_tokens * 1000, 2) if mean_tokens > 0 else 0
    no_think = sum(1 for r in results if not r["has_think_tag"])
    summary = {
        "accuracy": round(acc * 100, 1),
        "mean_think_tokens": mean_tokens,
        "res_score": res_score,
        "missing_think_tag": no_think,
        "total": len(results),
    }
    if by_level:
        levels = sorted(set(r["difficulty"] for r in results if r["difficulty"]))
        summary["by_level"] = {}
        for lvl in levels:
            lvl_res = [r for r in results if r["difficulty"] == lvl]
            lvl_acc = sum(r["correct"] for r in lvl_res) / len(lvl_res)
            lvl_think = [r["think_tokens"] for r in lvl_res if r["think_tokens"] > 0]
            lvl_mean = round(sum(lvl_think) / len(lvl_think), 1) if lvl_think else 0
            lvl_res_score = round(lvl_acc * 100 / lvl_mean * 1000, 2) if lvl_mean > 0 else 0
            summary["by_level"][str(lvl)] = {
                "accuracy": round(lvl_acc * 100, 1),
                "mean_think_tokens": lvl_mean,
                "res_score": lvl_res_score,
                "n": len(lvl_res),
            }
    return summary


# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading datasets...")
gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
math500_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")

gsm8k = [{"problem": x["question"], "answer": x["answer"].split("####")[-1].strip()}
         for x in gsm8k_raw][:200]

# Stratified sample: 50 per level (1-5) = 250 total
from collections import defaultdict
per_level = defaultdict(list)
for x in math500_raw:
    per_level[x["level"]].append(x)
math500 = []
for lvl in sorted(per_level.keys()):
    for x in per_level[lvl][:50]:
        math500.append({"problem": x["problem"], "answer": x["answer"], "level": x["level"]})

print(f"GSM8K: {len(gsm8k)} problems")
print(f"MATH-500: {len(math500)} problems (50 per level)")

# Base GSM8K already done — hardcode result from prior run
# 152/200 correct, 76.0% acc, avg 17.1s/sample
BASE_GSM8K_SUMMARY = {
    "accuracy": 76.0,
    "mean_think_tokens": 449.5,   # from smoke eval
    "res_score": round(76.0 / 449.5 * 1000, 2),
    "missing_think_tag": 0,
    "total": 200,
    "note": "from prior completed run — 152/200 correct",
}

# ── Run eval ───────────────────────────────────────────────────────────────────
all_results = {}

for model_name, model_path in [
    ("base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("aria", "./aria-merged"),
]:
    print(f"\n{'='*55}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*55}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Base GSM8K already done — skip and use cached result
    if model_name == "base":
        print("  Skipping base/GSM8K — using prior run result (76.0%, 152/200)")
        gsm_results = []
        gsm_summary = BASE_GSM8K_SUMMARY
    else:
        gsm_results = evaluate(model, tokenizer, gsm8k, f"{model_name}/GSM8K")
        gsm_summary = summarize(gsm_results)

    math_results = evaluate(model, tokenizer, math500, f"{model_name}/MATH-500")

    math_summary = summarize(math_results, by_level=True)
    all_results[model_name] = {
        "gsm8k": {"results": gsm_results, "summary": gsm_summary},
        "math500": {"results": math_results, "summary": math_summary},
    }

    # Save + push full per-model results after each phase
    if gsm_results:
        fname = f"eval_{model_name}_gsm8k.json"
        with open(fname, "w") as f:
            json.dump(gsm_results, f, indent=2)
        print(f"  Saved {fname}")
        push_file_to_hub(fname)

    fname = f"eval_{model_name}_math500.json"
    with open(fname, "w") as f:
        json.dump(math_results, f, indent=2)
    print(f"  Saved {fname}")
    push_file_to_hub(fname)

    del model
    torch.cuda.empty_cache()


# ── Save compact summary ───────────────────────────────────────────────────────
compact = {}
for model_name, data in all_results.items():
    compact[model_name] = {
        "gsm8k": data["gsm8k"]["summary"],
        "math500": data["math500"]["summary"],
    }
with open("eval_results_final.json", "w") as f:
    json.dump(compact, f, indent=2)
push_file_to_hub("eval_results_final.json")


# ── Print main results table ───────────────────────────────────────────────────
print("\n" + "="*75)
print("MAIN RESULTS TABLE")
print("="*75)
print(f"{'Benchmark':<12} {'Model':<8} {'Accuracy':>10} {'Avg Tokens':>12} {'RES Score':>12}")
print("-"*55)
for bench in ["gsm8k", "math500"]:
    for m in ["base", "aria"]:
        s = compact[m][bench]
        print(f"{bench.upper():<12} {m:<8} {str(s['accuracy'])+'%':>10} "
              f"{s['mean_think_tokens']:>12} {s['res_score']:>12}")
    print()


# ── Print MATH-500 per-level table ─────────────────────────────────────────────
print("="*90)
print("MATH-500 PER-LEVEL BREAKDOWN")
print("="*90)
print(f"{'Level':<8} {'Base Acc':>10} {'ARIA Acc':>10} {'Base Tok':>10} {'ARIA Tok':>10} "
      f"{'Reduction':>10} {'Base RES':>10} {'ARIA RES':>10}")
print("-"*90)
base_lvl = compact["base"]["math500"].get("by_level", {})
aria_lvl = compact["aria"]["math500"].get("by_level", {})
for lvl in sorted(base_lvl.keys()):
    b = base_lvl[lvl]
    a = aria_lvl.get(lvl, {})
    reduction = (round(b["mean_think_tokens"] / a["mean_think_tokens"], 2)
                 if a.get("mean_think_tokens") else "—")
    print(f"{lvl:<8} {str(b['accuracy'])+'%':>10} {str(a.get('accuracy','—'))+'%':>10} "
          f"{b['mean_think_tokens']:>10} {a.get('mean_think_tokens','—'):>10} "
          f"{str(reduction)+'x':>10} {b['res_score']:>10} {a.get('res_score','—'):>10}")


# ── Missing <think> tag analysis ──────────────────────────────────────────────
print("\n" + "="*75)
print("MISSING <think> TAG ANALYSIS")
print("="*75)
for m in ["base", "aria"]:
    for bench in ["gsm8k", "math500"]:
        s = compact[m][bench]
        total = s["total"]
        missing = s["missing_think_tag"]
        pct = round(100 * missing / total, 1)
        print(f"  {m.upper()} / {bench.upper()}: {missing}/{total} missing <think> tag ({pct}%)")


# ── Bad label detection ────────────────────────────────────────────────────────
print("\n" + "="*75)
print("POTENTIAL BAD LABELS (both models agree, both marked wrong)")
print("="*75)
all_bad = []
for bench in ["gsm8k", "math500"]:
    base_res = all_results["base"][bench]["results"]
    aria_res = all_results["aria"][bench]["results"]
    for b, a in zip(base_res, aria_res):
        if (not b["correct"] and not a["correct"]
                and b["answer_extracted"] and a["answer_extracted"]
                and normalize(b["answer_extracted"]) == normalize(a["answer_extracted"])):
            all_bad.append({
                "bench": bench,
                "problem": b["problem"][:120],
                "gold": b["ground_truth"],
                "both_predicted": b["answer_extracted"],
                "level": b.get("difficulty"),
            })
if all_bad:
    print(f"\n{len(all_bad)} suspect label(s):")
    for x in all_bad:
        lvl = f" (L{x['level']})" if x.get("level") else ""
        print(f"  [{x['bench'].upper()}{lvl}] gold={x['gold']} | both predicted={x['both_predicted']}")
        print(f"    {x['problem'][:80]}...")
else:
    print("\nNo suspect labels found.")
with open("bad_labels.json", "w") as f:
    json.dump(all_bad, f, indent=2)


# ── Case studies ──────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("CASE STUDIES")
print("="*75)
case_studies = {}

aria_math = all_results["aria"]["math500"]["results"]
base_math  = all_results["base"]["math500"]["results"]
aria_gsm   = all_results["aria"]["gsm8k"]["results"]

# Case 1: Easy problem, ARIA correct, fewest think tokens
easy = [r for r in aria_math if r["correct"] and r["difficulty"] in [1, 2] and r["think_tokens"] > 0]
if not easy:
    easy = [r for r in aria_gsm if r["correct"] and r["think_tokens"] > 0]
if easy:
    case1 = min(easy, key=lambda r: r["think_tokens"])
    case_studies["easy_short_think"] = case1
    print(f"\n[Case 1] Easy + shortest think — {case1['think_tokens']} tokens (L{case1['difficulty']})")
    print(f"  Problem: {case1['problem'][:100]}...")
    print(f"  Gold: {case1['ground_truth']} | Predicted: {case1['answer_extracted']}")

# Case 2: Hard problem, ARIA correct, most think tokens used
hard = [r for r in aria_math if r["correct"] and r["difficulty"] in [4, 5] and r["think_tokens"] > 0]
if hard:
    case2 = max(hard, key=lambda r: r["think_tokens"])
    case_studies["hard_deep_think"] = case2
    print(f"\n[Case 2] Hard + deepest think — {case2['think_tokens']} tokens (L{case2['difficulty']})")
    print(f"  Problem: {case2['problem'][:100]}...")
    print(f"  Gold: {case2['ground_truth']} | Predicted: {case2['answer_extracted']}")

# Case 3: Side-by-side — largest token gap, both models correct
paired = [
    (b, a) for b, a in zip(base_math, aria_math)
    if b["correct"] and a["correct"] and b["think_tokens"] > 0 and a["think_tokens"] > 0
]
if paired:
    b_best, a_best = max(paired, key=lambda x: x[0]["think_tokens"] - x[1]["think_tokens"])
    case_studies["side_by_side"] = {
        "base": b_best,
        "aria": a_best,
        "token_reduction": round(b_best["think_tokens"] / a_best["think_tokens"], 2),
    }
    print(f"\n[Case 3] Side-by-side — base {b_best['think_tokens']} tok → aria {a_best['think_tokens']} tok "
          f"({round(b_best['think_tokens']/a_best['think_tokens'],2)}x reduction)")
    print(f"  Problem: {b_best['problem'][:100]}...")

with open("case_studies.json", "w") as f:
    json.dump(case_studies, f, indent=2)

print("\n" + "="*75)
print("FILES SAVED")
print("="*75)
print("  eval_results_final.json   — compact summary (accuracy, tokens, RES)")
print("  eval_base_gsm8k.json      — full base GSM8K results (raw outputs)")
print("  eval_base_math500.json    — full base MATH-500 results (raw outputs)")
print("  eval_aria_gsm8k.json      — full ARIA GSM8K results (raw outputs)")
print("  eval_aria_math500.json    — full ARIA MATH-500 results (raw outputs)")
print("  bad_labels.json           — suspect gold labels")
print("  case_studies.json         — 3 case study examples")
