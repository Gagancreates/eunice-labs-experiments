# custom_eval.py — ARIA paper evaluation
# Measures think token reduction + accuracy on GSM8K and MATH-500
# Run: python custom_eval.py
#
# Think token counting rule (documented in paper):
#   All tokens from first <think> (or start of generation if absent)
#   to LAST </think>. Applied identically to ARIA and base model.

import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from math_verify import verify, parse as math_parse

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Think through problems carefully inside <think> tags, "
    "then provide a clean final answer."
)

MAX_NEW_TOKENS = 4096
DEVICE = "cuda"

# ── Token counting ─────────────────────────────────────────────────────────────
def count_think_tokens(text, tokenizer):
    """
    Count tokens between first <think> (or start of generation)
    and LAST </think>. Returns 0 if no </think> found (truncated).
    """
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]

    last_close = text.rfind("</think>")
    if last_close == -1:
        return 0  # truncated — don't count

    # Find start: first <think> if present, else start of text
    first_open = text.find("<think>")
    if first_open != -1:
        think_content = text[first_open + len("<think>"):last_close]
    else:
        think_content = text[:last_close]

    return len(tokenizer.encode(think_content))


# ── Answer extraction ──────────────────────────────────────────────────────────
def extract_answer(text):
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    # Remove think block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also remove content before last </think>
    last_close = text.rfind("</think>")
    if last_close != -1:
        text = text[last_close + len("</think>"):]
    # Boxed answer
    m = re.search(r"boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    # "the answer is X"
    m = re.search(r"answer\s+is\s+\$?([0-9][\d\s,\.]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    # Last number in text
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
    # Try math_verify first (handles LaTeX, fractions, symbolic equivalence)
    try:
        if verify(math_parse(gold), math_parse(pred)):
            return True
    except Exception:
        pass
    # Fallback: normalized string match
    return normalize(pred) == normalize(gold)


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(model, tokenizer, problem):
    prompt = f"{SYSTEM_PROMPT}<|User|>{problem}<|Assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,         # greedy decoding
            temperature=1.0,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0], skip_special_tokens=False)


# ── Eval loop ──────────────────────────────────────────────────────────────────
def evaluate(model, tokenizer, problems, label):
    """
    problems: list of dicts with keys: problem, answer, level (optional)
    Returns list of result dicts.
    """
    results = []
    for item in tqdm(problems, desc=label):
        decoded = run_inference(model, tokenizer, item["problem"])
        think_tokens = count_think_tokens(decoded, tokenizer)
        pred = extract_answer(decoded)
        correct = is_correct(pred, item["answer"])
        results.append({
            "problem": item["problem"][:100],
            "gold": item["answer"],
            "pred": pred,
            "correct": correct,
            "think_tokens": think_tokens,
            "level": item.get("level", None),
            "raw_output": decoded,
        })
    return results


def summarize(results, by_level=False):
    if not results:
        return {}
    acc = sum(r["correct"] for r in results) / len(results)
    think = [r["think_tokens"] for r in results if r["think_tokens"] > 0]
    summary = {
        "accuracy": round(acc * 100, 1),
        "mean_think_tokens": round(sum(think) / len(think), 1) if think else 0,
        "samples_with_think": len(think),
        "total": len(results),
    }
    if by_level:
        levels = sorted(set(r["level"] for r in results if r["level"]))
        summary["by_level"] = {}
        for lvl in levels:
            lvl_results = [r for r in results if r["level"] == lvl]
            lvl_acc = sum(r["correct"] for r in lvl_results) / len(lvl_results)
            lvl_think = [r["think_tokens"] for r in lvl_results if r["think_tokens"] > 0]
            summary["by_level"][str(lvl)] = {
                "accuracy": round(lvl_acc * 100, 1),
                "mean_think_tokens": round(sum(lvl_think) / len(lvl_think), 1) if lvl_think else 0,
                "n": len(lvl_results),
            }
    return summary


# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading datasets...")
gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
math500_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")

gsm8k = [{"problem": x["question"], "answer": x["answer"].split("####")[-1].strip()} for x in gsm8k_raw][:10]
math500 = [{"problem": x["problem"], "answer": x["answer"], "level": x["level"]} for x in math500_raw][:10]

print(f"GSM8K: {len(gsm8k)} problems")
print(f"MATH-500: {len(math500)} problems")

# ── Load models ────────────────────────────────────────────────────────────────
all_results = {}

for model_name, model_path in [
    ("base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("aria", "./aria-merged"),
]:
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    gsm_results = evaluate(model, tokenizer, gsm8k, f"{model_name}/GSM8K")
    math_results = evaluate(model, tokenizer, math500, f"{model_name}/MATH-500")

    all_results[model_name] = {
        "gsm8k": {"results": gsm_results, "summary": summarize(gsm_results)},
        "math500": {"results": math_results, "summary": summarize(math_results, by_level=True)},
    }

    # Free VRAM before loading next model
    del model
    torch.cuda.empty_cache()

# ── Save results ───────────────────────────────────────────────────────────────
with open("eval_results_final.json", "w") as f:
    # Save without raw outputs to keep file small
    compact = {}
    for model_name, data in all_results.items():
        compact[model_name] = {
            "gsm8k": data["gsm8k"]["summary"],
            "math500": data["math500"]["summary"],
        }
    json.dump(compact, f, indent=2)

# Also save full results (with raw outputs) for debugging
with open("eval_results_full.json", "w") as f:
    for model_name, data in all_results.items():
        for bench in ["gsm8k", "math500"]:
            for r in data[bench]["results"]:
                r.pop("raw_output", None)  # too large
    json.dump(all_results, f, indent=2)

# ── Print paper table ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print("\nGSM8K:")
print(f"{'Model':<10} {'Accuracy':>10} {'Mean Think Tokens':>20}")
print("-" * 45)
for m in ["base", "aria"]:
    s = compact[m]["gsm8k"]
    print(f"{m:<10} {str(s['accuracy'])+'%':>10} {s['mean_think_tokens']:>20}")

print("\nMATH-500 (per level):")
header = f"{'Level':<8} {'Base Acc':>10} {'ARIA Acc':>10} {'Base Tokens':>13} {'ARIA Tokens':>13} {'Reduction':>10}"
print(header)
print("-" * 70)
base_lvl = compact["base"]["math500"].get("by_level", {})
aria_lvl = compact["aria"]["math500"].get("by_level", {})
for lvl in sorted(base_lvl.keys()):
    b = base_lvl[lvl]
    a = aria_lvl.get(lvl, {})
    reduction = round(b["mean_think_tokens"] / a["mean_think_tokens"], 2) if a.get("mean_think_tokens") else "—"
    print(f"{lvl:<8} {str(b['accuracy'])+'%':>10} {str(a.get('accuracy','—'))+'%':>10} "
          f"{b['mean_think_tokens']:>13} {a.get('mean_think_tokens','—'):>13} {str(reduction)+'x':>10}")

print("\nSaved to eval_results_final.json and eval_results_full.json")
