# accuracy_eval.py — token reduction + accuracy on 10 samples each
# GSM8K: 10 samples | MATH-500: 2 per level (levels 1-5) = 10 samples
# Run: python accuracy_eval.py

import torch, re, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict

SYSTEM = (
    "You are a math reasoning assistant. "
    "Think through problems carefully inside <think> tags, "
    "then provide a clean final answer."
)

def count_think_tokens(text, tokenizer):
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    last_close = text.rfind("</think>")
    if last_close == -1:
        return 0
    first_open = text.find("<think>")
    content = text[first_open+7:last_close] if first_open != -1 else text[:last_close]
    return len(tokenizer.encode(content))

def extract_answer(text):
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    last_close = text.rfind("</think>")
    if last_close != -1:
        text = text[last_close + len("</think>"):]
    m = re.search(r"boxed\{([^}]+)\}", text)
    if m: return m.group(1).strip()
    m = re.search(r"answer\s+is\s+\$?([0-9][\d\s,\.]*)", text, re.IGNORECASE)
    if m: return m.group(1).strip().rstrip(".")
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if nums: return nums[-1].replace(",", "")
    return None

def normalize(ans):
    if ans is None: return None
    return ans.replace(",", "").replace("$", "").replace(" ", "").strip().lower()

def is_correct(pred, gold):
    return normalize(pred) == normalize(gold) if pred and gold else False

# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading datasets...")
gsm8k = load_dataset("openai/gsm8k", "main", split="test").select(range(10))
gsm8k_problems = [{"problem": x["question"], "answer": x["answer"].split("####")[-1].strip()} for x in gsm8k]

math500_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
per_level = defaultdict(list)
for x in math500_raw:
    per_level[x["level"]].append(x)
math500_problems = []
for lvl in sorted(per_level.keys()):
    for x in per_level[lvl][:2]:
        math500_problems.append({"problem": x["problem"], "answer": x["answer"], "level": x["level"]})

print(f"GSM8K: {len(gsm8k_problems)} | MATH-500: {len(math500_problems)} (2 per level)")

# ── Eval ───────────────────────────────────────────────────────────────────────
all_results = {}

for model_name, model_path in [
    ("base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("aria", "./aria-merged"),
]:
    print(f"\n=== {model_name} ===")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    def run(problems, tag):
        results = []
        for i, item in enumerate(problems):
            print(f"  {tag} {i+1}/{len(problems)}...", end="\r")
            prompt = f"{SYSTEM}<|User|>{item['problem']}<|Assistant|>"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=2048, do_sample=False, repetition_penalty=1.1)
            decoded = tokenizer.decode(out[0], skip_special_tokens=False)
            think_tokens = count_think_tokens(decoded, tokenizer)
            pred = extract_answer(decoded)
            correct = is_correct(pred, item["answer"])
            results.append({"think_tokens": think_tokens, "correct": correct, "pred": pred, "gold": item["answer"], "level": item.get("level")})
        return results

    gsm_res = run(gsm8k_problems, "GSM8K")
    math_res = run(math500_problems, "MATH-500")

    gsm_acc = round(100 * sum(r["correct"] for r in gsm_res) / len(gsm_res), 1)
    gsm_tokens = round(sum(r["think_tokens"] for r in gsm_res) / len(gsm_res), 1)
    print(f"  GSM8K     — acc: {gsm_acc}% | mean think tokens: {gsm_tokens}")

    by_level = defaultdict(list)
    for r in math_res:
        by_level[r["level"]].append(r)
    math_summary = {}
    for lvl in sorted(by_level.keys()):
        lvl_res = by_level[lvl]
        acc = round(100 * sum(r["correct"] for r in lvl_res) / len(lvl_res), 1)
        tokens = round(sum(r["think_tokens"] for r in lvl_res) / len(lvl_res), 1)
        math_summary[str(lvl)] = {"acc": acc, "mean_think_tokens": tokens}
        print(f"  MATH L{lvl}   — acc: {acc}% | mean think tokens: {tokens}")

    all_results[model_name] = {"gsm8k": {"acc": gsm_acc, "mean_think_tokens": gsm_tokens}, "math500": math_summary}
    del model
    torch.cuda.empty_cache()

# ── Save ───────────────────────────────────────────────────────────────────────
with open("accuracy_eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ── Print table ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ACCURACY + TOKEN REDUCTION")
print("="*70)
print(f"{'Benchmark':<15} {'Base Acc':>9} {'ARIA Acc':>9} {'Base Tok':>9} {'ARIA Tok':>9} {'Reduction':>10}")
print("-"*70)

def row(label, b_acc, a_acc, b_tok, a_tok):
    red = round(b_tok / a_tok, 2) if a_tok > 0 else 0
    print(f"{label:<15} {str(b_acc)+'%':>9} {str(a_acc)+'%':>9} {b_tok:>9} {a_tok:>9} {str(red)+'x':>10}")

b, a = all_results["base"], all_results["aria"]
row("GSM8K", b["gsm8k"]["acc"], a["gsm8k"]["acc"], b["gsm8k"]["mean_think_tokens"], a["gsm8k"]["mean_think_tokens"])
for lvl in sorted(b["math500"].keys()):
    row(f"MATH L{lvl}", b["math500"][lvl]["acc"], a["math500"][lvl]["acc"], b["math500"][lvl]["mean_think_tokens"], a["math500"][lvl]["mean_think_tokens"])

print("\nSaved to accuracy_eval_results.json")
