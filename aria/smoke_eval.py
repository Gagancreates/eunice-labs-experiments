# smoke_eval.py — quick directional eval before full benchmark
# 20 GSM8K + 4 per MATH-500 level (1-5) = 40 samples x 2 models
# Run: python smoke_eval.py

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

print("Loading datasets...")
gsm8k = load_dataset("openai/gsm8k", "main", split="test").select(range(20))
math500_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
math500 = []
per_level = defaultdict(list)
for x in math500_raw:
    per_level[x["level"]].append(x)
for lvl in sorted(per_level.keys()):
    math500.extend(per_level[lvl][:4])
print(f"GSM8K: {len(gsm8k)} samples | MATH-500: {len(math500)} samples (4 per level)")

all_results = {}

for model_name, model_path in [
    ("base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("aria", "./aria-merged"),
]:
    print(f"\n=== {model_name} ===")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # GSM8K
    gsm_tokens = []
    for i, item in enumerate(gsm8k):
        print(f"  GSM8K {i+1}/20...", end="\r")
        prompt = f"{SYSTEM}<|User|>{item['question']}<|Assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, do_sample=False, repetition_penalty=1.1)
        gsm_tokens.append(count_think_tokens(tokenizer.decode(out[0], skip_special_tokens=False), tokenizer))
    gsm_mean = round(sum(gsm_tokens) / len(gsm_tokens), 1)
    print(f"  GSM8K mean think tokens: {gsm_mean}")

    # MATH-500 per level
    level_tokens = defaultdict(list)
    for i, item in enumerate(math500):
        print(f"  MATH-500 {i+1}/20...", end="\r")
        prompt = f"{SYSTEM}<|User|>{item['problem']}<|Assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, do_sample=False, repetition_penalty=1.1)
        level_tokens[item["level"]].append(count_think_tokens(tokenizer.decode(out[0], skip_special_tokens=False), tokenizer))

    math_by_level = {}
    for lvl in sorted(level_tokens.keys()):
        t = level_tokens[lvl]
        mean = round(sum(t) / len(t), 1)
        math_by_level[str(lvl)] = {"mean_think_tokens": mean, "n": len(t)}
        print(f"  MATH-500 Level {lvl}: {mean} tokens")

    all_results[model_name] = {
        "gsm8k": {"mean_think_tokens": gsm_mean, "n": len(gsm_tokens), "all": gsm_tokens},
        "math500_by_level": math_by_level,
    }

    del model
    torch.cuda.empty_cache()

# ── Save results ───────────────────────────────────────────────────────────────
with open("smoke_eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to smoke_eval_results.json")

# ── Print comparison table ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("SMOKE EVAL — TOKEN REDUCTION SUMMARY")
print("="*55)
print(f"\n{'Benchmark':<20} {'Base':>10} {'ARIA':>10} {'Reduction':>12}")
print("-"*55)

base_gsm = all_results["base"]["gsm8k"]["mean_think_tokens"]
aria_gsm = all_results["aria"]["gsm8k"]["mean_think_tokens"]
red = round(base_gsm / aria_gsm, 2) if aria_gsm > 0 else "—"
print(f"{'GSM8K':<20} {base_gsm:>10} {aria_gsm:>10} {str(red)+'x':>12}")

for lvl in sorted(all_results["base"]["math500_by_level"].keys()):
    b = all_results["base"]["math500_by_level"][lvl]["mean_think_tokens"]
    a = all_results["aria"]["math500_by_level"].get(lvl, {}).get("mean_think_tokens", 0)
    red = round(b / a, 2) if a > 0 else "—"
    print(f"{'MATH-500 L'+lvl:<20} {b:>10} {a:>10} {str(red)+'x':>12}")
