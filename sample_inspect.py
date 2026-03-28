# sample_inspect.py — print full model output for 1 GSM8K + 1 MATH L4 sample
# Run: python sample_inspect.py

import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Think through problems carefully inside <think> tags, "
    "then provide a clean final answer."
)
MAX_NEW_TOKENS = 4096

# ── Pick samples ──────────────────────────────────────────────────────────────
print("Loading datasets...")
gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
math500_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")

random.seed(42)
gsm_sample = random.choice(list(gsm8k_raw))
math_l4 = [x for x in math500_raw if x["level"] == 4]
math_sample = random.choice(math_l4)

print("\n" + "="*70)
print("GSM8K SAMPLE")
print("="*70)
print(f"Problem: {gsm_sample['question']}")
print(f"Gold answer: {gsm_sample['answer'].split('####')[-1].strip()}")

print("\n" + "="*70)
print("MATH-500 LEVEL 4 SAMPLE")
print("="*70)
print(f"Problem: {math_sample['problem']}")
print(f"Gold answer: {math_sample['answer']}")

# ── Run both models ───────────────────────────────────────────────────────────
for model_name, model_path in [
    ("base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("aria", "./aria-merged"),
]:
    print(f"\n\n{'#'*70}")
    print(f"# MODEL: {model_name.upper()}")
    print(f"{'#'*70}")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    for label, problem in [("GSM8K", gsm_sample["question"]), ("MATH L4", math_sample["problem"])]:
        prompt = f"{SYSTEM_PROMPT}<|User|>{problem}<|Assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.1)
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        # Strip prompt, keep only generation
        if "<|Assistant|>" in decoded:
            generation = decoded.split("<|Assistant|>")[-1]
        else:
            generation = decoded

        print(f"\n{'─'*70}")
        print(f"[{model_name.upper()}] {label}")
        print(f"{'─'*70}")
        print(generation)

    del model
    torch.cuda.empty_cache()
