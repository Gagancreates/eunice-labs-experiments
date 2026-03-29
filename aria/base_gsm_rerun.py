# base_gsm_rerun.py — rerun base model on GSM8K to get real token counts
# Run: python base_gsm_rerun.py

import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from math_verify import verify, parse as math_parse
from huggingface_hub import HfApi
import re

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Think through problems carefully inside <think> tags, "
    "then provide a clean final answer."
)
MAX_NEW_TOKENS = 4096
DEVICE = "cuda"
HF_REPO = "Eunice-Labs/aria-eval-results"

def count_think_tokens(text, tokenizer):
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

def extract_boxed(text):
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

print("Loading GSM8K...")
gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
gsm8k = [{"problem": x["question"], "answer": x["answer"].split("####")[-1].strip()}
         for x in gsm8k_raw][:200]
print(f"GSM8K: {len(gsm8k)} problems")

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
model.eval()

results = []
correct_count = 0
total_time = 0.0

pbar = tqdm(gsm8k, desc="base/GSM8K")
for item in pbar:
    prompt = f"{SYSTEM_PROMPT}<|User|>{item['problem']}<|Assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.1)
    elapsed = time.time() - t0
    total_time += elapsed

    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    think_tokens = count_think_tokens(decoded, tokenizer)
    pred = extract_answer(decoded)
    correct = is_correct(pred, item["answer"])
    if correct:
        correct_count += 1

    n = len(results) + 1
    pbar.set_postfix({
        "acc": f"{round(100*correct_count/n,1)}%",
        "avg_s": f"{total_time/n:.1f}s",
        "ETA": f"{round((total_time/n)*(len(gsm8k)-n)/60,1)}m",
    })

    results.append({
        "problem": item["problem"],
        "model_output": decoded,
        "think_tokens": think_tokens,
        "answer_extracted": pred,
        "ground_truth": item["answer"],
        "correct": correct,
        "inference_time_s": round(elapsed, 2),
    })

total_min = round(total_time / 60, 1)
print(f"\nbase/GSM8K done — {correct_count}/200 correct | total: {total_min}m | avg: {round(total_time/200,1)}s/sample")

think = [r["think_tokens"] for r in results if r["think_tokens"] > 0]
mean_tok = round(sum(think)/len(think), 1) if think else 0
acc = round(100*correct_count/200, 1)
print(f"Accuracy: {acc}% | Mean think tokens: {mean_tok}")

with open("eval_base_gsm8k.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved eval_base_gsm8k.json")

try:
    HfApi().upload_file(
        path_or_fileobj="eval_base_gsm8k.json",
        path_in_repo="eval_base_gsm8k.json",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"Pushed eval_base_gsm8k.json to {HF_REPO}")
except Exception as e:
    print(f"Push failed: {e}")
