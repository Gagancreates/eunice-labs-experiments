# eval.py — ARIA Stage 1 evaluation
# Run: python eval.py

import re
import json
import time
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm

model, tokenizer = FastLanguageModel.from_pretrained(
    "Eunice-Labs/aria-stage1",
    max_seq_length=3072,
    load_in_4bit=True,
)
# NOTE: do NOT call FastLanguageModel.for_inference() — causes KV cache shape bug

ds = load_dataset("Eunice-Labs/aria-easy-medium", split="train")
easy   = [x for x in ds if x["difficulty"] == "easy"][:30]
medium = [x for x in ds if x["difficulty"] == "medium"][:30]

results = {}
for label, samples in [("easy", easy), ("medium", medium)]:
    think_lens = []
    for s in tqdm(samples, desc=f"{label:6s}", unit="sample"):
        t0 = time.time()
        prompt = s["text"].split("<|Assistant|>")[0] + "<|Assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        think = re.findall(r"<think>(.*?)</think>", decoded, re.DOTALL)
        think_lens.append(len(tokenizer.encode(think[0])) if think else 0)

    results[label] = {
        "mean_think_tokens": round(sum(think_lens) / len(think_lens), 1),
        "max_think_tokens": max(think_lens),
        "samples_with_think": sum(1 for t in think_lens if t > 0),
    }
    print(f"{label} done — mean think tokens: {results[label]['mean_think_tokens']}")

print(json.dumps(results, indent=2))
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to eval_results.json")
