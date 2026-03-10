import json
with open('final_training_dataset.json', 'r') as f:
    final_dataset = json.load(f)

print(f"Loaded {len(final_dataset)} examples")

INPUT_FILE = "final_training_dataset.json"
OUTPUT_FILE = "aria_train.jsonl"

SYSTEM_PROMPT = (
    "You are ARIA, a math reasoning assistant. "
    "Think through problems carefully but concisely inside <think> tags, "
    "then provide a clean final answer."
)

def convert(example):
    return {
        "conversations": [
            {
                "role": "system",
                "value": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "value": example["problem"]
            },
            {
                "role": "assistant",
                "value": example["training_text"]  # already has <think>...</think>\n\n{answer}
            }
        ]
    }

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for example in data:
        f.write(json.dumps(convert(example), ensure_ascii=False) + "\n")

print(f"Converted {len(data)} examples → {OUTPUT_FILE}")