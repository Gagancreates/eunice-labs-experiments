# push_dataset.py — convert eval JSONs to proper HF dataset with splits
# Run: python push_dataset.py

import json
from datasets import Dataset, DatasetDict

HF_REPO = "Eunice-Labs/aria-eval-results"

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Load all 4 files
print("Loading JSON files...")
base_gsm  = load_json("eval_base_gsm8k.json")
base_math = load_json("eval_base_math500.json")
aria_gsm  = load_json("eval_aria_gsm8k.json")
aria_math = load_json("eval_aria_math500.json")

# Strip raw model output to keep dataset small (still available in raw JSON on HF)
def clean(records):
    out = []
    for r in records:
        out.append({k: v for k, v in r.items() if k != "model_output"})
    return out

# Build DatasetDict with 4 splits
ds = DatasetDict({
    "base_gsm8k":   Dataset.from_list(clean(base_gsm)),
    "base_math500": Dataset.from_list(clean(base_math)),
    "aria_gsm8k":   Dataset.from_list(clean(aria_gsm)),
    "aria_math500": Dataset.from_list(clean(aria_math)),
})

print(ds)

print("\nPushing to HF Hub...")
ds.push_to_hub(HF_REPO, private=False)
print(f"Done — https://huggingface.co/datasets/{HF_REPO}")
