import json

with open("aria_train.jsonl") as f:
    sample = json.loads(f.readline())

for turn in sample["conversations"]:
    print(f"[{turn['role']}]")
    print(turn["value"][:300])
    print()