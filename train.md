# Project ARIA — Fine-Tuning Guide

Complete source of truth for training ARIA on compressed reasoning traces.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Step 1 — Convert Dataset](#step-1--convert-dataset)
4. [Step 2 — Unsloth Studio (Recommended)](#step-2--unsloth-studio-recommended)
5. [Step 3 — Manual Training (Kaggle / Colab Fallback)](#step-3--manual-training-kaggle--colab-fallback)
6. [Step 4 — Inference & Evaluation](#step-4--inference--evaluation)
7. [Step 5 — Export & Push to HuggingFace](#step-5--export--push-to-huggingface)
8. [Training Strategy](#training-strategy)
9. [Troubleshooting](#troubleshooting)

---

## Overview

**Goal:** Fine-tune a DeepSeek-R1-Distill model to reason efficiently using compressed thinking traces.

**What ARIA learns:**
- Solve math problems with concise, compressed `<think>` traces
- Maintain correctness while using far fewer reasoning tokens
- Generalize across easy, medium, and hard difficulty levels

**Training approach:** Supervised Fine-Tuning (SFT) with QLoRA via Unsloth

**Models (in order):**
1. `DeepSeek-R1-Distill-Qwen-1.5B` — dry run, fast validation (~30-45 min)
2. `DeepSeek-R1-Distill-Qwen-7B` — full production run (~4-6 hrs)

---

## Dataset

**File:** `final_training_dataset.json`

| Difficulty | Count | Avg Thinking (words) | Notes |
|---|---|---|---|
| Easy | 1,993 | 164 | Compressed 5.9x from 964 |
| Medium | 2,000 | 469 | Compressed 5.9x from 2,789 |
| Hard | 2,000 | 7,640 | Untouched original traces |
| **Total** | **5,993** | | |

**Training format per example:**
```
<think>
{compressed or original thinking trace}
</think>

{clean formatted answer}
```

**Fields available in JSON:**
- `problem` — the math question
- `thinking` — compressed (or original for hard) thinking trace
- `answer` — clean final answer
- `training_text` — pre-built `<think>...</think>\n\n{answer}` string
- `difficulty` — easy / medium / hard
- `source` — dataset origin
- `original_thinking_words` — word count before compression
- `compressed_thinking_words` — word count after compression

---

## Step 1 — Convert Dataset

Unsloth and SFT frameworks expect chat format (ShareGPT style). Run this script locally to convert before uploading.

**Script: `convert_dataset.py`**

```python
import json

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
```

**Run it:**
```bash
python convert_dataset.py
```

This produces `aria_train.jsonl` — ready to upload to Unsloth Studio or use in manual training.

**Verify the output:**
```python
import json

with open("aria_train.jsonl") as f:
    sample = json.loads(f.readline())

for turn in sample["conversations"]:
    print(f"[{turn['role']}]")
    print(turn["value"][:300])
    print()
```

---

## Step 2 — Unsloth Studio (Recommended)

The easiest path — no GPU setup, no environment issues.

### 2.1 Create Account
- Go to **https://unsloth.ai**
- Sign up / log in

### 2.2 Upload Dataset
- Go to **Studio → Datasets → Upload**
- Upload `aria_train.jsonl`
- Format: **ShareGPT / Chat**

### 2.3 Configure Fine-Tune

**Model (Dry Run):**
```
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

**Model (Production):**
```
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

**Recommended Settings:**

| Parameter | Dry Run (1.5B) | Production (7B) |
|---|---|---|
| Max Sequence Length | 4096 | 4096 |
| LoRA Rank (r) | 16 | 16 |
| LoRA Alpha | 16 | 16 |
| Batch Size | 4 | 2 |
| Gradient Accumulation | 4 | 8 |
| Learning Rate | 2e-4 | 2e-4 |
| Epochs | 2 | 2 |
| Warmup Steps | 10 | 10 |
| Optimizer | adamw_8bit | adamw_8bit |
| LR Scheduler | cosine | cosine |

### 2.4 Start Training
- Click **Run**
- Monitor loss curve — should drop steadily and stabilize
- Download the LoRA adapter when done

### 2.5 What Good Training Looks Like
- Loss starts ~1.5-2.5, drops to ~0.3-0.8 within first epoch
- No sudden spikes (if spikes appear → lower learning rate to 1e-4)
- Validation loss tracks training loss (no large gap = no overfitting)

---

## Step 3 — Manual Training (Kaggle / Colab Fallback)

Use this if Unsloth Studio doesn't work or you want full control.

### 3.1 Environment Setup

**On Kaggle:**
- New notebook → Settings → Accelerator: **T4 x2**
- Settings → Internet: **On**

**Install dependencies:**
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
```

**On Colab (A100):**
```python
!pip install unsloth
!pip install --no-deps trl peft accelerate bitsandbytes
```

### 3.2 Load Model

```python
from unsloth import FastLanguageModel
import torch

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # swap to 7B for production
MAX_SEQ_LENGTH = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,           # auto-detect (bf16 on A100, fp16 on T4)
    load_in_4bit=True,    # QLoRA
)
```

### 3.3 Apply LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### 3.4 Load & Format Dataset

```python
from datasets import Dataset
import json

# Upload aria_train.jsonl to Kaggle as a dataset and reference it here
with open("/kaggle/input/aria-dataset/aria_train.jsonl", "r") as f:
    raw = [json.loads(line) for line in f]

# Chat template formatting
def format_example(example):
    conversations = example["conversations"]
    text = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = Dataset.from_list(raw)
dataset = dataset.map(format_example, remove_columns=["conversations"])

print(f"Dataset size: {len(dataset)}")
print(dataset[0]["text"][:500])
```

**If the model doesn't have a chat template, use this fallback formatter:**

```python
SYSTEM_PROMPT = (
    "You are ARIA, a math reasoning assistant. "
    "Think through problems carefully but concisely inside <think> tags, "
    "then provide a clean final answer."
)

def format_example_manual(example):
    convs = example["conversations"]
    system = convs[0]["value"]
    user = convs[1]["value"]
    assistant = convs[2]["value"]

    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return {"text": text}
```

### 3.5 Train

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="aria_output",
        save_steps=100,
        save_total_limit=2,
    ),
)

trainer_stats = trainer.train()
print(trainer_stats)
```

### 3.6 Save LoRA Adapter

```python
model.save_pretrained("aria_lora")
tokenizer.save_pretrained("aria_lora")
print("Saved LoRA adapter to ./aria_lora")
```

---

## Step 4 — Inference & Evaluation

Test the model after training to verify it's working correctly.

### 4.1 Load for Inference

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="aria_lora",   # path to your saved adapter
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### 4.2 Run Inference

```python
SYSTEM_PROMPT = (
    "You are ARIA, a math reasoning assistant. "
    "Think through problems carefully but concisely inside <think> tags, "
    "then provide a clean final answer."
)

def ask_aria(problem, max_new_tokens=512):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response

# Test
problem = "What is the sum of all integers from 1 to 100?"
response = ask_aria(problem)
print(response)
```

### 4.3 Quick Evaluation

Run a small batch to check quality:

```python
import json

test_problems = [
    "What is 15% of 240?",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "Solve for x: 3x + 7 = 22",
    "What is the area of a circle with radius 5?",
    "A bag has 3 red and 5 blue marbles. What is the probability of drawing red?",
]

print("=" * 60)
for problem in test_problems:
    print(f"PROBLEM: {problem}")
    print(f"ARIA: {ask_aria(problem, max_new_tokens=300)}")
    print("=" * 60)
```

**What to look for:**
- `<think>` tags appear and contain actual reasoning
- Thinking is concise (not rambling for pages)
- Final answer after `</think>` is clean and correct
- No repetition loops or gibberish

---

## Step 5 — Export & Push to HuggingFace

### 5.1 Save Merged Model (Optional — large, ~14GB for 7B)

```python
# Only do this if you have enough disk space
model.save_pretrained_merged("aria_merged", tokenizer, save_method="merged_16bit")
```

### 5.2 Push LoRA Adapter to HuggingFace Hub

```python
from huggingface_hub import login

login(token="YOUR_HF_TOKEN")  # get from huggingface.co/settings/tokens

model.push_to_hub("your-username/aria-1.5b-lora", tokenizer=tokenizer)
# or for 7B:
# model.push_to_hub("your-username/aria-7b-lora", tokenizer=tokenizer)
```

### 5.3 Push Merged Model to HuggingFace Hub

```python
model.push_to_hub_merged(
    "your-username/aria-7b",
    tokenizer,
    save_method="merged_16bit",
    token="YOUR_HF_TOKEN"
)
```

### 5.4 Save as GGUF (for local use with Ollama / llama.cpp)

```python
model.save_pretrained_gguf("aria_gguf", tokenizer, quantization_method="q4_k_m")
# or push directly
model.push_to_hub_gguf(
    "your-username/aria-7b-gguf",
    tokenizer,
    quantization_method="q4_k_m",
    token="YOUR_HF_TOKEN"
)
```

---

## Training Strategy

### Why 1.5B First

| | 1.5B (Dry Run) | 7B (Production) |
|---|---|---|
| Training time | ~30-45 min | ~4-6 hrs |
| Purpose | Validate pipeline | Final model |
| GPU needed | T4 (16GB) | T4 x2 or A100 |
| Cost (RunPod) | ~$0.10 | ~$1-2 |

Run 1.5B → check outputs make sense → swap model name → run 7B.

### Epoch Count

- **2 epochs** is the sweet spot for 6K examples
- 1 epoch: may underfit
- 3+ epochs: risk of overfitting on repeated math patterns

### Learning Rate

- Start with `2e-4`
- If loss spikes or doesn't converge → drop to `1e-4`
- If loss is very flat after warmup → increase to `3e-4`

### Sequence Length & Hard Examples

Hard examples average ~7,640 words (~10,000 tokens). With `max_seq_length=4096`, they will be truncated. This is acceptable — the model will see the full problem and most of the thinking, with some truncation at the end. If you want to handle this better:

```python
# Filter out examples longer than 4096 tokens before training
def filter_long(example):
    tokens = tokenizer(example["text"], return_tensors="pt")
    return tokens["input_ids"].shape[1] <= 4096

dataset = dataset.filter(filter_long)
print(f"After filtering: {len(dataset)} examples")
```

Or just train on easy + medium only (3,993 examples, all short):

```python
with open("aria_train.jsonl") as f:
    raw = [json.loads(line) for line in f]

# Filter in the conversion script by difficulty
# Modify convert_dataset.py:
# for example in data:
#     if example["difficulty"] in ("easy", "medium"):
#         f.write(...)
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 16
- Reduce `max_seq_length` to 2048
- Use `load_in_4bit=True` (should already be set)

### Loss Not Decreasing
- Check dataset format — run the verify snippet from Step 1
- Make sure `<think>` tags are present in training text
- Try increasing learning rate to `3e-4`
- Make sure epochs > 1

### Loss Spikes / Diverges
- Lower learning rate to `1e-4`
- Check for corrupted examples in dataset (very long strings, weird characters)
- Add `max_grad_norm=1.0` to TrainingArguments

### Inference Outputs Gibberish
- Model may have learned wrong chat template format
- Try the manual formatter from Step 3.4
- Check that you're using `FastLanguageModel.for_inference(model)` before generating

### Training Very Slow on Kaggle
- Make sure GPU is enabled (not CPU)
- Use `dataset_num_proc=1` instead of 2 on Kaggle
- Set `packing=True` in SFTTrainer to pack short sequences together (improves GPU utilization)

### Unsloth Import Error
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```
Run these in separate cells, restart kernel, then import.

---

## Quick Reference

```
Dataset:        final_training_dataset.json  (5,993 examples)
Converted:      aria_train.jsonl             (ShareGPT chat format)
Dry run model:  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Production:     deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
LoRA rank:      r=16, alpha=16
Epochs:         2
LR:             2e-4
Max seq len:    4096
Batch:          2 (train) x 8 (grad accum) = effective batch 16
```
