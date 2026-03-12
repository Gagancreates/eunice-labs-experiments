# ARIA Training Plan
### Project: Adaptive Reasoning via Difficulty-Stratified SFT
**Model:** DeepSeek-R1-Distill-Qwen-7B  
**Goal:** Teach the model to adaptively calibrate thinking token usage based on problem difficulty — short reasoning for easy/medium, deep reasoning for hard.

---

## The Core Hypothesis

Large reasoning models like DeepSeek-R1 overthink — they use the same amount of thinking tokens regardless of problem difficulty. We fix this by fine-tuning on difficulty-stratified data with compressed reasoning traces for easy/medium problems, using a two-stage curriculum so the model learns *when* to think short and *when* to think deep.

---

## Dataset

| Split | Count | Trace Treatment |
|-------|-------|-----------------|
| Easy + Medium | ~4,000 problems | Compressed 5.9x via Gemini (short, concise thinking) |
| Hard | ~1,993 problems | Full traces kept intact (deep, long thinking) |
| **Total** | **5,993 problems** | — |

**For Stage 2 replay:** Mix 15% of easy/medium samples back into hard training batches to prevent catastrophic forgetting.

**Before training — push dataset to HuggingFace Hub:**
```bash
huggingface-cli login
# Push as private dataset with two splits: easy_medium and hard_with_replay
```

---

## Infrastructure

**Platform:** Vast.ai  
**GPU:** RTX 3090 (24GB VRAM) @ ~$0.20/hr  
**Why not 4090:** Same VRAM, 3090 is 2x cheaper, time difference doesn't justify cost at this scale.

| Stage | Est. Duration | Est. Cost |
|-------|--------------|-----------|
| Stage 1 (Easy + Medium) | ~9 hrs | ~$1.80 |
| Stage 2 (Hard + Replay) | ~4 hrs | ~$0.80 |
| **Total** | **~13 hrs** | **~$2.60** |

**Instance setup on Vast.ai:**
```bash
pip install unsloth trl transformers datasets huggingface_hub -q
huggingface-cli login  # paste HF write token
```

---

## Training Decisions & Rationale

### Why two-stage SFT (not DPO or RL)?

- We already have the dataset — SFT is the right v1 baseline
- DPO requires preference pairs (short vs. long traces per problem) — more work to construct
- RL (GRPO/PPO) is the ideal long-term approach but requires reward modeling and is training-unstable
- **Frame in paper as:** data-efficient baseline; RL-based training is the natural next step

### Why LoRA (not full fine-tune)?

- 24GB VRAM on 3090 — full fine-tune of 7B in bf16 won't fit
- LoRA with 4-bit quantization brings memory usage to ~12-14GB comfortably
- Adapter-only saves (~150MB) are easy to checkpoint and push to HF

### Why lower LR in Stage 2?

- Stage 1 has trained the model toward concise reasoning
- High LR in Stage 2 would overwrite Stage 1 representations — catastrophic forgetting
- `5e-5` in Stage 2 vs `2e-4` in Stage 1 preserves the compression behavior on easy problems

### Why replay in Stage 2?

- Even with low LR, training exclusively on hard problems biases the model back toward long traces
- 15% easy/medium replay keeps the model anchored to concise reasoning on simple inputs

---

## Training Script

```python
# train.py — handles both stages via --stage flag

import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, required=True)  # 1 or 2
args = parser.parse_args()

# ── Load model ──────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" if args.stage == 1
    else "your-username/aria-stage1",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
)

# ── Load dataset ─────────────────────────────────────────────────────────────
if args.stage == 1:
    dataset = load_dataset("your-username/aria-dataset", split="easy_medium")
else:
    dataset = load_dataset("your-username/aria-dataset", split="hard_with_replay")

# ── Trainer ──────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,        # critical for 24GB VRAM
        num_train_epochs=2,
        learning_rate=2e-4 if args.stage == 1 else 5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        push_to_hub=True,
        hub_model_id=f"your-username/aria-stage{args.stage}",
        hub_strategy="checkpoint",          # saves every 100 steps, not just end
    ),
)

trainer.train()
trainer.push_to_hub()
```

**Run Stage 1:**
```bash
python train.py --stage 1
```

**Run Stage 2 (on a fresh instance after Stage 1 completes):**
```bash
python train.py --stage 2
```

---

## Verification Gates — Do These Before Moving On

### After Stage 1 — MUST PASS before starting Stage 2

**1. Thinking token distribution check**

Run inference on 50 easy + 50 hard test samples. Plot `<think>` token counts per group.

```python
from transformers import AutoTokenizer
import re

def count_think_tokens(output_text, tokenizer):
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    if not think_match:
        return 0
    return len(tokenizer.encode(think_match.group(1)))
```

**Expected:** Easy problems should average significantly fewer think tokens than hard ones.  
**Red flag:** If easy and hard token counts are similar → Stage 1 didn't work, don't proceed.

**2. Easy problem accuracy check**

Run Stage 1 model on easy test set. Compare accuracy vs. base DeepSeek-R1-Distill-Qwen-7B.

**Acceptable loss:** ≤3-4% accuracy drop.  
**Red flag:** >5% drop → Gemini compression was too aggressive and removed necessary reasoning steps. This is a dataset problem, not a training problem.

**3. Loss curve sanity check**

Training loss should decrease smoothly. If it spikes erratically or plateaus early, something is wrong with data formatting.

---

### After Stage 2 — Full eval

**1. Calibration eval — the main result**

For each difficulty bucket (easy / medium / hard):
- Mean thinking tokens used
- Accuracy on test set

You want: easy → short + accurate, hard → long + accurate.

**2. Overthinking rate**

On easy problems where the model answers *correctly*, what % used more than 2x the median compressed trace length? Lower = better calibration.

```python
median_compressed_len = 120  # set from your dataset stats
threshold = 2 * median_compressed_len

overthinking_rate = sum(
    1 for tokens, correct in results
    if correct and tokens > threshold
) / len([r for r in results if r["correct"]])
```

**3. Underthinking rate**

On hard problems where the model answers *incorrectly*, what % used fewer than the median full trace length?

```python
median_full_len = 800  # set from your dataset stats

underthinking_rate = sum(
    1 for tokens, correct in results
    if not correct and tokens < median_full_len
) / len([r for r in results if not r["correct"]])
```

**4. Calibration score per bucket**

```
calibration_score = accuracy / mean_thinking_tokens_used
```

Compare your model vs. base R1 per bucket. Your model should have a higher calibration score on easy/medium (same or better accuracy at way fewer tokens).

**5. Forgetting check**

Re-run the Stage 1 verification (easy token distribution + easy accuracy) after Stage 2 completes. If easy problem token counts have crept back up significantly, Stage 2 caused forgetting. Fix: increase replay % to 25% and retrain Stage 2.

---

## What to Log for the Paper

Track and save all of the following — you'll need them for the results section and ablations.

| Metric | When to log | Why |
|--------|-------------|-----|
| Training loss curve (both stages) | During training | Shows stable convergence |
| Mean think tokens — easy/medium/hard | After each stage | Core calibration result |
| Accuracy — easy/medium/hard | After each stage | Shows no accuracy collapse |
| Overthinking rate | After Stage 2 | Quantifies efficiency gain |
| Underthinking rate | After Stage 2 | Quantifies safety of compression |
| Calibration score per bucket | After Stage 2 | Main table metric |
| Base model baseline (same metrics) | Pre-training | Needed for comparison |

**Log everything to a results.json after each eval run. Don't rely on memory.**

```python
import json, datetime

results = {
    "timestamp": datetime.datetime.now().isoformat(),
    "stage": 1,
    "easy_mean_think_tokens": 124,
    "hard_mean_think_tokens": 810,
    "easy_accuracy": 0.87,
    "hard_accuracy": 0.61,
    "overthinking_rate": 0.08,
    "underthinking_rate": 0.21,
}

with open("results_stage1.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Key Risks & Mitigations

| Risk | Signal | Fix |
|------|--------|-----|
| Gemini compression too lossy | Easy accuracy drops >5% after Stage 1 | Re-compress at lower ratio, retrain Stage 1 |
| Catastrophic forgetting | Easy token counts increase after Stage 2 | Increase replay to 25%, lower Stage 2 LR to 2e-5 |
| Stage 1 doesn't compress | Easy/hard token counts similar after Stage 1 | Check data formatting, ensure `<think>` tags are parsed correctly |
| OOM on 3090 | CUDA out of memory error | Reduce `max_seq_length` to 1024, reduce batch size |
| Instance crash mid-training | — | `hub_strategy="checkpoint"` auto-saves every 100 steps to HF |

---

## Order of Operations

```
1. Push dataset to HuggingFace (easy_medium split + hard_with_replay split)
2. Rent RTX 3090 on Vast.ai (~$0.20/hr)
3. Run: python train.py --stage 1
4. Wait ~9 hours — checkpoint auto-pushes to HF
5. Terminate instance
6. Run Stage 1 verification gates — stop if red flags
7. Rent fresh RTX 3090 instance
8. Run: python train.py --stage 2
9. Wait ~4 hours
10. Terminate instance
11. Run full Stage 2 eval
12. Log all results to results.json
```

Total wall time: ~13 hours. Total cost: ~$2.60.