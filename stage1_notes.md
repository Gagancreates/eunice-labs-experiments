# Stage 1 Training — Run Notes
**Date:** 2026-03-28

## Training

- **Model:** DeepSeek-R1-Distill-Qwen-7B (4-bit via Unsloth)
- **Dataset:** Eunice-Labs/aria-easy-medium (3,993 examples)
- **Hardware:** RTX 3090 24GB on Vast.ai (~$0.25/hr)
- **Duration:** ~1 hour (4s/step × 1,000 steps)
- **Total cost:** ~$0.25

### Config
- LoRA r=16, alpha=32, target: q/k/v/o projections
- Epochs: 2, LR: 2e-4 cosine, effective batch size: 8
- max_seq_length: 3,072 (covers 95th pct of easy/medium)
- bf16 (not fp16 — 3090 is bfloat16 native)

### Loss
- Start: ~0.85
- End: ~0.34
- Smooth decrease, no spikes — stable training

---

## Issues Encountered

### 1. fp16 vs bf16
Initial train.py used `fp16=True`. Unsloth threw an error because the 3090 loads the model in bfloat16 natively. Fixed by switching to `bf16=True`.
> **Lesson:** Always check `Bfloat16 = TRUE/FALSE` in Unsloth's startup banner before setting precision.

### 2. Unsloth inference KV cache bug
`FastLanguageModel.for_inference()` caused a RoPE shape mismatch crash during eval:
```
RuntimeError: output with shape [1, 28, 1, 128] doesn't match broadcast shape [1, 28, 135, 128]
```
Unsloth patches model.generate() at the class level via a `.pth` auto-import — even when not explicitly imported. Bug triggered when loading a LoRA adapter from HF Hub through PEFT on top of an Unsloth-quantized base.

**Fix:** Uninstalled Unsloth for eval, used standard transformers + PEFT + bitsandbytes instead.

> **Lesson:** Unsloth is great for training but unreliable for inference with externally-loaded LoRA adapters. Use vLLM or plain transformers for eval.

### 3. Inference speed without Unsloth
Plain transformers 4-bit inference: ~34s per sample. 60 samples = ~34 minutes for eval.
Acceptable for a one-time paper eval but not for production.

---

## Token Length Analysis (pre-training)

| Split | Examples | Mean tokens | 95th pct | Max |
|-------|----------|-------------|----------|-----|
| easy_medium | 3,993 | 1,287 | 2,433 | 10,070 |
| hard_with_replay | 2,598 | 11,106 | 20,772 | 35,943 |

- max_seq_length=3,072 covers 95% of easy/medium cleanly
- Hard dataset too long for Stage 2 on a 3090 — decision pending

---

## Eval Results

### Run 1 (max_new_tokens=512) — 30 samples each
| Bucket | Mean think tokens | Samples with think |
|--------|------------------|--------------------|
| Easy | 288.7 | 21/30 (70%) |
| Medium | 108.3 | 6/30 (20%) |

Low `samples_with_think` on medium was due to 512 token limit truncating mid-think — regex found no closing `</think>` tag.

### Run 2 (max_new_tokens=2048) — 10 samples each
| Bucket | Mean think tokens | Max | Samples with think |
|--------|------------------|-----|-------------------|
| Easy | 645.0 | 1,532 | 9/10 (90%) |
| Medium | 656.8 | 1,643 | 6/10 (60%) |

### Comparison vs base model
| | Think tokens (easy) |
|--|--|
| Base DeepSeek-R1-Distill-Qwen-7B | ~1,200+ |
| ARIA Stage 1 | ~717 (per sample with think) |
| Training target (compressed) | ~200 |

**ARIA thinks ~40% less than base model.** Not as concise as training target but direction is correct.

### Key findings

**1. Missing opening `<think>` tag**
Model sometimes generates thinking content + `</think>` but skips the opening `<think>` tag. Regex misses these. Actual thinking rate is higher than `samples_with_think` suggests. Fix: update regex to also capture `content</think>` pattern.

**2. Double BOS token in eval prompts**
Tokenizer auto-adds BOS on top of manually included BOS in prompt string. Minor issue — doesn't affect output quality.

**3. Thinking is genuinely concise when tags are present**
Qualitative check on "What is 2+2?" showed 3-sentence think trace. Correct answer. Format almost right.

---

## Consultation — AI Engineer Review (2026-03-28)

### Benchmarks
- Held-out OpenThoughts eval is in-distribution — reviewers will dismiss it
- **Must use:** GSM8K (1,319 problems) + MATH-500 (500 problems, per-level breakdown)
- ARC-Challenge optional for out-of-domain transfer
- Use **lm-evaluation-harness** (EleutherAI) — don't roll custom accuracy extraction
- Merge LoRA adapter before running lm-eval: `model.merge_and_unload()`

### Publishability threshold (GSM8K)
| Accuracy | Verdict |
|----------|---------|
| >84% | Solid paper, clear win |
| 80-84% | Publishable if token reduction is 3x+, frame as tradeoff study |
| <80% | Something broke in training, needs another run |

- Base DeepSeek-R1-Distill-Qwen-7B: ~86-89% on GSM8K
- **Real story isn't average reduction — it's adaptive behavior:** big reduction on easy, near-zero on hard. That's the finding.

### Venue
- arXiv preprint first, always
- If results strong → NeurIPS 2025 efficient ML / reasoning workshop
- Ship fast, iterate

---

## Bugs Found in Stage 1

### 1. Double BOS token (confirmed)
`tokenizer.add_bos_token = True` and `bos_token_id = 151646`.
Encoding "test" returns `[151646, 1944]` — tokenizer auto-adds BOS.
Our `prepare_dataset.py` also manually prepended BOS → every training example had **double BOS**.
Not catastrophic (loss still converged) but corrupted format for every example. Likely contributed to weaker compression.

### 2. LoRA targets too narrow
Only targeted attention: `q_proj, v_proj, k_proj, o_proj`.
Compression is a **generation behavior** — needs MLP layers too.
Missing: `gate_proj, up_proj, down_proj`.

### 3. LoRA rank too small
r=16 → insufficient capacity to learn a new reasoning style on top of existing weights.

---

## Stage 2 Training Plan (next run)

### Changes from Stage 1
| Setting | Stage 1 | Stage 2 |
|---------|---------|---------|
| Manual BOS in dataset | Yes (bug) | **No — removed** |
| LoRA rank | 16 | **64** |
| LoRA target modules | q/k/v/o (4) | **q/k/v/o + gate/up/down (7)** |
| lora_alpha | 32 | **128** (2x rank) |
| Eval benchmark | Training data (wrong) | **GSM8K + MATH-500** |
| max_new_tokens in eval | 512 (truncating) | **8192** |
| Inference method | transformers+PEFT | **merge adapter → lm-eval-harness** |

### Action items in order
1. Fix `prepare_dataset.py` — remove manual BOS
2. Regenerate `easy_medium.jsonl` + `hard_with_replay.jsonl`
3. Re-push both datasets to HF
4. Update `train.py` — r=64, alpha=128, all 7 target modules
5. Retrain on Vast.ai RTX 3090
6. Merge adapter: `model.merge_and_unload()` → push merged to HF
7. Run GSM8K + MATH-500 via lm-eval-harness
8. Report: accuracy + mean think tokens per difficulty level
