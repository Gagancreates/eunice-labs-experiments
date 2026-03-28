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

## Eval (in progress)
- Running 30 easy + 30 medium samples through ARIA
- Measuring mean `<think>` token length per difficulty bucket
- Results → eval_results.json
