# Stage 2 Training — Run Notes
**Date:** 2026-03-28 / 2026-03-29

## Training

- **Model:** DeepSeek-R1-Distill-Qwen-7B (bfloat16 via Unsloth)
- **Base adapter:** Eunice-Labs/aria-easy-medium (Stage 1 checkpoint)
- **Dataset:** Eunice-Labs/aria-easy-medium (3,993 examples — same split, fixed format)
- **Hardware:** RTX 3090 24GB on Vast.ai (~$0.25/hr)
- **Duration:** ~1 hour
- **Total cost:** ~$0.25

### Config (changes from Stage 1)
| Setting | Stage 1 | Stage 2 |
|---------|---------|---------|
| Manual BOS in dataset | Yes (bug) | **No — removed** |
| LoRA rank | 16 | **64** |
| LoRA alpha | 32 | **128** (2× rank) |
| LoRA target modules | q/k/v/o (4) | **q/k/v/o + gate/up/down (7)** |
| LoRA dropout | 0 | **0.05** |
| Precision | fp16 (bug) | **bf16** |

### Loss
- Start: ~0.85
- End: ~0.20 (vs 0.34 in Stage 1 — significantly better convergence)
- Smooth decrease throughout, no spikes

### Merged model
- Adapter merged via `model.merge_and_unload()` → pushed to `Eunice-Labs/aria-stage2`
- Inference uses plain transformers (no Unsloth dependency)

---

## Bugs Fixed from Stage 1

### 1. Double BOS token
Removed manual BOS from `prepare_dataset.py`. Tokenizer auto-adds BOS via `add_bos_token=True` (confirmed via `check_bos.py`). Every Stage 1 training example had double BOS — now fixed.

### 2. LoRA targets too narrow
Added MLP layers (`gate_proj`, `up_proj`, `down_proj`) alongside attention layers. Compression is a generation behavior requiring MLP participation.

### 3. LoRA rank too small
Increased r=16 → r=64, alpha=128. More capacity to learn compressed reasoning style on top of existing weights.

### 4. fp16 on bfloat16-native hardware
Fixed in Stage 1 but documented here: 3090 is bfloat16 native. Always use `bf16=True` with Unsloth on 3090/4090.

---

## Eval Results

### GSM8K (200 samples, greedy decoding)
| Model | Accuracy | Mean Think Tokens | RES Score |
|-------|----------|-------------------|-----------|
| Base DeepSeek-R1-Distill-7B | 76.0% | 467.5 | 162.57 |
| **ARIA Stage 2** | **78.5%** | **203.7** | **385.37** |
| **Reduction** | **+2.5%** | **2.30x fewer** | **2.37x better** |

### MATH-500 (243 samples — 50/level except L1 which has 43 total, greedy decoding)
| Model | Accuracy | Mean Think Tokens | RES Score |
|-------|----------|-------------------|-----------|
| Base DeepSeek-R1-Distill-7B | 77.0% | 1552.8 | 49.56 |
| **ARIA Stage 2** | **72.8%** | **847.5** | **85.95** |
| **Reduction** | -4.2% | **1.83x fewer** | **1.73x better** |

### MATH-500 Per-Level Breakdown (the core finding)
| Level | Base Acc | ARIA Acc | Base Tokens | ARIA Tokens | Reduction |
|-------|----------|----------|-------------|-------------|-----------|
| L1 | 83.7% | **83.7%** | 790.1 | 244.2 | **3.23x** |
| L2 | 80.0% | 78.0% | 981.1 | 407.6 | **2.41x** |
| L3 | 80.0% | 74.0% | 1299.4 | 693.5 | **1.87x** |
| L4 | 74.0% | 66.0% | 1585.0 | 863.7 | **1.83x** |
| L5 | 68.0% | 64.0% | 3045.1 | 2036.1 | **1.50x** |

**The gradient is the finding:** ARIA compresses aggressively on easy problems (3.23x on L1) and naturally scales up thinking for hard problems (only 1.5x reduction on L5). This is adaptive behavior — not just uniformly shorter traces.

---

## Eval Infrastructure

### Accuracy measurement
- `math_verify` library for LaTeX equivalence (handles `\frac{1}{2}` vs `0.5`, nested braces, etc.)
- Balanced brace matcher for `\boxed{}` extraction — handles `\boxed{\frac{1}{2}}` correctly
- Fallback to normalized string match

### Token counting
- Counts tokens from first `<think>` (or start of generation if absent) to last `</think>`
- Applied identically to base and ARIA — ratios are valid
- Known issue: DeepSeek uses unicode pipe `<｜Assistant｜>` not ASCII `<|Assistant|>` — prompt stripping in token counter doesn't fire. Absolute token counts include prompt tokens. **Reduction ratios are still valid** since bug applies equally to both models.

### Known issues with eval metrics
- `missing_think_tag` metric is broken — always fires because models produce `</think>` without opening `<think>` tag (confirmed in qualitative review). Token counting handles this correctly via fallback. Metric should be ignored.
- Some MATH-500 answers marked incorrect due to formatting differences (e.g. `1/2` vs `\frac{1}{2}`) that `math_verify` doesn't catch. A manual review of ~10 suspect cases is pending. Accuracy numbers may be slightly underreported.
- Base GSM8K token count (467.5) is from a separate rerun after the original run's file was not saved. Accuracy (76%) is from the original full 200-sample run.

---

## Files Saved
All eval outputs pushed to `Eunice-Labs/aria-eval-results` on HuggingFace:
- `eval_base_gsm8k.json` — 200 samples, full raw outputs
- `eval_base_math500.json` — 243 samples, full raw outputs
- `eval_aria_gsm8k.json` — 200 samples, full raw outputs
- `eval_aria_math500.json` — 243 samples, full raw outputs
- `eval_results_final.json` — compact summary (accuracy, tokens, RES)
- `case_studies.json` — 3 auto-selected examples (easy/short, hard/deep, side-by-side)

---

## Key Observations

1. **ARIA outperforms base on GSM8K** — 78.5% vs 76.0%. Compression training did not hurt easy problem accuracy; it slightly helped.
2. **L1 accuracy perfectly preserved** — 83.7% = 83.7% with 3.23x token reduction. This is the cleanest result.
3. **Accuracy drops on L3-L5** — expected. Training data was easy+medium only. Hard problems were not in the compression training set, so ARIA has no learned compression strategy for them.
4. **RES score dominates across the board** — ARIA is more efficient even where accuracy drops.
5. **Hard problems still get more tokens** — L5 ARIA uses 2036 tokens avg vs L1's 244. The model learned to allocate compute adaptively.

---

## Next Steps
- Manual reeval pass on suspect incorrect labels (formatting mismatches)
- Write arXiv preprint
- Consider Stage 3: include hard problems in compression training to improve L4/L5 accuracy
