# Prompt Baseline (P1/P2) — Run Notes
**Date started:** 2026-07-08
**Status:** 🟡 smoke tests in progress (Vast.ai RTX 3090, $0.177/hr, host 155385)

## What this tests
Can instruction-only prompting of the base model reproduce ARIA's efficiency gains —
without any fine-tuning? This is the ablation the paper's Limitations section flags as
most important, and the first baseline reviewers ask for.

- **P1 (`concise`)**: base model told to keep reasoning as brief as possible.
  Tests the *token reduction* claim.
- **P2 (`adaptive`)**: base model told to calibrate depth to difficulty.
  Tests the *adaptive gradient* claim.

Exact system prompts live in `eval_arm.py` (`PROMPT_PRESETS`) and are embedded in each
run's `summary_*.json` — report them verbatim in the paper appendix.

## Setup
- **Model:** deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (bf16, no adapter — stock)
- **Eval:** identical to the paper eval (`custom_eval.py`): GSM8K first 200, MATH-500
  stratified 50/level (243 total), greedy decoding, repetition_penalty 1.1,
  max_new_tokens 4096 (GSM8K, MATH L1/L2) / 8192 (MATH L3–L5)
- **Hardware:** RTX 3090 24GB, Vast.ai
- **Commands:**
  ```bash
  python eval_arm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --label p1-concise  --prompt-preset concise
  python eval_arm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --label p2-adaptive --prompt-preset adaptive
  ```
- **Artifacts:** per-sample JSONs + summaries saved to `results/` locally and
  auto-pushed to `Eunice-Labs/aria-eval-results` under `ablations/`
  (`eval_p1-concise_gsm8k.json`, `eval_p1-concise_math500.json`,
  `summary_p1-concise.json`, same for `p2-adaptive`)
- **Crash safety:** results checkpoint to `results/` every 10 samples and push to HF
  every 50. If the box dies, re-run the same command — it resumes from the checkpoint
  (at most 10 samples lost; pull the last HF copy first if local disk is gone).
  Don't reuse a `--label` across different models/prompts — the checkpoint is keyed by label.

## Results

### Overall (published numbers included for comparison)
| Arm | Bench | Accuracy | Mean Think Tokens | RES |
|-----|-------|----------|-------------------|-----|
| base (paper prompt) | GSM8K | 76.0% | 449.5 | 169.1 |
| ARIA | GSM8K | 78.5% | 203.7 | 385.4 |
| p1-concise | GSM8K | | | |
| p2-adaptive | GSM8K | | | |
| base (paper prompt) | MATH-500 | 77.0% | 1552.8 | 49.6 |
| ARIA | MATH-500 | 72.8% | 847.5 | 85.9 |
| p1-concise | MATH-500 | | | |
| p2-adaptive | MATH-500 | | | |

### MATH-500 per level — token reduction vs base (ARIA: 3.23x / 2.41x / 1.87x / 1.83x / 1.50x)
| Arm | L1 | L2 | L3 | L4 | L5 |
|-----|----|----|----|----|----|
| p1-concise | | | | | |
| p2-adaptive | | | | | |

## Interpretation guide
- **Prompting ≪ ARIA (likely):** strongest possible defense of the fine-tuning approach —
  headline it in the ablation section.
- **P1 matches ARIA's reduction but loses accuracy on hard levels:** ARIA's value is
  *calibration*, not brevity — also a good result, frame it that way.
- **P1/P2 match ARIA on both axes:** the result is a prompting effect; reframe the paper
  honestly around that comparison.
- Watch for: prompted models ignoring the instruction entirely (token counts ≈ base), or
  the instruction changing answer-format compliance (check `missing_think_tag` and a few
  raw outputs before trusting accuracy).

## Run log

### 2026-07-08 — p1-smoke (base + concise prompt, 15 GSM8K, greedy)
| Arm | Acc | Mean Think Tokens | RES |
|-----|-----|-------------------|-----|
| p1-smoke | 40.0% (6/15) | 120.1 | 333.1 |

- The instruction **bites hard**: 120 tokens vs base's ~450 (3.74x fewer) — R1-Distill
  does obey system-prompt brevity instructions.
- But accuracy collapsed 76% → 40%. If real, this is the ideal ablation outcome:
  *prompting trades accuracy for brevity; ARIA achieves brevity while gaining accuracy
  (78.5% @ 203.7 tok, RES 385 vs prompt's 333).*
- Pending verification before trusting it (n=15 is noisy, ±~25pp):
  1. `base-smoke` — paper prompt on the *same 15 problems* (paired control; rules out
     an unusually hard sample)
  2. Manual inspection of the 9 incorrect outputs — real errors vs answer-extraction
     failures (short outputs may change the final-answer format)

### (next) base-smoke — base + paper prompt, same 15 problems
- Result:

## Observations / issues
- Vast host's real HF download speed ~13MB/s (listed 1.2Gbps) — first model download ~20 min.
