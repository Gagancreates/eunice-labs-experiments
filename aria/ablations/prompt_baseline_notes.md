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

### 2026-07-08 — base-smoke (base + paper prompt, same 15 problems)
| Arm | Acc | Mean Think Tokens | RES |
|-----|-----|-------------------|-----|
| base-smoke | 73.3% (11/15) | 388.2 | 188.9 |

- Matches the published full-run base numbers (76.0% @ 449.5) → the 15-problem sample is
  representative, and the harness is consistent with the paper eval.
- **Paired result on identical problems: concise prompt 40.0% @ 120.1 vs paper prompt
  73.3% @ 388.2.** The brevity instruction alone costs ~33 accuracy points for a 3.2x
  token cut. ARIA gets a similar cut (2.2x) while *gaining* accuracy.
- Remaining check: inspect p1-smoke's 9 wrong outputs for extraction artifacts before
  treating the accuracy collapse as real.

### 2026-07-08 — ⚠️ scorer bug found: p1-smoke's accuracy collapse was mostly fake
Manual inspection of p1-smoke's 9 "wrong" answers: **5 were correct but mis-scored.**
- The concise prompt makes the model emit LaTeX currency inside the box —
  `\boxed{\$18}`, `\boxed{\$70,\!000}` — and `normalize()` stripped `$` but not `\`,
  so `\18 != 18`. (4 cases)
- One answer was correct but un-boxed (`**\$64**`); the last-number fallback grabbed
  "the 16 glasses" instead. (1 case)

**Corrected paired smoke result: concise 11/15 (73.3%) @ 120.1 tok vs paper prompt
11/15 (73.3%) @ 388.2 tok.** On this sample, prompting matches base accuracy at 3.2x
fewer tokens — i.e. the prompt baseline may genuinely rival ARIA. P1 full run is now
*essential*, with the fixed scorer.

Fixes (in `eval_arm.py`, mirrored in `rescore.py` which re-scores saved outputs without
GPU): normalize strips `\$ \! \% ** %`; extraction tries bolded final answers before
the last-number fallback; math_verify also tried on the de-noised prediction.

**Consequence for the paper:** the published base/ARIA numbers were produced with the
buggy scorer (stage2_notes.md already suspected under-counting). Before the final
ablation table, re-score the published raw outputs:
`python rescore.py --hub eval_base_gsm8k.json eval_aria_gsm8k.json eval_base_math500.json eval_aria_math500.json`

### 2026-07-08 — rescored smokes with fixed scorer (verified on box)
| Arm (same 15 GSM8K) | Acc | Mean Think Tokens | RES |
|-----|-----|-------------------|-----|
| base-smoke (paper prompt) | 80.0% (12/15) | 388.2 | 206.1 |
| p1-smoke (concise prompt) | 73.3% (11/15) | 120.1 | 610.6 |

- Bold-regex regression fixed (was grabbing numbered headings like `**2.**`).
- Corrected picture: concise prompt costs ~1 problem (73.3% vs 80.0%) for a 3.2x token
  cut on this sample. RES 610 would *beat* ARIA's 385 — the prompt baseline is a live
  threat to the paper's framing. n=15 → full P1 run launched to settle it.
- Note: the old scorer also under-scored the base model here (+1 flip). Published
  base/ARIA numbers must be rescored before the final table.

### 2026-07-08 — P1 full run launched (200 GSM8K + 243 MATH-500, ~3–4h)

### 2026-07-08 — P1 GSM8K complete (200 problems, fixed grader)
| Arm | Acc | Mean Think Tokens | RES |
|-----|-----|-------------------|-----|
| base (paper prompt, rescored) | 87.5% | 467.5 | 187.2 |
| ARIA (rescored) | 85.5% | 203.7 | 419.7 |
| **p1-concise (base + brief prompt)** | **86.5%** (173/200) | **138.9** | **622.9** |

**On GSM8K the prompt baseline strictly dominates ARIA**: accuracy within noise of both
base and ARIA, 1.5x fewer tokens than ARIA, 3.4x fewer than base, zero training.
ARIA's case now rests entirely on MATH-500 L3–L5 (in progress): does prompted brevity
collapse on hard problems where ARIA degrades gracefully?

### 2026-07-08 — 🚨 published numbers rescored with fixed scorer (raw HF outputs)
| | GSM8K Acc | Tok | RES | MATH-500 Acc | Tok | RES |
|---|---|---|---|---|---|---|
| base (publ. → corrected) | 76.0 → **87.5** | 467.5 | 187.2 | 77.0 → **78.2** | 1552.8 | 50.4 |
| ARIA (publ. → corrected) | 78.5 → **85.5** | 203.7 | 419.7 | 72.8 → **73.3** | 847.5 | 86.4 |

MATH-500 per level, corrected accuracy (tokens unchanged):
| Level | Base | ARIA | Δ |
|-------|------|------|---|
| L1 | 86.0 | 83.7 | −2.3 |
| L2 | 82.0 | 80.0 | −2.0 |
| L3 | 82.0 | 74.0 | −8.0 |
| L4 | 74.0 | 66.0 | −8.0 |
| L5 | 68.0 | 64.0 | −4.0 |

Consequences for the paper:
- **The "+2.5% GSM8K accuracy" headline was a grading artifact** (buggy scorer hit base
  harder: 23 flips vs ARIA's 14, almost all `\$…` money answers). Corrected: ARIA −2.0pp.
- **"L1 accuracy perfectly preserved" is gone too**: base L1 corrects to 86.0 vs ARIA 83.7.
- Token reductions, the adaptive gradient, and RES dominance (420 vs 187; 86 vs 50) all stand.
- Corrected base GSM8K (87.5%) finally matches the expected ~86–89% for R1-Distill-7B
  (stage1_notes consultation) — the published 76% was depressed by the scorer.
- Revision must reframe: ARIA = ~2x token cut for a small accuracy cost, defended by RES —
  not an accuracy improvement.

### 2026-07-08 — candidate reframing for the revision (agreed in discussion)
ARIA never trained on hard traces (3,993 easy+medium only), so the paper's mechanism
("hard traces kept long teach depth-preservation") is impossible. The honest — and more
interesting — claim: *train on compressed easy/medium only, and difficulty-adaptive
token allocation emerges from what the fine-tune leaves intact on out-of-distribution
(hard) problems.* The gradient is real; its explanation changes. s2-shuffled is the
direct test of this claim; f0-graded (full 5,993 mix on A100) tests the paper's original
mechanism as a comparison point.

## Observations / issues
- Vast host's real HF download speed ~13MB/s (listed 1.2Gbps) — first model download ~20 min.
