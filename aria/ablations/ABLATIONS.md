# ARIA Ablation Study — Design & Runbook

The paper's Limitations section states it plainly:

> **The ablations are missing.** We ran no comparison against random compression, uniform
> compression, or prompt-based baselines. Of everything we left undone, this is the one that
> matters most.

This directory addresses that. Everything here is designed to be directly comparable to the
published numbers: same seed-42 sample from OpenThoughts-114k-math, same training text format
(ASCII-pipe `<|User|>`/`<|Assistant|>` markers, `<|end▁of▁sentence|>` suffix), same QLoRA
config as `../train.py`, same eval harness as `../custom_eval.py`.

---

## ⚠️ Paper ↔ code discrepancy (read first)

The paper (Fig. 2, Table 1, §3.3) describes fine-tuning on the full **5,993-example mix**
(compressed easy + compressed medium + **uncompressed hard**). The shipped model was not
trained that way:

- `../train.py` loads `Eunice-Labs/aria-easy-medium` — **3,993 easy+medium examples only**.
- `../stage1_notes.md`: "Hard dataset too long for Stage 2 on a 3090 — decision pending."
- `../stage2_notes.md`: "Training data was easy+medium only. Hard problems were not in the
  compression training set."

So the published "adaptive gradient" was produced **without any hard traces in training**.
The graceful behavior on L5 is the base model's residual verbosity on problems far from the
training distribution — not (as the paper claims) learned from a difficulty-graded mixture
that preserved hard traces. §3.3 also reports α=32 / 3 epochs / effective batch 16 / packing,
while the code used α=128 / 2 epochs / effective batch 8 / no packing.

This changes what the ablations must test. We therefore run two regimes:

- **Regime S** — matched to the *shipped* model (easy+medium only, 3090-friendly). Isolates
  what caused the published numbers.
- **Regime F** — the *paper's described* setup (full 5,993 mix, hard kept). This both runs
  the paper's actual claimed experiment for the first time and carries its own ablations.
  Requires a bigger GPU (hard traces ≈ 11k tokens mean, ~20k p95).

Any revision of the paper should either report Regime F as the main experiment or correct
§3/Fig. 2 to describe the easy+medium-only training.

---

## Arms

### Prompt baselines (no training — run these first, cheapest)

| Arm | Model | System prompt | Tests |
|-----|-------|---------------|-------|
| `P1-concise` | base R1-Distill-7B | "keep reasoning as brief as possible" | Can prompting alone buy the token reduction? |
| `P2-adaptive` | base R1-Distill-7B | "calibrate thinking depth to difficulty" | Can prompting alone buy the *gradient*? |

If P1/P2 reproduce ARIA's RES numbers, the fine-tuning contributes nothing — this is the
single most dangerous baseline for the paper, and the reason reviewers ask for it.

### Regime S — ablations of the shipped model (3,993 easy+medium, RTX 3090)

The shipped training set is graded *within* easy+medium: easy→164 words, medium→469 words
(5.9× both). These arms hold the problems fixed and vary only the trace treatment:

| Arm | Treatment | Tests | Prediction if paper's thesis holds |
|-----|-----------|-------|------------------------------------|
| `s1-uniform` | easy AND medium → same 300-word target (token-budget-matched to shipped ≈317 avg) | Does the within-mix length *gradient* matter, or just "shorter traces"? | Gradient flattens: L1 reduction drops, L3+ reduction rises |
| `s2-shuffled` | 150w/400w targets randomly reassigned across tiers (seed 42; same marginal length distribution, correlation with difficulty destroyed) | Is difficulty→length *correlation* the causal signal? | Adaptive gradient disappears or flattens |
| `s3-uncompressed` | original traces, same 3,993 problems, no compression | Is the GSM8K +2.5% from compression, or just from SFT on math data? | Token counts stay near base; accuracy ≈ base |

### Regime F — the paper's described setup (5,993 mix, A100 80GB)

| Arm | Treatment | Tests |
|-----|-----------|-------|
| `f0-graded` | easy→150w, medium→400w, hard→kept (the paper's Fig. 2, run for real) | Does adding preserved hard traces close the L3–L5 accuracy gap? |
| `f1-uniform` | 5.9× ratio applied to *all* tiers (hard → ~1,300w) | Is preserving hard traces what protects hard accuracy? |
| `f2-shuffled` | the three treatments (150w / 400w / keep) randomly reassigned across all 5,993 | Same correlation test as s2, at full scale |

## Interpretation matrix

- **s2/f2 kills the gradient, s1/f1 doesn't** → the difficulty→length correlation is the
  mechanism. Strongest possible support for the paper's thesis.
- **s1 shows the same gradient as shipped ARIA** → the "adaptive gradient" is substantially
  the base model's difficulty response scaled down by a uniform brevity bias. The paper's
  central framing needs rewriting (the token *savings* survive; the "learned adaptivity"
  claim doesn't).
- **P1/P2 match ARIA's RES** → fine-tuning is unnecessary; result is a prompting effect.
- **s3 matches base** → gains come from compression, not from SFT exposure to math. (If s3
  *beats* base on GSM8K, part of the +2.5% is generic SFT effect.)
- **f0 closes the L3–L4 gap** → supports the paper's §5.2 explanation (compression removed
  load-bearing steps near the boundary); also fixes the discrepancy above.

---

## Runbook

All scripts live in this directory. Original paper scripts in `..` are untouched.

### 0. Prompt baselines (no GPU training, eval box only)

```bash
python eval_arm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --label p1-concise  --prompt-preset concise
python eval_arm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --label p2-adaptive --prompt-preset adaptive
```

### 1. Build datasets

Needs `GEMINI_API_KEY` in the environment (compression model defaults to `gemini-2.0-flash`,
matching the paper). Compressions are cached in `cache/compressions.jsonl` — interrupt and
re-run freely. `--seed-cache-from-shipped` pre-loads the cache with the already-published
easy@150/medium@400 compressions from `Eunice-Labs/aria-easy-medium` so `f0-graded` needs
zero new Gemini calls.

```bash
python build_ablation_data.py --arm all --stats-only          # dry run: counts + planned API calls
python build_ablation_data.py --seed-cache-from-shipped
python build_ablation_data.py --arm s1-uniform
python build_ablation_data.py --arm s2-shuffled
python build_ablation_data.py --arm s3-uncompressed           # no API calls
python build_ablation_data.py --arm f0-graded                 # no API calls if cache seeded
python build_ablation_data.py --arm f1-uniform
python build_ablation_data.py --arm f2-shuffled
```

Output: `data/{arm}.jsonl` with the same `{"text", "difficulty"}` schema as the shipped
dataset. Add `--push` to upload to `Eunice-Labs/aria-ablation-{arm}`.

### 2. Train (one run per arm)

```bash
# Regime S — RTX 3090, same config as ../train.py (~1h, ~$0.25/run on Vast.ai)
python train_ablation.py --data data/s1-uniform.jsonl     --run-name s1-uniform
python train_ablation.py --data data/s2-shuffled.jsonl    --run-name s2-shuffled
python train_ablation.py --data data/s3-uncompressed.jsonl --run-name s3-uncompressed --max-seq-len 6144

# Regime F — A100 80GB (hard traces p95 ≈ 20k tokens)
python train_ablation.py --data data/f0-graded.jsonl   --run-name f0-graded   --max-seq-len 24576
python train_ablation.py --data data/f1-uniform.jsonl  --run-name f1-uniform  --max-seq-len 4096
python train_ablation.py --data data/f2-shuffled.jsonl --run-name f2-shuffled --max-seq-len 24576
```

### 3. Merge + eval

```bash
python merge_lora.py --adapter output/s1-uniform --out merged/s1-uniform
python eval_arm.py --model merged/s1-uniform --label s1-uniform
```

`eval_arm.py` replicates `../custom_eval.py` exactly (same prompts, greedy decoding,
repetition_penalty 1.1, adaptive 4096/8192 token caps, math_verify + balanced-brace
extraction, RES score, per-level table) and writes
`results/eval_{label}_{gsm8k,math500}.json` + `results/summary_{label}.json`.
Add `--push` to upload to `Eunice-Labs/aria-eval-results` under `ablations/`.

Compare against the published numbers: base and shipped-ARIA rows in `../README.md`
(GSM8K 76.0%/449.5 tok vs 78.5%/203.7 tok; MATH-500 per-level table).

---

## Cost estimate

| Item | Estimate |
|------|----------|
| Gemini compression (s1+s2+f1+f2, ~14k calls, ~40M in / ~8M out tokens) | ~$8–10 total |
| Regime S training (3 runs, RTX 3090 @ $0.25/hr) | ~$1–2 |
| Regime F training (3 runs, A100 80GB @ ~$1.3/hr, longer seqs) | ~$15–25 |
| Eval (8 arms × 443 problems; ~4h/arm on a 3090 at bf16) | GPU time only |

Known quirks inherited deliberately (so numbers stay comparable): ASCII-pipe chat markers
are plain text, not special tokens; the eval token counter's prompt-strip doesn't fire
(absolute counts include prompt tokens; ratios remain valid); `missing_think_tag` metric is
broken — ignore it (see `../stage2_notes.md`).
