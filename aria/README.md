# ARIA: Adaptive Reasoning through Difficulty-Graded Trace Compression

ARIA is a fine-tuned version of DeepSeek-R1-Distill-Qwen-7B trained to produce concise reasoning traces while maintaining mathematical accuracy. The model learns to allocate thinking tokens proportional to problem difficulty rather than generating uniformly verbose traces.

## Key Results

### GSM8K (200 samples, greedy decoding)

| Model | Accuracy | Avg Think Tokens | RES Score |
|-------|----------|-----------------|-----------|
| DeepSeek-R1-Distill-7B (base) | 76.0% | 467.5 | 162.6 |
| ARIA | **78.5%** | **203.7** | **385.4** |

Token reduction: **2.30x** with +2.5% accuracy improvement.

### MATH-500 Per-Level Breakdown (243 samples, greedy decoding)

| Level | Base Acc | ARIA Acc | Base Tokens | ARIA Tokens | Reduction |
|-------|----------|----------|-------------|-------------|-----------|
| L1 | 83.7% | **83.7%** | 790 | 244 | **3.23x** |
| L2 | 80.0% | 78.0% | 981 | 408 | **2.41x** |
| L3 | 80.0% | 74.0% | 1299 | 694 | **1.87x** |
| L4 | 74.0% | 66.0% | 1585 | 864 | **1.83x** |
| L5 | 68.0% | 64.0% | 3045 | 2036 | **1.50x** |

The model compresses aggressively on easy problems and naturally scales thinking for hard problems. This adaptive behavior is emergent -- it was not explicitly trained.

RES Score = accuracy / mean_think_tokens * 1000. Higher is better.

## Method

ARIA is trained via supervised fine-tuning (SFT) on compressed thinking traces. Reasoning traces from the [OpenThoughts-114k-math](https://huggingface.co/datasets/open-r1/OpenThoughts-114k-math) dataset were compressed using Gemini 2.0 Flash, targeting 5-6x compression on easy and medium difficulty problems. Hard problems were left uncompressed.

The model is trained to reproduce these compressed traces, learning a more efficient reasoning style while retaining the base model's mathematical capability.

### Training Configuration

| Setting | Value |
|---------|-------|
| Base model | DeepSeek-R1-Distill-Qwen-7B |
| Method | SFT with QLoRA (Unsloth) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Target modules | q, k, v, o, gate, up, down projections |
| Training examples | 3,993 (easy + medium) |
| Hardware | RTX 3090 24GB |
| Training time | ~1 hour |

### Dataset

| Split | Examples | Compression |
|-------|----------|-------------|
| Easy | 1,993 | ~6x (964 to 164 words avg) |
| Medium | 2,000 | ~6x (2,789 to 469 words avg) |

Dataset available at [Eunice-Labs/aria-easy-medium](https://huggingface.co/datasets/Eunice-Labs/aria-easy-medium).

## Model

The merged model (no adapter dependency) is available at [Eunice-Labs/aria-7b-merged](https://huggingface.co/Eunice-Labs/aria-7b-merged).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Eunice-Labs/aria-7b-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Eunice-Labs/aria-7b-merged")

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Think through problems carefully inside <think> tags, "
    "then provide a clean final answer."
)

problem = "What is the sum of the first 100 positive integers?"
prompt = f"{SYSTEM_PROMPT}<|User|>{problem}<|Assistant|>"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Evaluation

Eval scripts and full results are in this directory. Raw model outputs and per-sample results are available at [Eunice-Labs/aria-eval-results](https://huggingface.co/datasets/Eunice-Labs/aria-eval-results).

Accuracy is measured using [math-verify](https://github.com/huggingface/math-verify) for LaTeX equivalence with normalized string match as fallback. Token counts cover the span from the first `<think>` tag (or start of generation) to the last `</think>` tag, tokenized with the model's own tokenizer.

## Files

| File | Description |
|------|-------------|
| `train.py` | Stage 2 training script |
| `custom_eval.py` | Full paper evaluation (GSM8K + MATH-500) |
| `base_gsm_rerun.py` | Base model GSM8K rerun for token count baseline |
| `sample_inspect.py` | Qualitative output inspection for two samples |
| `smoke_eval.py` | Quick token reduction check (no accuracy) |
| `accuracy_eval.py` | Small-scale accuracy + token check |
| `push_dataset.py` | Push eval JSONs as structured HF dataset |
| `stage1_notes.md` | Stage 1 training notes and bug log |
| `stage2_notes.md` | Stage 2 training notes, full results, known issues |
