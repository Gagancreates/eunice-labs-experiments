# ARIA Stage 1 — Training Runbook

**Goal:** Fine-tune DeepSeek-R1-Distill-Qwen-7B on compressed easy/medium math traces.
**Expected time:** ~8-10 hrs | **Expected cost:** ~$2-2.50

---

## Before you start (local machine)

- [ ] HF token ready (`huggingface-cli whoami` should return `gaganbuilds`)
- [ ] W&B account ready at wandb.ai — grab your API key
- [ ] Vast.ai account funded (~$5 is plenty)
- [ ] `train.py` committed and pushed to GitHub (or ready to paste)

---

## Step 1 — Rent the GPU on Vast.ai

1. Go to [vast.ai](https://vast.ai) → **Search**
2. Filter:
   - GPU: **RTX 3090** (24 GB)
   - Disk: **≥ 40 GB**
   - Image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`
3. Pick the cheapest one ≤ $0.25/hr
4. Click **Rent** → wait ~2 min for it to start
5. Click **Connect** → copy the SSH command and open a terminal

---

## Step 2 — Instance setup

Run these in order:

```bash
# 1. Install dependencies
pip install unsloth trl transformers datasets huggingface_hub wandb

# 2. Login to HuggingFace
huggingface-cli login
# paste your HF token when prompted

# 3. Login to W&B
wandb login
# paste your W&B API key when prompted

# 4. Upload train.py (run this on your LOCAL machine in a separate terminal)
scp -P <port> train.py root@<vast-ip>:/root/train.py
```

---

## Step 3 — Sanity check before training

```bash
# Verify dataset loads correctly
python -c "
from datasets import load_dataset
ds = load_dataset('Eunice-Labs/aria-easy-medium', split='train')
print('Examples:', len(ds))
print('Columns:', ds.column_names)
print('Sample (first 200 chars):', ds[0]['text'][:200])
"
```

Expected output:
```
Examples: 3993
Columns: ['text', 'difficulty']
Sample: <|begin▁of▁sentence|>You are ARIA...
```

---

## Step 4 — Run training

```bash
python train.py --stage 1
```

**What to watch:**
- Loss should start ~2.0-2.5 and decrease smoothly
- Check W&B dashboard for live loss curve → `wandb.ai/your-username`
- If you see OOM → restart with `--per_device_train_batch_size 1` (already set)
- Checkpoints auto-push to `Eunice-Labs/aria-stage1` every 100 steps

---

## Step 5 — Things to save during training (for paper)

### Auto-saved by W&B
- [ ] Loss curve (download as PNG from W&B dashboard)
- [ ] Learning rate schedule plot
- [ ] GPU memory usage

### Manually grab after training completes
```bash
# Save the final loss value
# (visible at end of training output — screenshot or copy it)

# Check model pushed to HF
python -c "
from huggingface_hub import list_repo_refs
refs = list_repo_refs('Eunice-Labs/aria-stage1')
print('Branches:', [b.name for b in refs.branches])
"
```

---

## Step 6 — Run eval (same instance, after training)

```bash
python -c "
from unsloth import FastLanguageModel
from datasets import load_dataset
import json, re

model, tokenizer = FastLanguageModel.from_pretrained(
    'Eunice-Labs/aria-stage1',
    max_seq_length=3072,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Load 30 samples from each difficulty bucket for quick eval
ds = load_dataset('Eunice-Labs/aria-easy-medium', split='train')
easy   = [x for x in ds if x['difficulty'] == 'easy'][:30]
medium = [x for x in ds if x['difficulty'] == 'medium'][:30]

results = {}
for label, samples in [('easy', easy), ('medium', medium)]:
    think_lens = []
    for s in samples:
        inputs = tokenizer(s['text'].split('<|Assistant|>')[0] + '<|Assistant|>', return_tensors='pt').to('cuda')
        out = model.generate(**inputs, max_new_tokens=1024)
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        think = re.findall(r'<think>(.*?)</think>', decoded, re.DOTALL)
        think_lens.append(len(tokenizer.encode(think[0])) if think else 0)
    results[label] = {
        'mean_think_tokens': round(sum(think_lens)/len(think_lens), 1),
        'max_think_tokens': max(think_lens),
    }

print(json.dumps(results, indent=2))
" 2>&1 | tee eval_results.json
```

---

## Step 7 — Download artifacts before destroying instance

```bash
# On your LOCAL machine:
scp -P <port> root@<vast-ip>:/root/eval_results.json ./eval_results.json
```

Model adapter is already on HF at `Eunice-Labs/aria-stage1` so nothing else to save.

**Then destroy the instance on Vast.ai.**

---

## Paper artifacts checklist

| Artifact | Source | Status |
|----------|--------|--------|
| Training loss curve | W&B → export PNG | [ ] |
| LR schedule plot | W&B → export PNG | [ ] |
| Mean think tokens (easy) | eval_results.json | [ ] |
| Mean think tokens (medium) | eval_results.json | [ ] |
| Compression ratio table | already in dataset stats | [ ] |
| Sample output (easy problem) | grab 1 nice example from eval | [ ] |
| Sample output (medium problem) | grab 1 nice example from eval | [ ] |

---

## Cost breakdown

| Item | Time | Rate | Cost |
|------|------|------|------|
| Instance setup + sanity check | ~30 min | $0.25/hr | ~$0.13 |
| Stage 1 training | ~8-10 hrs | $0.25/hr | ~$2.00 |
| Eval run | ~30 min | $0.25/hr | ~$0.13 |
| **Total** | | | **~$2.25** |

---

## If something goes wrong

| Problem | Fix |
|---------|-----|
| OOM during training | Already at batch 1 — reduce `max_seq_length` to `2048` in `train.py` |
| Loss spikes / diverges | Stop, check if dataset loaded correctly (Step 3) |
| W&B not logging | Set `report_to="none"` in train.py and just watch terminal |
| Push to HF fails | Run `trainer.push_to_hub()` manually after training |
| Instance dies mid-training | Resume from latest checkpoint at `Eunice-Labs/aria-stage1` |
