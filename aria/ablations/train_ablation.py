# train_ablation.py — ARIA ablation SFT
# Identical config to ../train.py (Stage 2), parameterized per ablation arm.
# Run: python train_ablation.py --data data/s1-uniform.jsonl --run-name s1-uniform

import argparse

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="local jsonl path or HF dataset id")
ap.add_argument("--run-name", required=True, help="arm name, e.g. s1-uniform")
ap.add_argument("--max-seq-len", type=int, default=3072,
                help="3072 covers easy/medium; use 6144 for s3, 24576 for f0/f2")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--no-push", action="store_true")
args = ap.parse_args()

hub_id = f"Eunice-Labs/aria-ablation-{args.run_name}"

# ── Model ─────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length=args.max_seq_len,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)

# ── Dataset ───────────────────────────────────────────────────────────────────
if args.data.endswith(".jsonl") or args.data.endswith(".json"):
    dataset = load_dataset("json", data_files=args.data, split="train")
else:
    dataset = load_dataset(args.data, split="train")
print(f"Loaded {len(dataset)} examples from {args.data}")

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=args.max_seq_len,
    dataset_num_proc=2,
    args=TrainingArguments(
        output_dir=f"./output/{args.run_name}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        seed=42,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="wandb",
        run_name=f"aria-ablation-{args.run_name}",
        push_to_hub=not args.no_push,
        hub_model_id=hub_id,
        hub_strategy="checkpoint",
    ),
)

trainer.train()
if not args.no_push:
    trainer.push_to_hub()
print(f"Done. Adapter at ./output/{args.run_name}"
      + ("" if args.no_push else f" and {hub_id}"))
