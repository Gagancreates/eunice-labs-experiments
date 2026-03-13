# train.py — ARIA two-stage SFT
# Stage 1: python train.py --stage 1
# Stage 2: python train.py --stage 2

import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, required=True)  # 1 or 2
args = parser.parse_args()

MAX_SEQ_LEN = 3072  # covers 95th pct of easy/medium (2,433 tokens)

# ── Load model ────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" if args.stage == 1
    else "Eunice-Labs/aria-stage1",
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",  # unsloth's optimized checkpointing
)

# ── Load dataset ──────────────────────────────────────────────────────────────
# Both datasets have a `text` column — pre-formatted with DeepSeek special tokens:
# <|begin▁of▁sentence|>{system}<|User|>{problem}<|Assistant|><think>...</think>\n\n{answer}<|end▁of▁sentence|>
if args.stage == 1:
    dataset = load_dataset("Eunice-Labs/aria-easy-medium", split="train")
else:
    dataset = load_dataset("Eunice-Labs/aria-hard-replay", split="train")

print(f"Stage {args.stage} — {len(dataset)} examples loaded")

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    args=TrainingArguments(
        report_to="wandb",
        run_name=f"aria-stage{args.stage}",
        logging_dir="./logs",
        output_dir="./output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4 if args.stage == 1 else 5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        push_to_hub=True,
        hub_model_id=f"Eunice-Labs/aria-stage{args.stage}",
        hub_strategy="checkpoint",  # auto-saves every 100 steps to HF
    ),
)

trainer.train()
trainer.push_to_hub()