# train.py — ARIA Stage 2 SFT
# Fixes from Stage 1: no manual BOS, r=64, all 7 LoRA target modules
# Run: python train.py

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

MAX_SEQ_LEN = 3072  # covers 95th pct of easy/medium (2,433 tokens)

# ── Model ─────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length=MAX_SEQ_LEN,
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
# Pre-formatted text column with DeepSeek special tokens + <think> tags intact
dataset = load_dataset("Eunice-Labs/aria-easy-medium", split="train")
print(f"Loaded {len(dataset)} examples")

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        seed=42,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,              # keep only last 3 checkpoints on disk
        report_to="wandb",
        run_name="aria-stage2",
        push_to_hub=True,
        hub_model_id="Eunice-Labs/aria-stage2",
        hub_strategy="checkpoint",       # pushes to HF every save_steps
    ),
)

trainer.train()
trainer.push_to_hub()
print("Done. Model at Eunice-Labs/aria-stage2")
