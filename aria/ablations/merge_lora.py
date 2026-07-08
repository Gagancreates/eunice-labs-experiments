# merge_lora.py — merge an ablation LoRA adapter into the base model
# Uses plain transformers + PEFT (Unsloth is unreliable for inference — see
# ../stage1_notes.md). Run: python merge_lora.py --adapter output/s1-uniform --out merged/s1-uniform

import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, help="local adapter dir or HF id")
ap.add_argument("--out", required=True, help="output dir for merged model")
ap.add_argument("--push", default=None, help="optional HF repo id to push merged model")
args = ap.parse_args()

model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(args.adapter)

model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)
print(f"Merged model saved to {args.out}")

if args.push:
    model.push_to_hub(args.push)
    tokenizer.push_to_hub(args.push)
    print(f"Pushed to {args.push}")
