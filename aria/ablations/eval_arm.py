# eval_arm.py — single-arm evaluation for ARIA ablations
# Replicates ../custom_eval.py exactly (prompts, greedy decoding, token caps,
# math_verify + balanced-brace extraction, RES, per-level table) for one model
# + one system prompt, so every arm and prompt baseline is measured identically.
#
# Prompt baselines (no training):
#   python eval_arm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --label p1-concise  --prompt-preset concise
#   python eval_arm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --label p2-adaptive --prompt-preset adaptive
# Trained arms:
#   python eval_arm.py --model merged/s2-shuffled --label s2-shuffled

import os
import re
import json
import time
import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from math_verify import verify, parse as math_parse
from huggingface_hub import HfApi

HF_REPO = "Eunice-Labs/aria-eval-results"
DEVICE = "cuda"
MAX_NEW_TOKENS_DEFAULT = 4096   # GSM8K + MATH L1/L2
MAX_NEW_TOKENS_HARD    = 8192   # MATH L3/L4/L5

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# The exact prompt text is part of the experiment — it is saved into the
# summary JSON and must be reported in the paper appendix.
PROMPT_PRESETS = {
    # verbatim from ../custom_eval.py — used for base + ARIA paper numbers
    "paper": (
        "You are a math reasoning assistant. "
        "Think through problems carefully inside <think> tags, "
        "then provide a clean final answer."
    ),
    # P1: can instruction-only brevity match ARIA's token reduction?
    "concise": (
        "You are a math reasoning assistant. "
        "Think through problems inside <think> tags, keeping your reasoning "
        "as brief as possible — include only the essential steps needed to "
        "reach the answer. Then provide a clean final answer."
    ),
    # P2: can instruction-only difficulty calibration match ARIA's gradient?
    "adaptive": (
        "You are a math reasoning assistant. "
        "Think through problems inside <think> tags, matching your reasoning "
        "depth to the problem's difficulty: for easy problems think in just a "
        "few sentences; reserve long, careful reasoning for genuinely hard "
        "problems. Then provide a clean final answer."
    ),
}

# Published numbers (../README.md, ../stage2_notes.md) for on-screen comparison
PUBLISHED = {
    "base": {"gsm8k": {"acc": 76.0, "tok": 449.5}, "math500": {"acc": 77.0, "tok": 1552.8},
             "by_level": {"1": (83.7, 790.1), "2": (80.0, 981.1), "3": (80.0, 1299.4),
                          "4": (74.0, 1585.0), "5": (68.0, 3045.1)}},
    "aria": {"gsm8k": {"acc": 78.5, "tok": 203.7}, "math500": {"acc": 72.8, "tok": 847.5},
             "by_level": {"1": (83.7, 244.2), "2": (78.0, 407.6), "3": (74.0, 693.5),
                          "4": (66.0, 863.7), "5": (64.0, 2036.1)}},
}


# ── Metrics (verbatim from ../custom_eval.py) ──────────────────────────────────
def count_think_tokens(text, tokenizer):
    """Tokens from first <think> (or start of generation) to LAST </think>."""
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    last_close = text.rfind("</think>")
    if last_close == -1:
        return 0
    first_open = text.find("<think>")
    if first_open != -1:
        think_content = text[first_open + len("<think>"):last_close]
    else:
        think_content = text[:last_close]
    return len(tokenizer.encode(think_content))

def has_think_tag(text):
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    return "<think>" in text

def extract_boxed(text):
    """Balanced brace matcher — handles nested braces e.g. \\boxed{\\frac{1}{2}}."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None

def extract_answer(text):
    if "<|Assistant|>" in text:
        text = text.split("<|Assistant|>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    last_close = text.rfind("</think>")
    if last_close != -1:
        text = text[last_close + len("</think>"):]
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"answer\s+is\s+\$?([0-9][\d\s,\.]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    # bolded final answer, e.g. "pay **\$64** for..." — but NOT numbered headings
    # like "**2.**": require the number not be followed by a bare period
    bolds = re.findall(r"\*\*\s*\\?\$?\s*(-?[\d,]+(?:\.\d+)?)\s*\*\*", text)
    if bolds:
        return bolds[-1]
    nums = re.findall(r"\$?-?[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "").replace("$", "")
    return None

def normalize(ans):
    """Strip LaTeX/formatting noise so \\boxed{\\$70,\\!000} == 70000."""
    if ans is None:
        return None
    ans = ans.replace("\\!", "").replace("\\$", "").replace("\\%", "")
    ans = ans.replace("**", "").replace("%", "")
    return ans.replace(",", "").replace("$", "").replace(" ", "").strip().rstrip(".").lower()

def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    # try math_verify on the raw and the de-noised prediction
    for p in (pred, normalize(pred)):
        try:
            if verify(math_parse(gold), math_parse(p)):
                return True
        except Exception:
            pass
    return normalize(pred) == normalize(gold)


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(model, tokenizer, system_prompt, problem,
                  max_new_tokens=MAX_NEW_TOKENS_DEFAULT):
    prompt = f"{system_prompt}<|User|>{problem}<|Assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0], skip_special_tokens=False)


# ── Eval loop (verbatim behavior from ../custom_eval.py, + checkpointing) ─────
# Results are appended to results/<fname> every SAVE_EVERY samples and pushed to
# HF every PUSH_EVERY, so a crash/preemption loses at most SAVE_EVERY samples.
# Re-running the same command resumes from the checkpoint automatically.
SAVE_EVERY = 10
PUSH_EVERY = 50

def evaluate(model, tokenizer, system_prompt, problems, label, fname, push):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, fname)

    results = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)
        if len(results) >= len(problems):
            print(f"  {label}: checkpoint already complete ({len(results)} samples) — skipping")
            return results
        print(f"  {label}: resuming from checkpoint ({len(results)}/{len(problems)} done)")

    correct_count = sum(r["correct"] for r in results)
    total_time = 0.0
    session_done = 0

    pbar = tqdm(problems[len(results):], desc=label,
                initial=len(results), total=len(problems))
    for item in pbar:
        level = item.get("level")
        max_tok = MAX_NEW_TOKENS_HARD if level in [3, 4, 5] else MAX_NEW_TOKENS_DEFAULT

        t0 = time.time()
        decoded = run_inference(model, tokenizer, system_prompt, item["problem"],
                                max_new_tokens=max_tok)
        elapsed = time.time() - t0
        total_time += elapsed
        session_done += 1

        think_tokens = count_think_tokens(decoded, tokenizer)
        pred = extract_answer(decoded)
        correct = is_correct(pred, item["answer"])
        if correct:
            correct_count += 1

        n = len(results) + 1
        avg_s = total_time / session_done
        pbar.set_postfix({
            "acc": f"{round(100 * correct_count / n, 1)}%",
            "avg_s": f"{avg_s:.1f}s",
            "ETA": f"{round(avg_s * (len(problems) - n) / 60, 1)}m",
        })

        results.append({
            "problem": item["problem"],
            "difficulty": item.get("level", None),
            "model_output": decoded,
            "think_tokens": think_tokens,
            "answer_extracted": pred,
            "ground_truth": item["answer"],
            "correct": correct,
            "has_think_tag": has_think_tag(decoded),
            "inference_time_s": round(elapsed, 2),
        })

        if len(results) % SAVE_EVERY == 0:
            with open(ckpt_path, "w") as f:
                json.dump(results, f, indent=2)
        if push and len(results) % PUSH_EVERY == 0:
            push_to_hub(ckpt_path, fname)

    with open(ckpt_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  {label} done — {correct_count}/{len(problems)} correct | "
          f"session: {round(total_time/60,1)}m | avg: {round(total_time/max(session_done,1),1)}s/sample")
    return results


def summarize(results, by_level=False):
    if not results:
        return {}
    acc = sum(r["correct"] for r in results) / len(results)
    think = [r["think_tokens"] for r in results if r["think_tokens"] > 0]
    mean_tokens = round(sum(think) / len(think), 1) if think else 0
    res_score = round(acc * 100 / mean_tokens * 1000, 2) if mean_tokens > 0 else 0
    summary = {
        "accuracy": round(acc * 100, 1),
        "mean_think_tokens": mean_tokens,
        "res_score": res_score,
        "missing_think_tag": sum(1 for r in results if not r["has_think_tag"]),
        "total": len(results),
    }
    if by_level:
        summary["by_level"] = {}
        for lvl in sorted(set(r["difficulty"] for r in results if r["difficulty"])):
            lvl_res = [r for r in results if r["difficulty"] == lvl]
            lvl_acc = sum(r["correct"] for r in lvl_res) / len(lvl_res)
            lvl_think = [r["think_tokens"] for r in lvl_res if r["think_tokens"] > 0]
            lvl_mean = round(sum(lvl_think) / len(lvl_think), 1) if lvl_think else 0
            summary["by_level"][str(lvl)] = {
                "accuracy": round(lvl_acc * 100, 1),
                "mean_think_tokens": lvl_mean,
                "res_score": round(lvl_acc * 100 / lvl_mean * 1000, 2) if lvl_mean > 0 else 0,
                "n": len(lvl_res),
            }
    return summary


# ── Benchmarks (same subsets as ../custom_eval.py) ─────────────────────────────
def load_benchmarks(gsm8k_n=200, math_per_level=50):
    gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
    math500_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")

    gsm8k = [{"problem": x["question"], "answer": x["answer"].split("####")[-1].strip()}
             for x in gsm8k_raw][:gsm8k_n]

    per_level = defaultdict(list)
    for x in math500_raw:
        per_level[x["level"]].append(x)
    math500 = []
    for lvl in sorted(per_level.keys()):
        for x in per_level[lvl][:math_per_level]:
            math500.append({"problem": x["problem"], "answer": x["answer"], "level": x["level"]})
    return gsm8k, math500


def push_to_hub(path, fname):
    try:
        HfApi().upload_file(
            path_or_fileobj=path,
            path_in_repo=f"ablations/{fname}",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print(f"  Pushed ablations/{fname} → {HF_REPO}")
    except Exception as e:
        print(f"  Warning: failed to push {fname}: {e}")


def save_and_push(obj, fname, push):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved {path}")
    if push:
        push_to_hub(path, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local merged-model dir")
    ap.add_argument("--label", required=True, help="arm name, e.g. p1-concise")
    ap.add_argument("--prompt-preset", default="paper", choices=sorted(PROMPT_PRESETS),
                    help="system prompt; trained arms use 'paper' (the eval default)")
    ap.add_argument("--gsm8k-n", type=int, default=200)
    ap.add_argument("--math-per-level", type=int, default=50)
    ap.add_argument("--skip-gsm8k", action="store_true")
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    push = not args.no_push
    system_prompt = PROMPT_PRESETS[args.prompt_preset]

    print(f"Arm: {args.label} | model: {args.model} | prompt preset: {args.prompt_preset}")
    gsm8k, math500 = load_benchmarks(args.gsm8k_n, args.math_per_level)
    print(f"GSM8K: {len(gsm8k)} problems | MATH-500: {len(math500)} problems")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    summary = {
        "label": args.label,
        "model": args.model,
        "prompt_preset": args.prompt_preset,
        "system_prompt": system_prompt,
        "date": time.strftime("%Y-%m-%d"),
    }

    if not args.skip_gsm8k:
        fname = f"eval_{args.label}_gsm8k.json"
        gsm_results = evaluate(model, tokenizer, system_prompt, gsm8k,
                               f"{args.label}/GSM8K", fname, push)
        summary["gsm8k"] = summarize(gsm_results)
        if push:
            push_to_hub(os.path.join(RESULTS_DIR, fname), fname)

    if not args.skip_math:
        fname = f"eval_{args.label}_math500.json"
        math_results = evaluate(model, tokenizer, system_prompt, math500,
                                f"{args.label}/MATH-500", fname, push)
        summary["math500"] = summarize(math_results, by_level=True)
        if push:
            push_to_hub(os.path.join(RESULTS_DIR, fname), fname)

    save_and_push(summary, f"summary_{args.label}.json", push)

    # ── Comparison vs published numbers ────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"RESULTS — {args.label}  (vs published base / ARIA)")
    print("=" * 78)
    for bench in ["gsm8k", "math500"]:
        if bench not in summary:
            continue
        s = summary[bench]
        b, a = PUBLISHED["base"][bench], PUBLISHED["aria"][bench]
        print(f"\n{bench.upper():<10} {'Acc':>8} {'Tokens':>9} {'RES':>8}")
        print(f"  base     {b['acc']:>7}% {b['tok']:>9} {round(b['acc']/b['tok']*1000,1):>8}")
        print(f"  aria     {a['acc']:>7}% {a['tok']:>9} {round(a['acc']/a['tok']*1000,1):>8}")
        print(f"  {args.label[:8]:<8} {s['accuracy']:>7}% {s['mean_think_tokens']:>9} "
              f"{s['res_score']:>8}   ({round(b['tok']/s['mean_think_tokens'],2)}x fewer than base)")

    if "math500" in summary and "by_level" in summary["math500"]:
        print(f"\nMATH-500 per level:")
        print(f"{'Level':<7} {'BaseAcc':>8} {'AriaAcc':>8} {'ThisAcc':>8} "
              f"{'BaseTok':>8} {'AriaTok':>8} {'ThisTok':>8} {'Red.':>6}")
        for lvl, s in summary["math500"]["by_level"].items():
            bacc, btok = PUBLISHED["base"]["by_level"][lvl]
            aacc, atok = PUBLISHED["aria"]["by_level"][lvl]
            red = round(btok / s["mean_think_tokens"], 2) if s["mean_think_tokens"] else 0
            print(f"L{lvl:<6} {bacc:>7}% {aacc:>7}% {s['accuracy']:>7}% "
                  f"{btok:>8} {atok:>8} {s['mean_think_tokens']:>8} {red:>5}x")

    # Markdown block for the run notes / paper
    print("\nMarkdown (paste into notes):\n")
    for bench in ["gsm8k", "math500"]:
        if bench in summary:
            s = summary[bench]
            print(f"| {args.label} | {bench.upper()} | {s['accuracy']}% | "
                  f"{s['mean_think_tokens']} | {s['res_score']} |")


if __name__ == "__main__":
    main()
