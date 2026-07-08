# rescore.py — re-score saved eval JSONs with the fixed answer extraction
# The original scorer missed LaTeX-formatted boxed answers (\boxed{\$18},
# \$70,\!000) and un-boxed bolded answers (**\$64**) — see prompt_baseline_notes.md.
# Works on raw model_output, so no GPU / re-run needed. Applies to our new runs
# AND the published base/ARIA files (fetch those with --hub).
#
# Usage:
#   python rescore.py results/eval_p1-smoke_gsm8k.json
#   python rescore.py --hub eval_base_gsm8k.json        # published paper files
#   python rescore.py results/*.json --write            # update files in place

import os
import json
import argparse

from eval_arm import extract_answer, is_correct, summarize

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="*", help="eval result JSONs to rescore")
ap.add_argument("--hub", nargs="*", default=[],
                help="filenames to fetch from Eunice-Labs/aria-eval-results (repo root)")
ap.add_argument("--write", action="store_true",
                help="write corrected 'correct'/'answer_extracted' back to the file")
args = ap.parse_args()

paths = list(args.files)
for fname in args.hub:
    from huggingface_hub import hf_hub_download
    paths.append(hf_hub_download(repo_id="Eunice-Labs/aria-eval-results",
                                 filename=fname, repo_type="dataset"))

for path in paths:
    with open(path) as f:
        results = json.load(f)
    if not isinstance(results, list):
        print(f"{path}: not a per-sample results file, skipping")
        continue

    flipped = []
    for r in results:
        pred = extract_answer(r["model_output"])
        correct = is_correct(pred, r["ground_truth"])
        if correct != r["correct"]:
            flipped.append((r["ground_truth"], r.get("answer_extracted"), pred, correct))
        r["answer_extracted"] = pred
        r["correct"] = correct

    old_summary_acc = None
    s = summarize(results, by_level=any(r.get("difficulty") for r in results))
    n_correct = sum(r["correct"] for r in results)
    print(f"\n{os.path.basename(path)}: {n_correct}/{len(results)} correct "
          f"→ {s['accuracy']}% acc | {s['mean_think_tokens']} think tokens | RES {s['res_score']}")
    for gold, old_pred, new_pred, now_correct in flipped:
        print(f"  {'WRONG→RIGHT' if now_correct else 'RIGHT→WRONG'}: "
              f"gold={gold} | old extract={old_pred} | new extract={new_pred}")
    if s.get("by_level"):
        for lvl, ls in s["by_level"].items():
            print(f"  L{lvl}: {ls['accuracy']}% @ {ls['mean_think_tokens']} tok (n={ls['n']})")

    if args.write:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote corrected file: {path}")
