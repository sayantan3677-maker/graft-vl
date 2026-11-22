"""Summarize lmms-eval runs (scripts/run_benchmarks.sh outputs) into one comparison table.

Per task: the official lmms-eval metric as reported, with two corrections that matter for --limit runs:
  * ocrbench: lmms-eval divides the summed score by 1000 regardless of --limit → reported here as mean score (×100).
  * MCQ tasks ai2d / realworldqa: also a "lenient letter" accuracy = official per-sample exact_match OR the option letter
    extracted from answers like "Answer: B" / "(B)" (the official ai2d filter only accepts "B." or a bare "B").
Usage: python scripts/summarize_benchmarks.py --run ours=reports/benchmarks/X-quick --run ceiling=reports/benchmarks/Y-quick \
          [--json out.json] [--md out.md]
"""

import argparse
import glob
import json
import re
from pathlib import Path

PRIMARY = {  # task -> (metric key in results.json, display name, scale)
    "docvqa_val": ("anls,none", "ANLS", 100), "infovqa_val": ("anls,none", "ANLS", 100),
    "chartqa": ("relaxed_overall,none", "relaxed acc", 100), "textvqa_val": ("exact_match,none", "VQA acc", 100),
    "ocrbench": (None, "score/n", 100), "ai2d": ("exact_match,flexible-extract", "EM (flexible-extract)", 100),
    "mmbench_en_dev": ("gpt_eval_score,none", "gpt_eval_score (rule match; no GPT key)", 1),
    "pope": ("pope_f1_score,none", "F1", 100), "realworldqa": ("exact_match,none", "EM", 100),
    "mmmu_val": ("mmmu_acc,none", "acc", 100),
}
MCQ = {"ai2d", "realworldqa"}  # tasks with a per-sample exact_match field; lenient = official OR letter-extracted
LETTER = re.compile(r"^\s*(?:the\s+)?(?:answer|option)?\s*(?:is)?\s*[:：]?\s*\(?([A-Z])\)?(?:[.:)\s,]|$)", re.I)


def lenient(resp: str, target: str):
    m = LETTER.match(resp.strip())
    pred = m.group(1).upper() if m else resp.strip().rstrip(".").upper()
    return pred == str(target).strip().upper()


def load_run(d: str) -> dict:
    res_files = sorted(f for f in glob.glob(f"{d}/**/*_results.json", recursive=True) if "results" in json.load(open(f)))
    if not res_files:
        return {}
    res = json.load(open(res_files[-1]))
    out = {}
    for task, (key, name, scale) in PRIMARY.items():
        files = sorted(glob.glob(f"{d}/**/*_samples_{task}.jsonl", recursive=True))
        if task not in res["results"] or not files:
            continue
        rows = [json.loads(l) for l in open(files[-1])]
        n = len(rows)
        if task == "ocrbench":
            val = 100 * sum(r["ocrbench_accuracy"]["score"] for r in rows) / n
        else:
            val = res["results"][task][key] * scale
        rec = {"metric": name, "value": round(val, 2), "n": n}
        if task in MCQ:
            letter_rows = [r for r in rows if re.fullmatch(r"[A-Z]", str(r["target"]).strip())]
            if letter_rows:
                resp = lambda r: r["filtered_resps"] if isinstance(r["filtered_resps"], str) else r["filtered_resps"][0]  # noqa: E731
                ok = [float(r["exact_match"]) > 0 or (re.fullmatch(r"[A-Z]", str(r["target"]).strip()) and lenient(resp(r), r["target"]))
                      for r in rows]
                rec["lenient_letter_acc"] = round(100 * sum(map(bool, ok)) / n, 2)
        out[task] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=dir")
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args()
    runs = {lab: load_run(d) for lab, d in (r.split("=", 1) for r in args.run)}
    labels = list(runs)
    lines = ["| Benchmark | Metric | n | " + " | ".join(labels) + " |", "|---|---|---:|" + "---:|" * len(labels)]
    for task in PRIMARY:
        recs = [runs[l].get(task) for l in labels]
        if not any(recs):
            continue
        r0 = next(r for r in recs if r)
        cells = [f"{r['value']:.1f}" + (f" ({r['lenient_letter_acc']:.1f})" if "lenient_letter_acc" in r else "") if r else "–" for r in recs]
        lines.append(f"| {task} | {r0['metric']} | {r0['n']} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n\nMCQ cells: official metric (lenient letter extraction in parentheses)."
    print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(runs, indent=2))
    if args.md:
        Path(args.md).write_text(md + "\n")


if __name__ == "__main__":
    main()
