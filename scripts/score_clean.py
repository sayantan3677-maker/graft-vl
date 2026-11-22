"""Re-score eval_hybrid_export.py qa JSONs on all items and on the contamination-free subset, plus answer-format habits.

Usage: python scripts/score_clean.py [--bad experiments/s3-eval/contaminated_keys.json] name=path.qa.json ...
"""
import argparse
import json
import re
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bad", default="experiments/s3-eval/contaminated_keys.json")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    bad = set(json.load(open(args.bad)))
    print(f"{'model':28s} {'all':>6s} {'clean':>6s}  per-source clean  | ends '.'  'Answer:'")
    for spec in args.runs:
        name, path = spec.split("=", 1)
        rows = json.load(open(path))["rows"]
        per = defaultdict(list)
        for r in rows:
            if r["key"] not in bad:
                per[r["source"]].append(r["ok"])
        clean = [ok for v in per.values() for ok in v]
        short = [r["pred"].strip() for r in rows if len(r["pred"].split()) <= 4]
        dots = sum(p.endswith(".") and not re.search(r"\b[A-Za-z]\.$", p) for p in short) / max(1, len(short))
        pre = sum(r["pred"].strip().startswith("Answer:") for r in rows) / len(rows)
        ps = " ".join(f"{k.split('_')[0][:6]}={100 * sum(v) / len(v):.1f}" for k, v in sorted(per.items()))
        print(f"{name:28s} {100 * sum(r['ok'] for r in rows) / len(rows):6.1f} {100 * sum(clean) / len(clean):6.1f}  {ps} | "
              f"{100 * dots:5.1f}%  {100 * pre:5.1f}%")


if __name__ == "__main__":
    main()
