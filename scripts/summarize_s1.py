"""Summarize S1 runs: final train loss, throughput, and image-reliance gaps per image type.

Usage: python scripts/summarize_s1.py [glob]   (default: experiments/EXP-S1-*)
"""

import glob
import json
import sys
from pathlib import Path

pattern = sys.argv[1] if len(sys.argv) > 1 else "experiments/EXP-S1-*"
rows = []
for run in sorted(p for p in glob.glob(pattern) if Path(p).is_dir()):
    run = Path(run)
    metrics = [json.loads(line) for line in open(run / "metrics.jsonl")] if (run / "metrics.jsonl").exists() else []
    evals = sorted((run / "eval").glob("s1-*.json")) if (run / "eval").exists() else []
    if not metrics or not evals:
        continue
    rel = json.loads(evals[-1].read_text())["reliance"]
    last = metrics[-3:]
    rows.append({
        "run": run.name.replace("EXP-S1-Q3-1B7-S2P16-384-", ""),
        "train_loss": sum(m["loss"] for m in last) / len(last),
        "tok/s": sum(m["tokens_per_s"] for m in last) / len(last),
        **{f"{t}:real": rel[t]["real"] for t in ("photo", "screenshot", "page") if t in rel},
        **{f"{t}:gap_swap": rel[t]["gap_swapped"] for t in ("all", "photo", "screenshot", "page") if t in rel},
        "all:gap_blind": rel["all"]["gap_blind"],
    })

if not rows:
    sys.exit("no evaluated runs")
cols = list(rows[0])
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) and c != "tok/s" else
                            (f"{r[c]:,.0f}" if c == "tok/s" else str(r[c])) for c in cols) + " |")
