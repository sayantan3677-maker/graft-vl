"""Collect scaling-4b results (4B hybrid vs 1.7B hybrid) into reports/agents/scaling-4b.json.

Usage: python scripts/report_scaling4b.py
"""

import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports/agents/scaling-4b"


def qa(path):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"probe": d["probe"], "qa_overall": d["qa_overall"], "qa": d["qa"], "n": len(d.get("rows", []))}


def run_evals(run):
    return {Path(f).stem.replace("hybrid-eval-", ""): qa(f) for f in sorted(glob.glob(str(ROOT / f"experiments/{run}/eval/hybrid-eval-step-*.json")))}


def timing(run):
    p = ROOT / f"experiments/{run}/metrics.jsonl"
    if not p.exists():
        return None
    m = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    st = [r["step_time_s"] for r in m]
    return {"steps": m[-1]["step"], "median_step_time_s": statistics.median(st), "mean_step_time_s": statistics.mean(st),
            "train_gpu_hours": sum(r["step_time_s"] * 25 for r in m) / 3600, "peak_alloc_gb": max(r["peak_alloc_gb"] for r in m),
            "final_loss_avg_last4": statistics.mean(r["loss"] for r in m[-4:]),
            "tokens_per_s_median": statistics.median(r["tokens_per_s"] for r in m)}


def text(path):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())["results"]
    return {k: {"acc_mean": v["acc_mean"], "acc_std": v["acc_std"], "n_items": v["n_items"], "seeds": v["seeds"],
                "minutes": v.get("minutes")} for k, v in d.items()}


def main():
    out = {
        "equivalence_fp32": json.loads((R / "text-equivalence-fp32.json").read_text()) if (R / "text-equivalence-fp32.json").exists() else None,
        "zero_shot": {"4b": qa(R / "hybrid-eval-4b-init.json"), "1.7b": qa(ROOT / "reports/hybrid-eval-init.json")},
        "align": {"4b": run_evals("HYB-ALIGN-Q3-4B-VIT4B-MERGER-1500"), "1.7b": run_evals("HYB-ALIGN-Q3-1B7-VIT2B-MERGER-1500")},
        "s2": {"4b": run_evals("HYB-S2-Q3-4B-LORA32-2000"), "1.7b": run_evals("HYB-S2-Q3-1B7-LORA32-3000")},
        "full1400": {Path(f).stem: qa(f) for f in sorted(glob.glob(str(R / "s2-*-full1400.json")))},
        "timing": {r: timing(r) for r in ["HYB-ALIGN-Q3-4B-VIT4B-MERGER-1500", "HYB-ALIGN-Q3-1B7-VIT2B-MERGER-1500",
                                          "HYB-S2-Q3-4B-LORA32-2000", "HYB-S2-Q3-1B7-LORA32-3000"]},
        "text": {Path(f).stem: text(f) for f in sorted(glob.glob(str(R / "text-*.json"))) if not f.endswith(".items.json")
                 and "equivalence" not in f},
        "replay": {n: json.loads((ROOT / f"manifests/{n}/manifest.json").read_text())
                   for n in ["replay-q3-4b-16k", "replay-q3-4b-ifsafety-4k", "replay-q3-1b7-20k", "replay-q3-1b7-ifsafety-5k"]
                   if (ROOT / f"manifests/{n}/manifest.json").exists()},
    }
    base = out["text"].get("text-qwen3-4b-original")
    if base:
        out["text_deltas"] = {}
        for name, res in out["text"].items():
            if name == "text-qwen3-4b-original" or not res:
                continue
            out["text_deltas"][name] = {k: {"delta": res[k]["acc_mean"] - base[k]["acc_mean"],
                                            "base_std": base[k]["acc_std"], "std": res[k]["acc_std"]}
                                        for k in res if k in base}
    (ROOT / "reports/agents/scaling-4b.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
