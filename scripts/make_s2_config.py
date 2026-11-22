"""Write the S2 training config's `data` list from completed per-source manifests.

Multimodal sources get weight = s2.json share × MM_SHARE (renormalized over the sources that
exist); verified traces and self-distilled replay get fixed shares of the whole stream.

Usage: python scripts/make_s2_config.py --base configs/train/s2_q3-1b7_full.json \
          --replay manifests/replay-q3-1b7-20k --traces manifests/traces-q3-1b7 [--mm 0.70 --trace-share 0.10 --replay-share 0.20]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--replay", required=True, nargs="+", help="replay manifests as path[:weight]; default weights split --replay-share evenly")
    ap.add_argument("--traces")
    ap.add_argument("--mm", type=float, default=0.70)
    ap.add_argument("--mixture", default="configs/data/s2.json")
    ap.add_argument("--trace-share", type=float, default=0.10)
    ap.add_argument("--replay-share", type=float, default=0.20)
    args = ap.parse_args()

    weights = json.loads((ROOT / args.mixture).read_text())["weights"]
    present = {k: w for k, w in weights.items() if (ROOT / "manifests/s2" / k / "manifest.json").exists()}
    missing = sorted(set(weights) - set(present))
    traces_ok = args.traces and (ROOT / args.traces / "manifest.json").exists()
    mm_share = args.mm + (0 if traces_ok else args.trace_share)  # no traces yet → give their share to multimodal
    total = sum(present.values())
    data = [{"path": f"manifests/s2/{k}", "weight": round(mm_share * w / total, 5)} for k, w in present.items()]
    if traces_ok:
        data.append({"path": args.traces, "weight": args.trace_share})
    specs = [r.split(":") for r in args.replay]
    explicit = [float(x[1]) for x in specs if len(x) == 2]
    for spec in specs:
        w = float(spec[1]) if len(spec) == 2 else (args.replay_share - sum(explicit)) / sum(len(x) == 1 for x in specs)
        if not (ROOT / spec[0] / "manifest.json").exists():
            raise FileNotFoundError(spec[0])
        data.append({"path": spec[0], "weight": round(w, 5)})

    cfg = json.loads((ROOT / args.base).read_text())
    cfg["data"] = data
    (ROOT / args.base).write_text(json.dumps(cfg, indent=2))
    counts = {k: json.loads((ROOT / "manifests/s2" / k / "manifest.json").read_text())["written"] for k in present}
    print(f"{len(present)} multimodal sources ({sum(counts.values()):,} samples), missing: {missing}")
    print(f"traces: {'yes' if traces_ok else 'NO (share moved to multimodal)'}; replay share {args.replay_share}")


if __name__ == "__main__":
    main()
