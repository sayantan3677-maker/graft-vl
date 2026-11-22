"""Configs for the scaling ladder (roadmap §6, §8): per-size S1, and matched-budget LoRA S2
(family A: identical data, tokens and adapter settings at 1.7B / 4B / 8B).

Usage: python scripts/make_scaling_configs.py [--s2-steps 3300]
"""

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"1b7": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B", "8b": "Qwen/Qwen3-8B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-steps", type=int, default=3300, help="≈0.1B tokens at ~30K tokens/step")
    args = ap.parse_args()
    s1 = json.loads((ROOT / "configs/train/s1_50k_lr1e-3.json").read_text())
    s2 = json.loads((ROOT / "configs/train/s2_q3-1b7_full.json").read_text())
    for tag, llm in SIZES.items():
        s1_run = f"EXP-S1-Q3-{tag.upper()}-S2P16-384-MLP-50K-LR1e-3"
        if tag != "1b7":  # 1.7B S1 already exists
            c = copy.deepcopy(s1)
            c["run_id"] = s1_run
            c["model"]["llm_name"] = llm
            c["gradient_checkpointing"] = tag == "8b"
            (ROOT / f"configs/train/s1_q3-{tag}.json").write_text(json.dumps(c, indent=2))
        c = copy.deepcopy(s2)
        c["run_id"] = f"EXP-S2A-Q3-{tag.upper()}-384-LORA32-MATCHED"
        c["init_from"] = f"experiments/{s1_run}/checkpoints/step-0001700"
        c["model"]["llm_name"] = llm
        c["model"]["lora"] = {"r": 32, "alpha": 64, "dropout": 0.05}
        c.update(train_llm=True, llm_dtype="bfloat16", lr_llm=1.5e-4, lr_projector=5e-5, lr_special_rows=5e-5,
                 max_steps=args.s2_steps, save_every=max(1, args.s2_steps // 3), gradient_checkpointing=True)
        (ROOT / f"configs/train/s2a_lora_q3-{tag}.json").write_text(json.dumps(c, indent=2))
    print("wrote", sorted(p.name for p in (ROOT / "configs/train").glob("s[12]*q3-[148]*.json")))


if __name__ == "__main__":
    main()
