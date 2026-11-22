"""Export the language model of a VLM checkpoint as a plain Qwen3ForCausalLM (roadmap §7.1).

Special-row deltas are merged into the embedding (and LM head when untied), so the
exported model behaves exactly like the VLM on text-only input and loads in vLLM.

Usage:
    python scripts/export_llm.py --run experiments/<run_id> [--step N] --out exports/<name>-llm
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from vlm.model.multimodal_qwen import QwenVLM, VLMConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-scale", type=float, default=1.0, help="LoRA strength for retention sweeps (roadmap §5.7)")
    args = ap.parse_args()

    run_dir = ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())["train"]
    ckpts = sorted((run_dir / "checkpoints").glob("step-*"))
    ckpt = run_dir / "checkpoints" / f"step-{args.step:07d}" if args.step else ckpts[-1]

    model = QwenVLM.from_pretrained(VLMConfig(**cfg["model"]), dtype=torch.float32)
    state = torch.load(ckpt / "trainable.pt", map_location="cpu")
    unknown = [k for k in state if k not in dict(model.named_parameters())]
    assert not unknown, f"unknown keys: {unknown[:5]}"
    model.load_state_dict(state, strict=False)
    if model.config.lora:
        print(f"merged LoRA in {model.merge_lora(args.lora_scale)} layers (scale {args.lora_scale})")
    model.merge_special_rows()

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    model.llm.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg["model"]["llm_name"]).save_pretrained(out)
    (out / "vlm_export.json").write_text(json.dumps({"run": str(run_dir), "checkpoint": ckpt.name,
                                                     "llm_trained": bool(cfg.get("train_llm"))}, indent=2))
    print(f"exported {ckpt.name} → {out}")


if __name__ == "__main__":
    main()
