"""Export a hybrid training checkpoint as (1) a full Qwen3-VL checkpoint (vLLM-servable) and
(2) the language model alone as a plain Qwen3ForCausalLM (for text-regression evals).

LoRA adapters are merged into the base weights, optionally scaled (retention sweep, roadmap §5.7).

Usage: python scripts/export_hybrid.py --run experiments/<run> [--step N] [--lora-scale 1.0] --out exports/<name>
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LEGACY_INFO = "exports/hybrid-q3-1b7-vit2b/hybrid_info.json"  # pre-chain default (1.7B)


def resolve_base_llm(init: str | None, max_depth: int = 10) -> str:
    """Follow the run's init chain (export_info.json → its init) to the hybrid_info.json written by
    build_hybrid.py and return its `language_model_from`. Falls back to the 1.7B hybrid's info."""
    cur = init
    for _ in range(max_depth):
        if not cur:
            break
        d = ROOT / cur
        if (d / "hybrid_info.json").exists():
            return json.loads((d / "hybrid_info.json").read_text())["language_model_from"]
        if (d / "export_info.json").exists():
            info = json.loads((d / "export_info.json").read_text())
            if info.get("base_llm"):
                return info["base_llm"]
            cur = info.get("init")
            continue
        break
    return json.loads((ROOT / LEGACY_INFO).read_text())["language_model_from"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-llm-only", action="store_true")
    ap.add_argument("--no-serving-prep", action="store_true", help="skip prepare_hybrid_serving.py")
    ap.add_argument("--base-llm", help="text-only LM the hybrid was built from (default: resolved from the init chain)")
    args = ap.parse_args()
    from eval_hybrid import load
    from transformers import AutoConfig, AutoModelForCausalLM

    model, proc, ck, cfg = load(args.run, args.step, None)
    if cfg.get("lora"):
        from peft.tuners.lora import LoraLayer

        lm, n = model.model.language_model, 0
        for name, module in list(lm.named_modules()):
            if isinstance(module, LoraLayer):
                for a in module.scaling:
                    module.scaling[a] *= args.lora_scale
                module.merge()
                parent_name, _, child = name.rpartition(".")
                setattr(lm.get_submodule(parent_name) if parent_name else lm, child, module.get_base_layer())
                n += 1
        print(f"merged LoRA into {n} layers (scale {args.lora_scale})")
    out = ROOT / args.out
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    proc.save_pretrained(out)
    base = args.base_llm or resolve_base_llm(cfg.get("init"))
    print(f"base language model: {base}")
    info = {"run": args.run, "checkpoint": ck, "lora_scale": args.lora_scale, "init": cfg.get("init"), "base_llm": base}
    (out / "export_info.json").write_text(json.dumps(info, indent=2))
    print(f"hybrid → {out}")
    if not args.no_serving_prep:  # VL-aware chat template, pixel cap, generation_config (scripts/prepare_hybrid_serving.py)
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_hybrid_serving.py"), args.out,
                        "--max-pixels", str(cfg.get("max_pixels", 1_600_000)), "--base-lm", base], check=True, cwd=ROOT)

    if not args.no_llm_only:  # language model alone, as plain Qwen3 (text evals in vLLM)
        qcfg = AutoConfig.from_pretrained(base)
        q = AutoModelForCausalLM.from_config(qcfg, dtype=torch.bfloat16)
        sd = {k.replace("model.language_model.", "model.", 1): v for k, v in model.state_dict().items() if "language_model" in k}
        if not qcfg.tie_word_embeddings:
            sd["lm_head.weight"] = model.lm_head.weight
        missing, unexpected = q.load_state_dict(sd, strict=False)
        assert not unexpected and all(m == "lm_head.weight" for m in missing), (missing[:3], unexpected[:3])
        q.tie_weights()
        llm_out = out.parent / (out.name + "-llm")
        q.save_pretrained(llm_out, safe_serialization=True)
        proc.tokenizer.save_pretrained(llm_out)
        print(f"language model → {llm_out}")


if __name__ == "__main__":
    main()
