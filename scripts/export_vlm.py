"""Export a VLM checkpoint as a self-contained serving checkpoint for vLLM (roadmap §12.3).

    exports/<name>/
      config.json            model_type qwen_siglip_vlm (text_config, truncated vision_config, tiling)
      model.safetensors      language_model.* | vision_tower.vision_model.* | projector.*
      tokenizer + chat template + generation_config.json

Special-row deltas are merged; the vision tower is stored truncated at the feature layer.

Usage: python scripts/export_vlm.py --run experiments/<run_id> [--step N] --out exports/<name>
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safetensors.torch import save_file  # noqa: E402
from transformers import AutoTokenizer, GenerationConfig  # noqa: E402

from vlm.model.multimodal_qwen import QwenVLM, VLMConfig  # noqa: E402
from vlm.serving.config import ARCHITECTURE, QwenSiglipVLMConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-scale", type=float, default=1.0, help="LoRA strength for retention sweeps (roadmap §5.7)")
    ap.add_argument("--max-tiles", type=int)
    args = ap.parse_args()

    run_dir = ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())["train"]
    ckpts = sorted((run_dir / "checkpoints").glob("step-*"))
    ckpt = run_dir / "checkpoints" / f"step-{args.step:07d}" if args.step else ckpts[-1]

    vcfg = VLMConfig(**cfg["model"])
    model = QwenVLM.from_pretrained(vcfg, dtype=torch.float32)
    state = torch.load(ckpt / "trainable.pt", map_location="cpu")
    unknown = [k for k in state if k not in dict(model.named_parameters())]
    assert not unknown, f"unknown keys: {unknown[:5]}"
    model.load_state_dict(state, strict=False)
    if model.config.lora:
        print(f"merged LoRA in {model.merge_lora(args.lora_scale)} layers (scale {args.lora_scale})")
    model.merge_special_rows()

    vision_cfg = model.vision.model.config.to_dict()
    vision_cfg["num_hidden_layers"] = len(model.vision.tower.encoder.layers)  # truncated at the feature layer
    vision_cfg["vision_use_head"] = False
    processor_cfg = cfg.get("processor", {})
    hf_cfg = QwenSiglipVLMConfig(
        text_config=model.llm.config.to_dict(), vision_config=vision_cfg, image_pad_id=vcfg.image_pad_id,
        reduce_factor=vcfg.reduce_factor, projector_layernorm=vcfg.projector_layernorm,
        tile_size=model.vision.image_size, max_tiles=args.max_tiles or processor_cfg.get("max_tiles", 6), thumbnail=True,
        architectures=[ARCHITECTURE], torch_dtype="bfloat16",
    )

    tensors = {}
    for k, v in model.llm.state_dict().items():
        if hf_cfg.text_config.tie_word_embeddings and k == "lm_head.weight":
            continue
        tensors[f"language_model.{k}"] = v
    for k, v in model.vision.tower.state_dict().items():
        if k.startswith(("post_layernorm", "head")):
            continue
        tensors[f"vision_tower.vision_model.{k}"] = v
    for k, v in model.projector.state_dict().items():
        tensors[f"projector.{k}"] = v
    tensors = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in tensors.items()}

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    hf_cfg.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(vcfg.llm_name)
    tok.save_pretrained(out)
    GenerationConfig.from_pretrained(vcfg.llm_name).save_pretrained(out)
    # Non-thinking variant of the chat template for benchmark/eval clients that can't pass enable_thinking.
    template = (out / "chat_template.jinja").read_text()
    (out / "chat_template_nothink.jinja").write_text("{%- set enable_thinking = false %}\n" + template)
    (out / "export_info.json").write_text(json.dumps({"run": str(run_dir), "checkpoint": ckpt.name,
                                                      "n_tensors": len(tensors), "train_config": cfg}, indent=2))
    print(f"exported {ckpt.name} → {out} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
