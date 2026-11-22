"""Build the real Qwen3 + SigLIP2 VLM and run one forward/backward (CPU or GPU).

Verifies real checkpoint loading (no missing vision weights), tiling → token
counts, the S1 trainable set, and a finite loss with gradients on the projector.

Usage: python scripts/smoke_real_model.py [--device cpu] [--llm Qwen/Qwen3-1.7B]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer, SiglipVisionModel  # noqa: E402

from vlm.data.processor import ProcessorConfig, VLMProcessor  # noqa: E402
from vlm.model.multimodal_qwen import QwenVLM, VLMConfig  # noqa: E402


def test_image() -> Image.Image:
    img = Image.new("RGB", (1280, 720), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 600, 300], outline="black", width=4)
    d.text((60, 60), "Sign in", fill="black")
    d.ellipse([800, 200, 1100, 500], fill="red")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--vision", default="google/siglip2-so400m-patch16-384")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16

    # Loading report: every vision weight we keep must come from the checkpoint.
    _, info = SiglipVisionModel.from_pretrained(args.vision, output_loading_info=True)
    print("vision missing keys:", info["missing_keys"])
    assert not info["missing_keys"], "SigLIP2 vision weights did not load"

    cfg = VLMConfig(llm_name=args.llm, vision_name=args.vision)
    t0 = time.time()
    model = QwenVLM.from_pretrained(cfg, dtype=dtype).to(args.device)
    model.set_trainable()
    print(f"loaded in {time.time() - t0:.1f}s; tokens/tile={model.tokens_per_tile}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train / 1e6:.2f}M / total {n_total / 1e6:.1f}M")

    tok = AutoTokenizer.from_pretrained(args.llm)
    proc = VLMProcessor(tok, ProcessorConfig(tile_size=model.vision.image_size, max_tiles=6))
    sample = proc({
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this screenshot."}]},
            {"role": "assistant", "content": "A sign-in box on the left and a red circle on the right."},
        ],
        "images": [test_image()],
    })
    print(f"seq len {len(sample['input_ids'])}, tiles {sample['pixel_values'].shape[0]}, image tokens {sample['n_image_tokens']}")

    t0 = time.time()
    out = model(sample["input_ids"][None].to(args.device), labels=sample["labels"][None].to(args.device),
                pixel_values=sample["pixel_values"].to(args.device))
    out["loss"].backward()
    print(f"loss {out['loss'].item():.3f} (fwd+bwd {time.time() - t0:.1f}s)")
    assert torch.isfinite(out["loss"])
    g = model.projector.fc1.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert all(p.grad is None for p in model.llm.parameters())
    print("OK")


if __name__ == "__main__":
    main()
