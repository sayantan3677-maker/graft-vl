"""Minimal reading probe: one large word per image, single tile. Can the model read at all?

Usage: python scripts/ocr_probe.py --run experiments/<run> [--step N] [--n 60]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

WORDS = ["BANANA", "GARDEN", "ROCKET", "PLANET", "SILVER", "MARKET", "WINDOW", "CASTLE", "BUTTER", "FOREST",
         "ORANGE", "PENCIL", "TICKET", "SUMMER", "DRAGON", "COFFEE", "BRIDGE", "CAMERA", "ISLAND", "LETTER",
         "PURPLE", "TURTLE", "HAMMER", "JUNGLE", "MIRROR", "PILLOW", "SHADOW", "TOMATO", "VIOLIN", "WALNUT"]


def render(word: str, size: int, font_px: int, rng: random.Random) -> Image.Image:
    bg = rng.choice([(255, 255, 255), (245, 240, 225), (230, 240, 255)])
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_px)
    w = d.textlength(word, font=font)
    d.text(((size - w) / 2, (size - font_px) / 2), word, fill=(0, 0, 0), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from eval_s1 import load_model
    from vlm.data.processor import ProcessorConfig, VLMProcessor

    model, cfg, ckpt = load_model(ROOT / args.run, args.step, "cuda")
    tok = AutoTokenizer.from_pretrained(cfg["model"]["llm_name"])
    proc = VLMProcessor(tok, ProcessorConfig(tile_size=model.vision.image_size, **cfg["processor"]))
    rng = random.Random(0)
    results = {}
    for font_px in (80, 40, 20):
        correct, examples = 0, []
        for i in range(args.n):
            word = WORDS[i % len(WORDS)]
            img = render(word, model.vision.image_size, font_px, rng)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What word is written in this image? Answer with the word only."}]}]
            p = proc.prompt(msgs, [img], enable_thinking=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = model.generate(p["input_ids"][None].cuda(), p["pixel_values"].cuda(), max_new_tokens=12)
            ans = tok.decode(ids, skip_special_tokens=True).strip()
            correct += word.lower() in ans.lower()
            if len(examples) < 4:
                examples.append((word, ans))
        results[font_px] = {"acc": round(100 * correct / args.n, 1), "examples": examples}
        print(f"font {font_px:3d}px: {results[font_px]['acc']:5.1f}%  e.g. {examples}", flush=True)
    out = ROOT / args.run / "eval" / f"ocr_probe-{ckpt}.json"
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
