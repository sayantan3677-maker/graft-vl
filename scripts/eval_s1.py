"""Evaluate a Stage-1 checkpoint: image-reliance losses + sample generations.

Held-out samples come from a separately built eval manifest; any sample whose
image hash also appears in the training manifest is dropped (contamination guard).

Usage:
    python scripts/eval_s1.py --run experiments/<run_id> [--step 100] --eval-data manifests/s1-eval300 \
        --train-data manifests/s1-5k [--n 200] [--n-generate 12]
"""

import argparse
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import webdataset as wds  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from vlm.data.processor import ProcessorConfig, VLMProcessor  # noqa: E402
from vlm.eval.visual_reliance import evaluate_reliance  # noqa: E402
from vlm.model.multimodal_qwen import QwenVLM, VLMConfig  # noqa: E402


def load_eval_samples(eval_dir: Path, train_dir: Path | None, n: int) -> list[dict]:
    banned_exact, banned_dhash = set(), set()
    if train_dir is not None:
        for line in open(train_dir / "records.jsonl"):
            r = json.loads(line)
            banned_exact.add(r["exact_hash"])
            banned_dhash.add(r["dhash"])
    shards = sorted(str(p) for p in (eval_dir / "shards").glob("*.tar"))
    samples, dropped = [], 0
    for item in wds.WebDataset(shards, shardshuffle=False, empty_check=False):
        rec = json.loads(item["json"])
        if rec["meta"]["exact_hash"] in banned_exact or rec["meta"]["dhash"] in banned_dhash:
            dropped += 1
            continue
        img = Image.open(io.BytesIO(item[next(k for k in item if k.startswith("img."))]))
        img.load()
        samples.append({"messages": rec["messages"], "images": [img], "meta": rec["meta"]})
        if len(samples) >= n:
            break
    print(f"eval samples: {len(samples)} (dropped {dropped} overlapping with train)")
    return samples


def load_model(run_dir: Path, step: int | None, device: str) -> tuple[QwenVLM, dict, str]:
    cfg = json.loads((run_dir / "config.json").read_text())["train"]
    model = QwenVLM.from_pretrained(VLMConfig(**cfg["model"]), dtype=torch.bfloat16).to(device)
    ckpts = sorted((run_dir / "checkpoints").glob("step-*"))
    ckpt = run_dir / "checkpoints" / f"step-{step:07d}" if step is not None else ckpts[-1]
    state = torch.load(ckpt / "trainable.pt", map_location=device)
    missing = [k for k in state if k not in dict(model.named_parameters())]
    assert not missing, f"unknown keys in checkpoint: {missing[:5]}"
    model.load_state_dict(state, strict=False)
    return model.eval(), cfg, ckpt.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--train-data")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-generate", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
    model, cfg, ckpt_name = load_model(run_dir, args.step, args.device)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["llm_name"])
    processor = VLMProcessor(tok, ProcessorConfig(tile_size=model.vision.image_size, **cfg["processor"]))
    samples = load_eval_samples(ROOT / args.eval_data, ROOT / args.train_data if args.train_data else None, args.n)

    reliance = evaluate_reliance(model, processor, samples, args.device)
    print(json.dumps(reliance, indent=2))

    generations = []
    per_type: dict[str, int] = {}
    for s in samples:
        t = s["meta"]["image_type"]
        if per_type.get(t, 0) >= max(1, args.n_generate // 3):
            continue
        per_type[t] = per_type.get(t, 0) + 1
        p = processor.prompt(s["messages"][:-1], s["images"], enable_thinking=False)
        pv = p["pixel_values"].to(args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            out_ids = model.generate(p["input_ids"][None].to(args.device), pv, max_new_tokens=200)
        generations.append({
            "key": s["meta"]["key"], "image_type": t,
            "prompt": s["messages"][0]["content"][-1]["text"],
            "reference": s["messages"][-1]["content"][:600],
            "generated": tok.decode(out_ids, skip_special_tokens=True)[:600],
        })
    for g in generations:
        print(f"\n[{g['image_type']}] {g['key']}\n  REF: {g['reference'][:200]}\n  GEN: {g['generated'][:200]}")

    out = run_dir / "eval" / f"s1-{ckpt_name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"checkpoint": ckpt_name, "eval_data": args.eval_data, "reliance": reliance,
                               "generations": generations}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
