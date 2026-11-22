"""Evaluate a hybrid checkpoint: single-word reading probe + official-split QA (short answers).

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/eval_hybrid.py --run experiments/<run> [--step N] [--n-qa 60]
       python scripts/eval_hybrid.py --init exports/hybrid-q3-1b7-vit2b   # untrained hybrid
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def load(run: str | None, step: int | None, init: str | None):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    cfg, ckpt = ({}, None)
    if run:
        run_dir = ROOT / run
        cfg = json.loads((run_dir / "config.json").read_text())
        cks = sorted((run_dir / "checkpoints").glob("step-*"))
        ckpt = run_dir / "checkpoints" / f"step-{step:07d}" if step else cks[-1]
        init = cfg["init"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(ROOT / init, dtype=torch.bfloat16)
    if cfg.get("lora"):
        from peft import LoraConfig, inject_adapter_in_model

        inject_adapter_in_model(LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=0.0,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                                model.model.language_model)
    if ckpt:
        state = torch.load(ckpt / "trainable.pt", map_location="cpu")
        params = dict(model.named_parameters())
        unknown = [k for k in state if k not in params]
        assert not unknown, unknown[:5]
        with torch.no_grad():
            for k, v in state.items():
                params[k].copy_(v.to(params[k].dtype))
    proc = AutoProcessor.from_pretrained(ROOT / init)
    return model.cuda().eval(), proc, (ckpt.name if ckpt else "init"), cfg


@torch.no_grad()
def generate(model, proc, image, question, max_pixels, max_new=32, thinking=False):
    tok = proc.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>" + question}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    inputs = proc(text=[text], images=[image], size={"longest_edge": max_pixels, "shortest_edge": 65536}, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).split("</think>")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--step", type=int)
    ap.add_argument("--init")
    ap.add_argument("--n-probe", type=int, default=30)
    ap.add_argument("--n-qa", type=int, default=60)
    ap.add_argument("--max-pixels", type=int, default=1_200_000)
    ap.add_argument("--out", help="output JSON path (default: <run>/eval/hybrid-eval-<ckpt>.json or reports/hybrid-eval-init.json)")
    args = ap.parse_args()
    from eval_s1 import load_eval_samples
    from ocr_probe import WORDS, render
    from vlm.eval.answer_match import answers_match

    model, proc, ck, cfg = load(args.run, args.step, args.init)
    rng = random.Random(0)
    probe = {}
    for font_px in (80, 40, 20):
        ok = sum(WORDS[i % len(WORDS)].lower() in generate(model, proc, render(WORDS[i % len(WORDS)], 384, font_px, rng),
                 "What word is written in this image? Answer with the word only.", args.max_pixels).lower()
                 for i in range(args.n_probe))
        probe[font_px] = round(100 * ok / args.n_probe, 1)
    samples = load_eval_samples(ROOT / "manifests/eval-official", None, 1400)
    by = {}
    for s in samples:
        by.setdefault(s["meta"]["source"], []).append(s)
    per, rows = {}, []
    for src in sorted(by):
        for s in by[src][: args.n_qa]:
            q = s["messages"][0]["content"][-1]["text"] + "\nAnswer the question using a single word or phrase."
            ans = generate(model, proc, s["images"][0], q, args.max_pixels)
            ok = any(answers_match(ans, g, relaxed=src.startswith(("chartqa", "infovqa"))) for g in s["meta"]["golds"])
            per.setdefault(src, []).append(ok)
            rows.append({"source": src, "q": q[:200], "golds": s["meta"]["golds"][:3], "pred": ans[:200], "ok": ok})
    qa = {k: round(100 * sum(v) / len(v), 1) for k, v in per.items()}
    overall = round(100 * sum(r["ok"] for r in rows) / len(rows), 1)
    print(json.dumps({"checkpoint": ck, "probe": probe, "qa_overall": overall, "qa": qa}))
    out_dir = (ROOT / args.run / "eval") if args.run else (ROOT / "reports")
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"hybrid-eval-{ck}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"probe": probe, "qa_overall": overall, "qa": qa, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
