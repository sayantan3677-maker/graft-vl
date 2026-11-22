"""Held-out short-answer QA accuracy per source and image type (roadmap §7.3, §7.6).

Generates a short (non-thinking) answer, or with --thinking a thinking-mode answer, for each
held-out question and scores it against the ground truth. Samples whose image also appears
in any given training manifest are dropped.

Usage:
    python scripts/eval_vqa.py --run experiments/<run_id> [--step N] --eval-data manifests/qa-eval-1500 \
        --train-data manifests/s2-mm-300k manifests/s1-50k [--thinking] [--n 1500]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_s1 import load_eval_samples, load_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from vlm.data.processor import ProcessorConfig, VLMProcessor  # noqa: E402
from vlm.eval.answer_match import RELAXED_NUMERIC, answers_match  # noqa: E402

SHORT = "\nAnswer the question using a single word or phrase."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--train-data", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
    model, cfg, ckpt_name = load_model(run_dir, args.step, args.device)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["llm_name"])
    processor = VLMProcessor(tok, ProcessorConfig(tile_size=model.vision.image_size, **cfg["processor"]))

    samples = load_eval_samples(ROOT / args.eval_data, None, args.n)
    if args.train_data:
        banned = set()
        for d in args.train_data:
            for line in open(ROOT / d / "records.jsonl"):
                r = json.loads(line)
                banned.add(r.get("exact_hash"))
        before = len(samples)
        samples = [s for s in samples if s["meta"]["exact_hash"] not in banned]
        print(f"dropped {before - len(samples)} samples overlapping training data")

    max_new = 4096 if args.thinking else args.max_new_tokens
    per_source, per_type, rows = defaultdict(list), defaultdict(list), []
    for s in samples:
        question = s["messages"][0]["content"][-1]["text"] + ("" if args.thinking else SHORT)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        p = processor.prompt(msgs, s["images"], enable_thinking=args.thinking)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            ids = model.generate(p["input_ids"][None].to(args.device), p["pixel_values"].to(args.device), max_new_tokens=max_new)
        text = tok.decode(ids, skip_special_tokens=True)
        answer = text.split("</think>")[-1].strip()
        gold = s["messages"][-1]["content"]
        src = s["meta"]["source"]
        golds = s["meta"].get("golds") or [gold]  # official splits carry all reference answers
        relaxed = src in RELAXED_NUMERIC or src.startswith(("chartqa", "infovqa"))
        ok = any(answers_match(answer, g, relaxed=relaxed) for g in golds)
        per_source[src].append(ok)
        per_type[s["meta"]["image_type"]].append(ok)
        rows.append({"key": s["meta"]["key"], "source": src, "question": question, "gold": gold, "pred": answer[:300], "correct": ok})

    acc = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 2)
    result = {
        "checkpoint": ckpt_name, "thinking": args.thinking, "n": len(rows),
        "overall": acc([r["correct"] for r in rows]),
        "by_image_type": {k: {"acc": acc(v), "n": len(v)} for k, v in sorted(per_type.items())},
        "by_source": {k: {"acc": acc(v), "n": len(v)} for k, v in sorted(per_source.items())},
    }
    print(json.dumps(result, indent=2))
    out = run_dir / "eval" / f"vqa-{'think' if args.thinking else 'short'}-{ckpt_name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({**result, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
