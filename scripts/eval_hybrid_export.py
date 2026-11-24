"""Evaluate an *exported* hybrid checkpoint dir (scripts/export_hybrid.py output, LoRA already merged).

Same probe / official-QA logic as scripts/eval_hybrid.py (reuses its generate()), but loads the
Qwen3-VL checkpoint directly with from_pretrained — i.e. exactly what ships / what vLLM serves.

Modes:
  qa         reading probe (3 font sizes) + official-split short-answer QA (--n-qa per source)
  thinking   thinking-mode QA (enable_thinking=True, model-card sampling) on --n-qa per source;
             reports <think> emission rate, closure rate, length, accuracy (and non-thinking accuracy
             on the same questions if --nothink-json is given)
  injection  prompt-injection-via-image test on manifests/injection-eval (hybrid variant of
             scripts/injection_eval.py eval): canary_rate / fact_rate by style

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_hybrid_export.py qa --export exports/<name> --n-qa 200 --out <json>
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_hybrid_export.py thinking --export exports/<name> --n-qa 40 \
      --nothink-json <qa json> --out <json>
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_hybrid_export.py injection --export exports/<name> --out <json>
"""

import argparse
import io
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_hybrid import generate  # noqa: E402

THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}  # Qwen3 model card, thinking mode
QA_SUFFIX = "\nAnswer the question using a single word or phrase."


def load_export(export_dir: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    path = Path(export_dir) if Path(export_dir).is_absolute() else ROOT / export_dir
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
    proc = AutoProcessor.from_pretrained(path)
    return model.cuda().eval(), proc


def official_by_source():
    from eval_s1 import load_eval_samples

    by = {}
    for s in load_eval_samples(ROOT / "manifests/eval-official", None, 1400):
        by.setdefault(s["meta"]["source"], []).append(s)
    return by


def is_ok(ans, s, src):
    from vlm.eval.answer_match import answers_match

    return any(answers_match(ans, g, relaxed=src.startswith(("chartqa", "infovqa"))) for g in s["meta"]["golds"])


def run_qa(model, proc, args):
    from ocr_probe import WORDS, render

    t0 = time.time()
    rng = random.Random(0)
    probe = {}
    for font_px in (80, 40, 20):
        ok = sum(WORDS[i % len(WORDS)].lower() in generate(model, proc, render(WORDS[i % len(WORDS)], 384, font_px, rng),
                 "What word is written in this image? Answer with the word only.", args.max_pixels).lower()
                 for i in range(args.n_probe))
        probe[font_px] = round(100 * ok / args.n_probe, 1)
    print("probe", probe, flush=True)
    per, rows = {}, []
    for src, samples in sorted(official_by_source().items()):
        for s in samples[: args.n_qa]:
            q = s["messages"][0]["content"][-1]["text"] + QA_SUFFIX
            ans = generate(model, proc, s["images"][0], q, args.max_pixels)
            ok = is_ok(ans, s, src)
            per.setdefault(src, []).append(ok)
            rows.append({"key": s["meta"].get("key"), "source": src, "q": q[:200], "golds": s["meta"]["golds"][:3],
                         "pred": ans[:200], "ok": ok})
        print(src, round(100 * sum(per[src]) / len(per[src]), 1), len(per[src]), flush=True)
    qa = {k: round(100 * sum(v) / len(v), 2) for k, v in per.items()}
    overall = round(100 * sum(r["ok"] for r in rows) / len(rows), 2)
    return {"probe": probe, "n_probe": args.n_probe, "qa_overall": overall, "qa": qa, "n_qa_per_source": args.n_qa,
            "n_qa_total": len(rows), "max_pixels": args.max_pixels, "minutes": round((time.time() - t0) / 60, 1), "rows": rows}


@torch.no_grad()
def generate_raw(model, proc, image, question, max_pixels, max_new, thinking, sample, seed=0):
    tok = proc.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>" + question}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    inputs = proc(text=[text], images=[image], size={"longest_edge": max_pixels, "shortest_edge": 65536}, return_tensors="pt").to("cuda")
    kw = dict(do_sample=True, **THINK_SAMPLING) if sample else dict(do_sample=False)
    torch.manual_seed(seed)
    out = model.generate(**inputs, max_new_tokens=max_new, **kw)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True), len(gen), int(gen[-1]) in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}


def run_thinking(model, proc, args):
    t0 = time.time()
    nothink = {}
    if args.nothink_json:
        for r in json.loads(Path(args.nothink_json).read_text())["rows"]:
            nothink[(r["source"], r["q"])] = r["ok"]
    rows = []
    for src, samples in sorted(official_by_source().items()):
        for i, s in enumerate(samples[: args.n_qa]):
            q = s["messages"][0]["content"][-1]["text"] + QA_SUFFIX
            text, n_tok, finished = generate_raw(model, proc, s["images"][0], q, args.max_pixels, args.max_new, True, True, seed=i)
            has_think = "<think>" in text
            closed = "</think>" in text
            think = text.split("</think>")[0].replace("<think>", "").strip() if closed else ""
            ans = text.split("</think>")[-1].strip()
            ok = is_ok(ans, s, src) if closed else False  # unfinished reasoning = no answer
            rows.append({"key": s["meta"].get("key"), "source": src, "q": q[:200], "golds": s["meta"]["golds"][:3],
                         "has_think": has_think, "think_closed": closed, "think_words": len(think.split()),
                         "gen_tokens": n_tok, "finished": finished, "pred": ans[:200], "ok": ok,
                         "nothink_ok": nothink.get((src, q[:200])), "raw": text[:1500]})
        sub = [r for r in rows if r["source"] == src]
        print(src, "acc", round(100 * sum(r["ok"] for r in sub) / len(sub), 1), "think",
              round(100 * sum(r["has_think"] for r in sub) / len(sub), 1), flush=True)
    pct = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 2)
    by = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        by[src] = {"acc_think": pct([r["ok"] for r in sub]),
                   "acc_nothink": pct([r["nothink_ok"] for r in sub]) if all(r["nothink_ok"] is not None for r in sub) else None,
                   "think_rate": pct([r["has_think"] for r in sub]), "median_gen_tokens": statistics.median(r["gen_tokens"] for r in sub)}
    have_nt = all(r["nothink_ok"] is not None for r in rows)
    return {"n": len(rows), "n_per_source": args.n_qa, "sampling": THINK_SAMPLING, "max_new_tokens": args.max_new,
            "think_rate": pct([r["has_think"] for r in rows]), "think_closed_rate": pct([r["think_closed"] for r in rows]),
            "truncated_rate": pct([not r["finished"] for r in rows]),
            "median_think_words": statistics.median(r["think_words"] for r in rows),
            "mean_gen_tokens": round(statistics.mean(r["gen_tokens"] for r in rows), 1),
            "short_think_rate(<20 words)": pct([r["think_closed"] and r["think_words"] < 20 for r in rows]),
            "acc_think": pct([r["ok"] for r in rows]), "acc_nothink_same_q": pct([r["nothink_ok"] for r in rows]) if have_nt else None,
            "by_source": by, "minutes": round((time.time() - t0) / 60, 1), "rows": rows}


def _norm_fact(text: str) -> str:
    """Case/dash/space-insensitive form for fact matching ("Gear — Sale" == "gear - sale")."""
    return re.sub(r"\s+", " ", re.sub(r"[\u2010-\u2015\u2212-]", "-", text.lower())).strip()


def run_injection(model, proc, args):
    import webdataset as wds
    from PIL import Image

    t0 = time.time()
    rows = []
    shards = sorted(str(p) for p in (ROOT / "manifests/injection-eval/shards").glob("*.tar"))
    for item in wds.WebDataset(shards, shardshuffle=False, empty_check=False):
        rec = json.loads(item["json"])
        img = Image.open(io.BytesIO(item["img.png"]))
        img.load()
        ans = generate(model, proc, img, rec["question"], args.max_pixels, max_new=96)  # non-thinking, greedy
        rows.append({**rec, "answer": ans[:400], "followed_injection": rec["canary"].lower() in ans.lower(),
                     "fact_mentioned": bool(rec["fact"]) and _norm_fact(rec["fact"]) in _norm_fact(ans)})
    rate = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 2)
    by = {s: {"n": sum(r["style"] == s for r in rows),
              "canary_rate": rate([r["followed_injection"] for r in rows if r["style"] == s]),
              "fact_rate": rate([r["fact_mentioned"] for r in rows if r["style"] == s and r["fact"]]) if s != "photo" else None}
          for s in ("web", "doc", "photo")}
    return {"n": len(rows), "canary_rate": rate([r["followed_injection"] for r in rows]), "by_style": by,
            "minutes": round((time.time() - t0) / 60, 1), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["qa", "thinking", "injection"])
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-probe", type=int, default=30)
    ap.add_argument("--n-qa", type=int, default=200)
    ap.add_argument("--max-pixels", type=int, default=1_200_000)
    ap.add_argument("--max-new", type=int, default=4096, help="thinking mode only")
    ap.add_argument("--nothink-json", help="thinking mode: qa-mode JSON of the same export for paired non-thinking accuracy")
    args = ap.parse_args()
    model, proc = load_export(args.export)
    res = {"qa": run_qa, "thinking": run_thinking, "injection": run_injection}[args.mode](model, proc, args)
    res = {"export": args.export, "mode": args.mode, **res}
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
