"""Verified multimodal thinking traces (roadmap §4.7). Runs in `vlm-eval`, two phases.

  describe: a captioning VLM writes an exhaustive description (incl. all visible text,
            table cells, chart values) of each image from short-answer QA sources.
  reason:   the ORIGINAL text-only Qwen3 (student size) answers the question from that
            description in thinking mode (n samples). A trace is kept only if its final
            answer matches the ground truth and it never refers to a description/caption/OCR.

Output: a WebDataset manifest of thinking samples {user: [image, question], assistant:
answer + reasoning}, i.e. the image replaces the description the teacher saw.

Usage:
    CUDA_VISIBLE_DEVICES=1 $EVAL_PY scripts/gen_traces.py describe --n 6000 --name traces-src-6k
    CUDA_VISIBLE_DEVICES=1 $EVAL_PY scripts/gen_traces.py reason --src traces-src-6k \
        --model Qwen/Qwen3-1.7B --name traces-q3-1b7
"""

import argparse
import os
import io
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# FlashInfer's JIT sampler misdetects sm_120 (Blackwell) as unsupported; use the PyTorch sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(ROOT / "src"))

# Short-answer QA sources with reliable ground truth.
QA_SOURCES = {"docvqa": 0.15, "chartqa": 0.15, "infographic_vqa": 0.10, "textvqa": 0.10, "ai2d": 0.10,
              "scienceqa": 0.05, "tallyqa": 0.08, "vsr": 0.05, "tat_qa": 0.07, "screenqa": 0.10, "iconqa": 0.05}
MAX_GOLD_CHARS = 60

DESCRIBE_PROMPT = (
    "Describe this image exhaustively for someone who cannot see it. Transcribe ALL visible text exactly, "
    "including small print, labels, buttons and numbers. For tables give every cell; for charts give the title, "
    "axes, legend and every data value you can read; for diagrams describe every part and connection; for photos "
    "describe objects, counts, colors, positions and spatial relations. Be factual; do not guess."
)
REASON_PROMPT = (
    "[Image]\n{desc}\n[/Image]\n\n{question}\n\n"
    "Answer based on what you see in the image. Give only the final answer, as short as possible."
)
# Harmless references to the text input are rewritten to refer to the image (the student sees pixels).
REWRITES = [
    (re.compile(r"\b(the|this) (provided |given |above )?(visual facts?|description|image description|facts?)\b( (says|states|mentions|shows|lists|notes))?", re.I),
     lambda m: "the image" + (" shows" if m.group(4) else "")),
    (re.compile(r"\b(as |is |are )?(described|listed|mentioned) (above|below|in the (image|description|facts))\b", re.I),
     lambda m: (m.group(1) or "") + "shown in the image"),
    (re.compile(r"\[/?Image\]", re.I), lambda m: "the image"),
]
# After rewriting, any remaining reference to a text source disqualifies the trace.
FORBIDDEN = re.compile(r"\b(OCR|transcri(bed|ption|pt)|caption|visual[_ ]facts?|the (text|image) description|"
                       r"according to the (text|description)|description (says|states|mentions))\b", re.I)


def rewrite_references(text: str) -> str:
    for pattern, repl in REWRITES:
        text = pattern.sub(repl, text)
    return text


from vlm.eval.answer_match import RELAXED_NUMERIC, answers_match, norm  # noqa: E402,F401


def phase_describe(args):
    from vllm import LLM, SamplingParams

    from vlm.data.image_checks import validate
    from vlm.data.sources import SOURCES, iter_source

    rng = random.Random(args.seed)
    items = []
    for name, share in QA_SOURCES.items():
        quota = int(share * args.n)
        got = 0
        for s in iter_source(SOURCES[name], args.seed):
            gold = s["messages"][-1]["content"].strip()
            if len(gold) > MAX_GOLD_CHARS:
                continue
            img, reason = validate(s["images"][0])
            if img is None:
                continue
            question = s["messages"][0]["content"][-1]["text"]
            items.append({"source": name, "image": img, "question": question, "gold": gold, "meta": s["meta"]})
            got += 1
            if got >= quota:
                break
        print(f"{name}: {got}", flush=True)
    rng.shuffle(items)

    llm = LLM(model=args.captioner, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, limit_mm_per_prompt={"image": 1}, seed=args.seed)
    msgs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": it["image"]},
                                          {"type": "text", "text": DESCRIBE_PROMPT}]}] for it in items]
    outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=1536))

    import webdataset as wds

    out_dir = ROOT / "manifests" / args.name
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    w = wds.ShardWriter(str(out_dir / "shards" / "shard-%06d.tar"), maxcount=2000, verbose=0)
    kept = 0
    for i, (it, o) in enumerate(zip(items, outs)):
        c = o.outputs[0]
        if c.finish_reason == "length" or len(c.text) < 40:
            continue
        buf = io.BytesIO()
        it["image"].save(buf, format="PNG")
        w.write({"__key__": f"tsrc-{i:08d}", "img.png": buf.getvalue(), "json": json.dumps({
            "question": it["question"], "gold": it["gold"], "description": c.text.strip(),
            "source": it["source"], "meta": it["meta"], "captioner": args.captioner}).encode()})
        kept += 1
    w.close()
    print(json.dumps({"described": kept, "requested": len(items)}))


def phase_reason(args):
    import webdataset as wds
    from vllm import LLM, SamplingParams

    src_dir = ROOT / "manifests" / args.src
    items = []
    for item in wds.WebDataset(sorted(str(p) for p in (src_dir / "shards").glob("*.tar")), shardshuffle=False, empty_check=False):
        rec = json.loads(item["json"])
        rec["img"] = item["img.png"]
        rec["key"] = item["__key__"]
        items.append(rec)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=12288, seed=args.seed)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": REASON_PROMPT.format(desc=r["description"], question=r["question"])}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in items]
    outs = llm.generate(prompts, SamplingParams(n=args.samples, temperature=0.6, top_p=0.95, top_k=20,
                                                max_tokens=6144, seed=args.seed))

    out_dir = ROOT / "manifests" / args.name
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    w = wds.ShardWriter(str(out_dir / "shards" / "shard-%06d.tar"), maxcount=2000, verbose=0)
    stats, records, raw_log = Counter(), [], []
    for r, o in zip(items, outs):
        best = None
        for c in o.outputs:
            text = c.text if "<think>" in c.text[:20] else "<think>\n" + c.text
            m = re.match(r"\s*<think>\n?(.*?)\n?</think>\s*(.*)", text, re.S)
            if c.finish_reason == "length" or not m:
                stats["unfinished"] += 1
                continue
            reasoning, answer = rewrite_references(m.group(1).strip()), rewrite_references(m.group(2).strip())
            raw_log.append({"key": r["key"], "gold": r["gold"], "answer": answer, "reasoning": reasoning[:3000]})
            if FORBIDDEN.search(reasoning) or FORBIDDEN.search(answer):
                stats["mentions_description"] += 1
                continue
            if not answers_match(answer, r["gold"], relaxed=r["source"] in RELAXED_NUMERIC):
                stats["wrong"] += 1
                continue
            stats["correct"] += 1
            if best is None or len(reasoning) < len(best[0]):
                best = (reasoning, answer)  # shortest correct trace
        if best is None:
            stats["question_dropped"] += 1
            continue
        key = f"trace-{len(records):08d}"
        meta = {**r["meta"], "key": key, "thinking": True, "task": "visual_reasoning_trace", "trace_teacher": args.model,
                "captioner": r["captioner"], "gold": r["gold"], "image_ext": "png"}
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["question"]}]},
                {"role": "assistant", "content": best[1], "reasoning": best[0]}]
        w.write({"__key__": key, "img.png": r["img"], "json": json.dumps({"messages": msgs, "meta": meta}).encode()})
        records.append(meta)
    w.close()
    with open(out_dir / "records.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in records)
    with open(out_dir / "raw_generations.jsonl", "w") as f:  # audit trail for the filters
        f.writelines(json.dumps(x) + "\n" for x in raw_log)
    manifest = {"name": args.name, "teacher": args.model, "questions": len(items), "kept": len(records),
                "acceptance": len(records) / max(1, len(items)), "stats": dict(stats),
                "by_source": dict(Counter(m["source"] for m in records))}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["describe", "reason"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--src")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--captioner", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    phase_describe(args) if args.phase == "describe" else phase_reason(args)


if __name__ == "__main__":
    main()
