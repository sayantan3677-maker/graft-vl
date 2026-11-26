"""v2 data: question-answer pairs for permissively licensed document pages, written by an Apache-2.0 teacher
(Qwen3-VL-8B-Instruct) and kept only if they pass two checks.

Pass 1 (generate): the teacher sees the page and writes N diverse QA pairs as JSON
        (extractive / numeric-reasoning / layout / summary), each with a short answer and a one-sentence rationale.
Pass 2 (verify):   the teacher answers every question again, independently, without seeing its first answer.
Keep a pair when:  both answers agree (normalized match or numeric equality), and, for extractive questions,
                   the answer occurs in the page's ground-truth PDF text (DocLayNet `pdf_cells`).

Processes one parquet shard at a time and writes <out-dir>/<shard>.jsonl + .summary.json; finished shards are
skipped, so an interrupted run resumes.
  CUDA_VISIBLE_DEVICES=1 python experiments/v2/gen_doc_qa.py --parquet data/v2/raw/doclaynet/data/train-*.parquet \
      --out-dir data/v2/doclaynet-qa
"""

import argparse
import base64
import io
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

GEN_PROMPT = """You are creating training data for a document-understanding model.
Look at this document page and write {n} question-answer pairs about it. Use a mix of these types:
- "extractive": the answer is a short span copied exactly from the page (a name, number, date, heading, value).
- "reasoning": the answer needs reading several values and comparing or computing (e.g. a difference, a total, which is largest).
- "layout": about structure (e.g. what a table column contains, which section comes after another, what a figure shows).
- "summary": what the page or one section is about, answered in one or two sentences.
Rules: every question must be answerable from this page alone; never ask about something not visible;
answers must be correct and as short as possible (except "summary"); prefer specific, non-trivial questions.
Reply with JSON only, a list of objects: {{"type": ..., "question": ..., "answer": ..., "rationale": <one sentence>}}."""

VERIFY_PROMPT = "{q}\nAnswer as briefly as possible, using only the page."


def page_text(row) -> str:
    cells = row.get("pdf_cells") or []
    return " ".join(c.get("text", "") for page in cells for c in (page or []))


def norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"[\s,]+", " ", s)
    return re.sub(r"[^\w%$.\- ]", "", s).strip(" .")


def numbers(s: str):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", s)]


def agree(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb or (len(na) > 3 and na in nb) or (len(nb) > 3 and nb in na):
        return True
    xa, xb = numbers(a), numbers(b)
    return bool(xa) and bool(xb) and len(xa) == 1 and any(abs(xa[0] - y) <= 1e-6 * max(1, abs(y)) for y in xb)


def parse_json_list(text: str):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    keep = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and all(isinstance(it.get(k), str) and it[k].strip() for k in ("type", "question", "answer")):
            keep.append({k: it.get(k, "").strip() if isinstance(it.get(k), str) else "" for k in ("type", "question", "answer", "rationale")})
    return keep


def data_url(img_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode()


def load_shard(path, n, seed):
    """All pages of one parquet shard (or a random n of them), with their row index for building training shards later."""
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["image", "pdf_cells", "metadata"]).to_pylist()
    for i, r in enumerate(rows):
        r["_row"] = i
    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--pages-per-shard", type=int, default=0, help="0 = every page")
    ap.add_argument("--qa-per-page", type=int, default=5)
    ap.add_argument("--teacher", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--out-dir", required=True, help="one <shard>.jsonl + <shard>.summary.json per shard; done shards are skipped")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in args.parquet if not (out_dir / (Path(p).stem + ".summary.json")).exists()]
    print(f"{len(args.parquet) - len(todo)} shards done, {len(todo)} to go", flush=True)
    if not todo:
        return
    llm = LLM(model=args.teacher, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
              limit_mm_per_prompt={"image": 1}, seed=args.seed)
    gen_sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=1500, seed=args.seed)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=200)

    def conv(img, text):
        return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(img)}},
                                             {"type": "text", "text": text}]}]

    for path in todo:
        t0, shard = time.time(), Path(path).stem
        pages = load_shard(path, args.pages_per_shard, args.seed)
        outs = llm.chat([conv(p["image"]["bytes"], GEN_PROMPT.format(n=args.qa_per_page)) for p in pages], gen_sp,
                        use_tqdm=False)
        cands = [(pi, qa) for pi, o in enumerate(outs) for qa in parse_json_list(o.outputs[0].text)]
        vouts = llm.chat([conv(pages[pi]["image"]["bytes"], VERIFY_PROMPT.format(q=qa["question"])) for pi, qa in cands],
                         ver_sp, use_tqdm=False)
        stats, n_kept = Counter(), 0
        tmp = out_dir / (shard + ".jsonl.tmp")
        with open(tmp, "w") as f:
            for (pi, qa), vo in zip(cands, vouts):
                p, v = pages[pi], vo.outputs[0].text.strip()
                typ = qa["type"].lower()
                stats[f"cand/{typ}"] += 1
                if typ == "summary":
                    ok, why = True, "summary (not verified)"
                elif not agree(qa["answer"], v):
                    ok, why = False, "disagree"
                elif typ == "extractive" and norm(qa["answer"]) not in norm(page_text(p)):
                    ok, why = False, "not in page text"
                else:
                    ok, why = True, "ok"
                stats[f"{'keep' if ok else 'drop'}/{typ}/{why}"] += 1
                n_kept += ok
                f.write(json.dumps({"shard": shard, "row": p["_row"], "page_hash": p["metadata"]["page_hash"],
                                    "doc_category": p["metadata"]["doc_category"], "type": typ,
                                    "question": qa["question"], "answer": qa["answer"], "rationale": qa["rationale"],
                                    "verify_answer": v, "keep": ok, "reason": why}) + "\n")
        tmp.rename(out_dir / (shard + ".jsonl"))
        summary = {"shard": shard, "pages": len(pages), "candidates": len(cands), "kept": n_kept,
                   "kept_per_page": round(n_kept / max(1, len(pages)), 2),
                   "categories": dict(Counter(p["metadata"]["doc_category"] for p in pages)),
                   "stats": dict(sorted(stats.items())), "minutes": round((time.time() - t0) / 60, 1)}
        (out_dir / (shard + ".summary.json")).write_text(json.dumps(summary, indent=1))
        print(shard, summary["pages"], "pages", n_kept, "kept", summary["minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
