"""v2 data: question-answer pairs for permissively licensed document pages, written by an Apache-2.0 teacher
(Qwen3-VL-8B-Instruct) and kept only if they pass two checks.

Pass 1 (generate): the teacher sees the page and writes N diverse QA pairs as JSON
        (extractive / numeric-reasoning / layout / summary), each with a short answer and a one-sentence rationale.
Pass 2 (verify):   the teacher answers every question again, independently, without seeing its first answer.
Keep a pair when:  both answers agree (normalized match or numeric equality), and, for extractive questions,
                   the answer occurs in the page's ground-truth PDF text (DocLayNet `pdf_cells`).

Usage (pilot):
  CUDA_VISIBLE_DEVICES=1 python experiments/v2/gen_doc_qa.py --parquet data/v2/raw/doclaynet/data/*.parquet \
      --n-pages 300 --out data/v2/doclaynet-qa-pilot.jsonl
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


def load_pages(paths, n, seed):
    import pyarrow.parquet as pq

    rows = []
    for p in paths:
        t = pq.read_table(p, columns=["image", "pdf_cells", "metadata"])
        rows += t.to_pylist()
    random.Random(seed).shuffle(rows)
    by_cat = {}
    for r in rows:  # stratify across document categories
        by_cat.setdefault(r["metadata"]["doc_category"], []).append(r)
    out, i = [], 0
    while len(out) < n and any(by_cat.values()):
        cat = sorted(by_cat)[i % len(by_cat)]
        if by_cat[cat]:
            out.append(by_cat[cat].pop())
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--n-pages", type=int, default=300)
    ap.add_argument("--qa-per-page", type=int, default=5)
    ap.add_argument("--teacher", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    t0 = time.time()
    pages = load_pages(args.parquet, args.n_pages, args.seed)
    llm = LLM(model=args.teacher, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
              limit_mm_per_prompt={"image": 1}, seed=args.seed)
    gen_sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=1500, seed=args.seed)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=200)

    def conv(img, text):
        return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(img)}},
                                             {"type": "text", "text": text}]}]

    outs = llm.chat([conv(p["image"]["bytes"], GEN_PROMPT.format(n=args.qa_per_page)) for p in pages], gen_sp)
    cands = []
    for pi, o in enumerate(outs):
        for qa in parse_json_list(o.outputs[0].text):
            cands.append((pi, qa))
    vouts = llm.chat([conv(pages[pi]["image"]["bytes"], VERIFY_PROMPT.format(q=qa["question"])) for pi, qa in cands], ver_sp)

    stats, kept = Counter(), []
    with open(args.out, "w") as f:
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
            rec = {"page_hash": p["metadata"]["page_hash"], "doc_category": p["metadata"]["doc_category"],
                   "type": typ, "question": qa["question"], "answer": qa["answer"], "rationale": qa["rationale"],
                   "verify_answer": v, "keep": ok, "reason": why}
            if ok:
                kept.append(rec)
            f.write(json.dumps(rec) + "\n")
    summary = {"pages": len(pages), "candidates": len(cands), "kept": len(kept),
               "kept_per_page": round(len(kept) / max(1, len(pages)), 2), "stats": dict(sorted(stats.items())),
               "minutes": round((time.time() - t0) / 60, 1)}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
