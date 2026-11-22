"""Contamination-free QA eval from OFFICIAL validation/test splits (roadmap §4.4, §7.3).

Our training data comes from FineVision *train* splits; these validation/test splits are
disjoint by construction. Each record keeps all reference answers (`meta.golds`); a
prediction is correct if it matches any of them (consistent across runs, not the official
ANLS/VQA metrics — use lmms-eval for those).

Usage: python scripts/build_official_eval.py [--per-set 200]
"""

import argparse
import ast
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import webdataset as wds  # noqa: E402

from vlm.data.image_checks import dhash, exact_hash  # noqa: E402

LETTERS = "ABCDEFGHIJ"


def _list(x):
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else [v]
    except Exception:
        return [x]


def sets():
    """(name, image_type, repo, config, split, row → (question, golds) or None)."""
    def docvqa(r):
        return r["question"], _list(r["answers"])

    def chartqa(r):
        return r["question"], [str(r["answer"])]

    def textvqa(r):
        golds = _list(r["answers"])
        return r["question"], sorted(set(golds), key=golds.count, reverse=True)

    def ai2d(r):
        opts = _list(r["options"])
        body = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(opts))
        return f"{r['question']}\n{body}\nAnswer with the option's letter.", [LETTERS[int(r["answer"])]]

    def vqav2(r):
        return r["question"], [r["multiple_choice_answer"]]

    def ocrbench_screen(r):
        if r.get("dataset_name") not in {"rico", "screen2words", "websrc", "webui", "screenqa"} and "APP" not in str(r.get("type", "")):
            return None
        return r["question"], _list(r["answers"])

    return [
        ("docvqa_val", "page", "lmms-lab/DocVQA", "DocVQA", "validation", docvqa),
        ("infovqa_val", "page", "lmms-lab/DocVQA", "InfographicVQA", "validation", docvqa),
        ("chartqa_test", "chart", "lmms-lab/ChartQA", None, "test", chartqa),
        ("textvqa_val", "photo", "lmms-lab/textvqa", None, "validation", textvqa),
        ("vqav2_val", "photo", "lmms-lab/VQAv2", None, "validation", vqav2),
        ("ai2d_test", "diagram", "lmms-lab/ai2d", None, "test", ai2d),
        ("ocrbench2_screen", "screenshot", "lmms-lab/OCRBench-v2", None, "test", ocrbench_screen),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-set", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from datasets import load_dataset

    out = ROOT / "manifests" / "eval-official"
    (out / "shards").mkdir(parents=True, exist_ok=True)
    records = []
    with wds.TarWriter(str(out / "shards" / "shard-000000.tar")) as w:
        for name, itype, repo, cfg, split, fn in sets():
            ds = load_dataset(repo, cfg, split=split, streaming=True).shuffle(seed=args.seed, buffer_size=2000)
            n = 0
            for r in ds:
                qa = fn(r)
                if qa is None:
                    continue
                question, golds = qa
                golds = [str(g).strip() for g in golds if str(g).strip()]
                if not golds:
                    continue
                img = r["image"].convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                key = f"{name}-{n:05d}"
                meta = {"key": key, "source": name, "image_type": itype, "split": split, "golds": golds,
                        "exact_hash": exact_hash(img), "dhash": dhash(img)}
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                        {"role": "assistant", "content": golds[0]}]
                w.write({"__key__": key, "img.png": buf.getvalue(), "json": json.dumps({"messages": msgs, "meta": meta}).encode()})
                records.append(meta)
                n += 1
                if n >= args.per_set:
                    break
            print(f"{name}: {n}", flush=True)
    with open(out / "records.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    print(f"wrote {len(records)} → {out}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os

    os._exit(0)  # skip HF streaming thread teardown
