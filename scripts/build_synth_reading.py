"""Build the synthetic reading manifest (see vlm/data/synth_text.py) in parallel.

Usage: python scripts/build_synth_reading.py --n 200000 --name synth-reading-200k [--workers 48]
"""

import argparse
import io
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHUNK = 1000


def load_corpus(n_sentences: int = 200_000) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train").shuffle(seed=0).select(range(120_000))
    words, sentences = Counter(), []
    for r in ds:
        for m in r["messages"]:
            text = m["content"]
            if not text.isascii():
                continue
            for s in re.split(r"(?<=[.!?])\s+", text):
                s = " ".join(s.split())
                if 25 <= len(s) <= 160 and not re.search(r"[`{}\\|<>#*_=\[\]]", s):
                    sentences.append(s)
            words.update(w for w in re.findall(r"[A-Za-z]{3,14}", text))
        if len(sentences) > n_sentences:
            break
    vocab = [w for w, c in words.most_common(30_000) if c >= 3]
    return vocab, sentences[:n_sentences]


def load_backgrounds(n: int = 1500):
    import webdataset as wds
    from PIL import Image

    bgs = []
    for it in wds.WebDataset(sorted(str(p) for p in (ROOT / "manifests/s2/pixmo_cap/shards").glob("*.tar"))[:4],
                             shardshuffle=False, empty_check=False):
        im = Image.open(io.BytesIO(it[next(k for k in it if k.startswith("img."))]))
        im.thumbnail((1280, 1280))
        bgs.append(im.convert("RGB"))
        if len(bgs) >= n:
            break
    return bgs


_STATE = {}


def _init(vocab, sentences, backgrounds_path):
    import pickle

    _STATE["vocab"], _STATE["sentences"] = vocab, sentences
    with open(backgrounds_path, "rb") as f:
        _STATE["bgs"] = pickle.load(f)


def _chunk(args):
    chunk_id, out_dir = args
    import webdataset as wds

    from vlm.data.image_checks import dhash, exact_hash
    from vlm.data.synth_text import SynthText

    gen = SynthText(_STATE["vocab"], _STATE["sentences"], _STATE["bgs"], seed=chunk_id)
    recs = []
    with wds.TarWriter(f"{out_dir}/shards/shard-{chunk_id:06d}.tar") as w:
        for i in range(CHUNK):
            s = gen.sample()
            buf = io.BytesIO()
            s.image.save(buf, format="PNG")
            key = f"synth-{chunk_id:05d}-{i:04d}"
            meta = {"key": key, "source": "synth_reading", "task": f"read_{s.task}", "image_type": "synthetic_text",
                    "origin": "synthetic", "license": "generated in-house", "commercial_ok": True, "thinking": False,
                    "split": "train", "width": s.image.width, "height": s.image.height,
                    "exact_hash": exact_hash(s.image), "dhash": dhash(s.image)}
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s.question}]},
                    {"role": "assistant", "content": s.answer}]
            w.write({"__key__": key, "img.png": buf.getvalue(), "json": json.dumps({"messages": msgs, "meta": meta}).encode()})
            recs.append(meta)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    import pickle

    out = ROOT / "manifests" / args.name
    (out / "shards").mkdir(parents=True, exist_ok=True)
    vocab, sentences = load_corpus()
    bgs = load_backgrounds()
    bg_path = out / "_backgrounds.pkl"
    with open(bg_path, "wb") as f:
        pickle.dump(bgs, f)
    print(f"vocab {len(vocab)}, sentences {len(sentences)}, backgrounds {len(bgs)}", flush=True)
    n_chunks = args.n // CHUNK
    with Pool(args.workers, initializer=_init, initargs=(vocab, sentences, str(bg_path))) as pool:
        records = [r for recs in pool.imap_unordered(_chunk, [(i, str(out)) for i in range(n_chunks)]) for r in recs]
    os.remove(bg_path)
    with open(out / "records.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    manifest = {"name": args.name, "written": len(records), "by_task": dict(Counter(r["task"] for r in records)),
                "origin": "synthetic", "license": "generated in-house"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
