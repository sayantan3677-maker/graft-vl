"""Build versioned WebDataset shards + manifest for a training stage (roadmap §4.1).

Streams each source, validates images, removes exact and near-duplicates across
sources, filters over-long targets, and writes:

    manifests/<name>/shards/shard-000000.tar     (key.img.{jpg,png} + key.json)
    manifests/<name>/records.jsonl               (one metadata record per sample)
    manifests/<name>/manifest.json               (mixture, counts, rejections, manifest hash)

Usage:
    python scripts/build_shards.py --mixture configs/data/s1.json --n 5000 --name s1-5k
"""

import argparse
import io
import os
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import webdataset as wds  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from vlm.data.image_checks import dhash, exact_hash, validate  # noqa: E402
from vlm.data.sources import SOURCES, iter_source, manifest_hash  # noqa: E402


def encode(image, image_type: str) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if image_type == "photo":
        image.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "jpg"
    image.save(buf, format="PNG", optimize=False)  # screenshots/pages: lossless for text
    return buf.getvalue(), "png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", help="mixture JSON; or use --source for a single-source manifest")
    ap.add_argument("--source", help="build one source only (parallel per-source builds, mixed at load time)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--max-target-tokens", type=int, default=1024)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--all-turns", action="store_true", help="keep every QA turn per image (multi-turn sample)")
    args = ap.parse_args()

    if args.source:
        weights = {args.source: 1.0}
    else:
        weights = json.loads(Path(args.mixture).read_text())["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "mixture weights must sum to 1"
    quotas = {name: round(w * args.n) for name, w in weights.items()}
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out = ROOT / "manifests" / args.name
    (out / "shards").mkdir(parents=True, exist_ok=True)
    writer = wds.ShardWriter(str(out / "shards" / "shard-%06d.tar"), maxcount=args.shard_size, verbose=0)
    seen_exact, seen_dhash = set(), set()
    records, rejections, counts = [], Counter(), Counter()

    for name, quota in quotas.items():
        source = SOURCES[name]
        t0 = time.time()
        for sample in iter_source(source, args.seed, all_turns=args.all_turns):
            if counts[name] >= quota:
                break
            image, reason = validate(sample["images"][0])
            if image is None:
                rejections[f"{name}:{reason}"] += 1
                continue
            eh, dh = exact_hash(image), dhash(image)
            if eh in seen_exact or dh in seen_dhash:
                rejections[f"{name}:duplicate"] += 1
                continue
            answer = sample["messages"][-1]["content"]
            n_target = max(len(tok(m["content"], add_special_tokens=False)["input_ids"])
                           for m in sample["messages"] if m["role"] == "assistant")
            if not answer or n_target > args.max_target_tokens:
                rejections[f"{name}:target_length"] += 1
                continue
            seen_exact.add(eh)
            seen_dhash.add(dh)

            key = f"{name}-{counts[name]:08d}"
            img_bytes, ext = encode(image, source.image_type)
            meta = {**sample["meta"], "key": key, "width": image.width, "height": image.height,
                    "exact_hash": eh, "dhash": dh, "n_target_tokens": n_target, "image_ext": ext}
            writer.write({"__key__": key, f"img.{ext}": img_bytes,
                          "json": json.dumps({"messages": sample["messages"], "meta": meta}).encode()})
            records.append(meta)
            counts[name] += 1
        print(f"{name}: {counts[name]}/{quota} in {time.time() - t0:.0f}s", flush=True)
    writer.close()

    with open(out / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    by_type = Counter(r["image_type"] for r in records)
    by_origin = Counter(r["origin"] for r in records)
    manifest = {
        "name": args.name, "mixture_file": args.mixture, "mixture": weights, "seed": args.seed,
        "requested": args.n, "written": len(records), "counts": dict(counts), "quotas": quotas,
        "by_image_type": dict(by_type), "by_origin": dict(by_origin),
        "synthetic_fraction": by_origin["synthetic"] / max(1, len(records)),
        "rejections": dict(rejections), "all_turns": args.all_turns, "max_target_tokens": args.max_target_tokens,
        "sources": {n: SOURCES[n].to_dict() for n in weights},
        "manifest_hash": manifest_hash(records),
        "contamination_checked": False,  # set once eval sets exist (roadmap §4.4)
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["written", "counts", "by_image_type", "synthetic_fraction", "manifest_hash"]}, indent=2))


if __name__ == "__main__":
    main()
    # HF streaming leaves prefetch threads alive; skip interpreter teardown (all files are closed).
    sys.stdout.flush()
    os._exit(0)
