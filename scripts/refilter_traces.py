"""Post-filter a traces manifest: drop empty/short reasoning and references to the text input.

Usage: python scripts/refilter_traces.py --src manifests/traces-q3-1b7-s2 --dst manifests/traces-q3-1b7-s2-v3
"""

import argparse
import collections
import glob
import json
import os
import re

import webdataset as wds

LEAK = re.compile(r"\b(the user (mentioned|said|described|provided|noted|listed|gave)|(list|text|information|details|data|content) (provided|given)|"
                  r"provided (list|text|information|details|data|content|image description)|(as|was) (provided|given) (above|below)|"
                  r"based on the (information|details|text) (given|provided))\b", re.I)
MIN_REASONING_CHARS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    os.makedirs(args.dst + "/shards", exist_ok=True)
    stats, recs = collections.Counter(), []
    w = wds.ShardWriter(args.dst + "/shards/shard-%06d.tar", maxcount=1000, verbose=0)
    for it in wds.WebDataset(sorted(glob.glob(args.src + "/shards/*.tar")), shardshuffle=False, empty_check=False):
        rec = json.loads(it["json"])
        a = rec["messages"][-1]
        reasoning = a.get("reasoning") or ""
        if len(reasoning.strip()) < MIN_REASONING_CHARS:
            stats["empty_or_short_reasoning"] += 1
            continue
        if LEAK.search(reasoning) or LEAK.search(a["content"]):
            stats["leak_phrase"] += 1
            continue
        w.write({k: v for k, v in it.items() if k in ("__key__", "json") or k.startswith("img.")})
        recs.append(rec["meta"])
    w.close()
    with open(args.dst + "/records.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    manifest = {"name": os.path.basename(args.dst), "derived_from": args.src, "written": len(recs), "kept": len(recs),
                "refilter_drops": dict(stats), "by_source": dict(collections.Counter(r["source"] for r in recs))}
    with open(args.dst + "/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
