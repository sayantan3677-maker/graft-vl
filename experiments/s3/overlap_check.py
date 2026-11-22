"""Exact-hash and dHash overlap between S3 training manifests and eval-official (+ the S2 eval overlap rule)."""
import json, glob, collections
ev = [json.loads(l) for l in open("manifests/eval-official/records.jsonl")]
ex = {r["exact_hash"]: r["source"] for r in ev}; dh = {r["dhash"]: r["source"] for r in ev}
tot = collections.Counter(); hits = collections.defaultdict(collections.Counter)
for f in sorted(glob.glob("manifests/s3/*/records.jsonl")):
    for l in open(f):
        r = json.loads(l); tot[r["source"]] += 1
        if r["exact_hash"] in ex: hits[r["source"]]["exact:" + ex[r["exact_hash"]]] += 1
        elif r["dhash"] in dh: hits[r["source"]]["dhash:" + dh[r["dhash"]]] += 1
for s in sorted(tot): print(f"{s:20s} {tot[s]:6d}", dict(hits[s]) or "")
json.dump({k: dict(v) for k, v in hits.items()}, open("experiments/s3/overlap.json", "w"), indent=1)
