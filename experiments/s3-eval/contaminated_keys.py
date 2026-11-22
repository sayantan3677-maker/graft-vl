"""eval-official keys whose image (exact hash or dHash) appears in any manifest used by the given training configs."""
import json, sys, collections
from pathlib import Path
paths = set()
for cfg in sys.argv[1:]:
    for d in json.load(open(cfg))["data"]:
        paths.add(d["path"])
ex, dh = set(), set()
for p in sorted(paths):
    f = Path(p) / "records.jsonl"
    if not f.exists(): continue
    for l in open(f):
        r = json.loads(l)
        if "exact_hash" in r: ex.add(r["exact_hash"]); dh.add(r["dhash"])
ev = [json.loads(l) for l in open("manifests/eval-official/records.jsonl")]
bad = sorted(r["key"] for r in ev if r["exact_hash"] in ex or r["dhash"] in dh)
print(dict(collections.Counter(k.rsplit("-", 1)[0] for k in bad)))
json.dump(bad, open("experiments/s3-eval/contaminated_keys.json", "w"))
