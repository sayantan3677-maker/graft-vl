"""Rewrite S3 manifests without samples whose image matches eval-official (exact hash or dHash)."""
import json, glob, io, tarfile, shutil
from pathlib import Path
ev = [json.loads(l) for l in open("manifests/eval-official/records.jsonl")]
ex = {r["exact_hash"] for r in ev}; dh = {r["dhash"] for r in ev}
import sys
for man in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "manifests/s3/*/manifest.json")):
    d = Path(man).parent
    recs = [json.loads(l) for l in open(d / "records.jsonl")]
    bad = {r["key"] for r in recs if r["exact_hash"] in ex or r["dhash"] in dh}
    if not bad:
        continue
    new = d / "shards.new"; new.mkdir(exist_ok=True)
    for shard in sorted((d / "shards").glob("*.tar")):
        with tarfile.open(shard) as src, tarfile.open(new / shard.name, "w") as dst:
            for m in src.getmembers():
                if m.name.split(".", 1)[0] in bad:
                    continue
                dst.addfile(m, src.extractfile(m))
    shutil.rmtree(d / "shards"); new.rename(d / "shards")
    keep = [r for r in recs if r["key"] not in bad]
    with open(d / "records.jsonl", "w") as f:
        for r in keep: f.write(json.dumps(r) + "\n")
    m = json.loads(Path(man).read_text())
    m["counts"] = {k: len(keep) for k in m["counts"]}; m["written"] = len(keep)
    m["dropped_eval_overlap"] = len(bad)
    Path(man).write_text(json.dumps(m, indent=2))
    print(d.name, "dropped", len(bad), "kept", len(keep))
