"""Audit S2 answer formats per source: trailing periods on short answers, 'Answer:' prefixes, MCQ answer shapes."""
import json, re, sys, itertools
from pathlib import Path
sys.path.insert(0, "src")
from vlm.data.loader import _read_samples

def shards(p): return sorted(str(x) for x in Path(p).rglob("*.tar"))
out = {}
for src in sorted(Path("manifests/s2").iterdir()):
    if not src.is_dir() or src.name.startswith("_"): continue
    n = per = ans_pre = mcq = mcq_bare = short = 0; ex = []
    for s in itertools.islice(_read_samples(shards(src), 0, False), 300):
        msgs = s["messages"]
        for i in range(0, len(msgs) - 1, 2):
            q = msgs[i]["content"]; q = q if isinstance(q, str) else "".join(p.get("text", "") for p in q)
            a = msgs[i + 1]["content"].strip()
            n += 1
            if len(a.split()) <= 4:
                short += 1; per += a.endswith(".")
            ans_pre += bool(re.match(r"(?i)^answer\s*:", a))
            if re.search(r"(?m)^\(?[A-D][\.\):]\s", q) or "Options:" in q or "Choices" in q:
                mcq += 1; mcq_bare += bool(re.fullmatch(r"[A-H]\.?", a))
            if len(ex) < 2: ex.append((q[-160:], a[:80]))
    out[src.name] = dict(turns=n, short_answers=short, short_with_period=per, answer_prefix=ans_pre, mcq=mcq, mcq_bare_letter=mcq_bare, ex=ex)
    print(f"{src.name:22s} turns {n:4d} short {short:4d} period {per:4d} 'Answer:' {ans_pre:4d} mcq {mcq:4d} bare {mcq_bare:4d}", flush=True)
json.dump(out, open("experiments/s3/format_audit.json", "w"), indent=1)
