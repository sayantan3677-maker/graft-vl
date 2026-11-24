"""Answer-style probe: open-ended questions with NO format instruction (as a demo/API user would ask).
Counts replies that end in an "Answer: X" line — the CoSyn/Geo170K habit S4F's rationale normalization targets.
Two prompt styles: the bare question, and the question + "Explain your reasoning." (where CoSyn-style
"rationale + Answer: X" would surface).
Usage: python experiments/s4f/answer_style_probe.py <out.json> exports/<a> exports/<b> ..."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from eval_hybrid_export import generate_raw, load_export, official_by_source  # noqa: E402
import torch  # noqa: E402

SOURCES, N = ("ai2d_test", "chartqa_test", "docvqa_val", "infovqa_val"), 40
ANS = re.compile(r"(^|\n)\s*\**answer\**\s*:", re.I)
by = official_by_source()
items = [(src, s) for src in SOURCES for s in by[src][-N:]]  # tail of each source: not the items of the 200-item QA runs
out = {}
for exp in sys.argv[2:]:
    model, proc = load_export(exp)
    rows = []
    for src, s in items:
        base = s["messages"][0]["content"][-1]["text"].split("\n")[0].strip()
        for style, q in (("bare", base), ("explain", base + " Explain your reasoning.")):
            text, n_tok, _ = generate_raw(model, proc, s["images"][0], q, 2_560_000, 384, False, False)
            rows.append({"source": src, "style": style, "q": q, "pred": text, "tokens": n_tok,
                         "answer_line": bool(ANS.search(text))})
    out[exp] = {}
    for style in ("bare", "explain"):
        rs = [r for r in rows if r["style"] == style]
        out[exp][style] = {"n": len(rs), "answer_line_pct": round(100 * sum(r["answer_line"] for r in rs) / len(rs), 1),
                           "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs), 1)}
    out[exp]["rows"] = rows
    print(exp, {k: v for k, v in out[exp].items() if k != "rows"}, file=sys.stderr, flush=True)
    del model; torch.cuda.empty_cache()
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
