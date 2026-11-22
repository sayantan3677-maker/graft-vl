"""Client-side serving benchmark + accuracy subset against a running OpenAI-compatible vLLM server (roadmap §12.3).

perf: realistic single-image requests (document page / phone screenshot / photo from manifests/eval-official),
      64 output tokens (ignore_eos), non-thinking, streamed → TTFT, per-request decode speed, aggregate output tok/s.
      Every request uses a distinct image (no prefix/mm-cache hits across requests).
acc:  N questions per eval-official source (default 30 × 7 = 210), short-answer prompt + scoring as eval_hybrid.py.

  python scripts/bench_hybrid_serving.py --url http://localhost:8011 --model <served-name> --tag bf16 \
      [--concurrency 1,8,32] [--acc-per-source 30] --out reports/serving/bench-<tag>.json
"""

import argparse
import asyncio
import base64
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TYPES = {"page": "docvqa_val", "screenshot": "ocrbench2_screen", "photo": "vqav2_val"}
PERF_PROMPT = {"page": "Summarize this document.", "screenshot": "Describe what this screen shows.",
               "photo": "Describe this photo."}
SHORT = "\nAnswer the question using a single word or phrase."


def b64(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def one(session, url, model, img_url, text, max_tokens, ignore_eos, thinking=False, retries=2):
    for attempt in range(retries + 1):
        try:
            return await _one(session, url, model, img_url, text, max_tokens, ignore_eos, thinking)
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
            if attempt == retries:
                raise


def _session():  # fresh connection per request: avoids reusing keep-alive sockets the server already closed
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800), connector=aiohttp.TCPConnector(force_close=True, limit=0))


async def _one(session, url, model, img_url, text, max_tokens, ignore_eos, thinking=False):
    body = {"model": model, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url}},
                                                                       {"type": "text", "text": text}]}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "ignore_eos": ignore_eos, "chat_template_kwargs": {"enable_thinking": thinking}}
    t0 = time.perf_counter()
    ttft, text_out, usage, last = None, [], None, None
    async with session.post(f"{url}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                piece = ch.get("delta", {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_out.append(piece)
                    last = time.perf_counter()
    end = time.perf_counter()
    out_toks = usage["completion_tokens"] if usage else None
    return {"ttft": ttft, "e2e": end - t0, "out_tokens": out_toks, "prompt_tokens": usage["prompt_tokens"] if usage else None,
            "tpot": (last - t0 - ttft) / max(1, out_toks - 1) if (ttft and out_toks and out_toks > 1) else None,
            "text": "".join(text_out)}


async def run_level(url, model, reqs, conc, max_tokens):
    sem = asyncio.Semaphore(conc)
    async with _session() as s:
        async def go(r):
            async with sem:
                res = await one(s, url, model, r["img"], r["text"], max_tokens, True)
                res["type"] = r["type"]
                return res
        t0 = time.perf_counter()
        res = await asyncio.gather(*(go(r) for r in reqs))
        wall = time.perf_counter() - t0
    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    ttft = [r["ttft"] for r in res]
    tpot = [r["tpot"] for r in res if r["tpot"]]
    out = {"concurrency": conc, "requests": len(res), "wall_s": round(wall, 2),
           "ttft_mean_ms": round(1000 * st.mean(ttft), 1), "ttft_p50_ms": round(1000 * pct(ttft, .5), 1),
           "ttft_p90_ms": round(1000 * pct(ttft, .9), 1),
           "tpot_mean_ms": round(1000 * st.mean(tpot), 2),
           "per_request_decode_tok_s": round(1 / st.mean(tpot), 1),
           "output_tok_s": round(sum(r["out_tokens"] for r in res) / wall, 1),
           "req_s": round(len(res) / wall, 2),
           "mean_prompt_tokens": round(st.mean(r["prompt_tokens"] for r in res), 1)}
    out["by_type"] = {t: {"ttft_mean_ms": round(1000 * st.mean(r["ttft"] for r in res if r["type"] == t), 1),
                          "prompt_tokens": round(st.mean(r["prompt_tokens"] for r in res if r["type"] == t), 1)}
                      for t in sorted({r["type"] for r in res})}
    return out


async def run_acc(url, model, samples, conc=32):
    from vlm.eval.answer_match import answers_match

    sem = asyncio.Semaphore(conc)
    urls = [b64(smp["images"][0]) for smp in samples]  # encode up front (blocking work outside the event loop's requests)
    async with _session() as s:
        async def go(smp, img_url):
            async with sem:
                q = smp["messages"][0]["content"][-1]["text"] + SHORT
                r = await one(s, url, model, img_url, q, 32, False)
                src = smp["meta"]["source"]
                ans = r["text"].split("</think>")[-1].strip()
                ok = any(answers_match(ans, g, relaxed=src.startswith(("chartqa", "infovqa"))) for g in smp["meta"]["golds"])
                return {"key": smp["meta"]["key"], "source": src, "pred": ans[:200], "golds": smp["meta"]["golds"][:3], "ok": ok}
        rows = await asyncio.gather(*(go(x, u) for x, u in zip(samples, urls)))
    per = {}
    for r in rows:
        per.setdefault(r["source"], []).append(r["ok"])
    return {"n": len(rows), "overall": round(100 * sum(r["ok"] for r in rows) / len(rows), 2),
            "per_source": {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(per.items())}, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8011")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--acc-per-source", type=int, default=30)
    ap.add_argument("--min-requests", type=int, default=12)
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from eval_s1 import load_eval_samples

    samples = load_eval_samples(ROOT / "manifests/eval-official", None, 1400)
    by = {}
    for s in samples:
        by.setdefault(s["meta"]["source"], []).append(s)
    report = {"tag": args.tag, "model": args.model, "max_tokens": args.max_tokens, "levels": []}
    concs = [int(c) for c in args.concurrency.split(",") if c]
    # warmup (distinct images from the tail of each pool, not reused below)
    warm = [{"type": t, "img": b64(by[src][-1 - i]["images"][0]), "text": PERF_PROMPT[t]} for t, src in TYPES.items() for i in range(2)]
    asyncio.run(run_level(args.url, args.model, warm, 6, 8))
    cursor = {t: 40 for t in TYPES}  # images 0..39 of each source are reserved for the accuracy subset
    for conc in concs:
        n = max(args.min_requests, args.waves * conc)  # ≥3 waves at the target concurrency, mixed types
        reqs = []
        for i in range(n):
            t = list(TYPES)[i % 3]
            reqs.append({"type": t, "img": b64(by[TYPES[t]][cursor[t]]["images"][0]), "text": PERF_PROMPT[t]})
            cursor[t] += 1
        lvl = asyncio.run(run_level(args.url, args.model, reqs, conc, args.max_tokens))
        print(json.dumps(lvl), flush=True)
        report["levels"].append(lvl)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    if args.acc_per_source:
        acc_samples = [s for src in sorted(by) for s in by[src][: args.acc_per_source]]
        report["accuracy"] = asyncio.run(run_acc(args.url, args.model, acc_samples))
        print(json.dumps({k: v for k, v in report["accuracy"].items() if k != "rows"}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
