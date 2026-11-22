"""Same evaluation as scripts/eval_text.py (same task loaders, prompts, Qwen3 model-card sampling, per-request
seed 1234, n=--seeds, same JSON output format), but submits every task x mode in ONE vLLM generate call.

Why: with per-task calls the GPU idles on the long thinking tails (a few 32K-token generations decode alone
for tens of minutes). One combined call keeps the batch full. Per-request seeding makes the samples
independent of batch composition (up to batched-kernel numerics), so results are comparable with eval_text.py.

Usage (vlm-eval env):
    CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/vlm-eval/bin/python scripts/eval_text_batched.py --model <dir> \
        --tasks mmlu_pro gsm8k ifeval math500 --seeds 2 --limit mmlu_pro=500 gsm8k=500 math500=200 \
        --gpu-mem 0.45 --out <json> [--items-out <json>]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_text import SAMPLING  # noqa: E402
from vlm.eval.text_tasks import TASKS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--tasks", nargs="+", default=["mmlu_pro", "gsm8k", "ifeval", "math500"])
    ap.add_argument("--modes", nargs="+", default=["think", "nothink"], choices=["think", "nothink"])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--limit", nargs="*", default=[])
    ap.add_argument("--max-tokens-think", type=int, default=32768)
    ap.add_argument("--max-tokens-nothink", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=36864)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items-out")
    args = ap.parse_args()
    limits = {k: int(v) for k, v in (x.split("=") for x in args.limit)}

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, seed=0)
    tok = llm.get_tokenizer()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(out_path.read_text()) if out_path.exists() else {"model": args.model, "results": {}}
    report["runner"] = "eval_text_batched.py (all task x mode in one generate call)"

    jobs, prompts, params = [], [], []
    for task_name in args.tasks:
        items = TASKS[task_name].load(limits.get(task_name), 0)
        for mode in args.modes:
            key = f"{task_name}/{mode}"
            if key in report["results"]:
                continue
            thinking = mode == "think"
            sp = SamplingParams(n=args.seeds, seed=1234, max_tokens=args.max_tokens_think if thinking else args.max_tokens_nothink,
                                **SAMPLING[thinking])
            jobs.append((key, task_name, thinking, items, len(prompts), sp.max_tokens))
            prompts += [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=thinking) for it in items]
            params += [sp] * len(items)
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    minutes = (time.time() - t0) / 60
    items_report = {}
    for key, task_name, thinking, items, start, max_tokens in jobs:
        task = TASKS[task_name]
        per_seed = [[] for _ in range(args.seeds)]
        lengths, truncated, item_scores = [], 0, {}
        for it, out in zip(items, outputs[start:start + len(items)]):
            for s, comp in enumerate(out.outputs):
                sc = task.score(it, comp.text)
                per_seed[s].append(sc)
                item_scores.setdefault(str(it["id"]), []).append(sc)
                lengths.append(len(comp.token_ids))
                truncated += comp.finish_reason == "length"
        accs = [100 * sum(x) / len(x) for x in per_seed]
        report["results"][key] = {
            "n_items": len(items), "seeds": args.seeds, "acc_mean": statistics.mean(accs),
            "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0, "acc_per_seed": accs,
            "mean_gen_tokens": statistics.mean(lengths), "truncated_frac": truncated / len(lengths),
            "sampling": SAMPLING[thinking], "max_tokens": max_tokens, "minutes": minutes, "minutes_note": "whole combined call",
        }
        items_report[key] = item_scores
        r = report["results"][key]
        print(f"{key:24s} acc {r['acc_mean']:6.2f} ± {r['acc_std']:.2f}  (n={len(items)}, gen {r['mean_gen_tokens']:.0f} tok, "
              f"trunc {100 * r['truncated_frac']:.1f}%)", flush=True)
    out_path.write_text(json.dumps(report, indent=2))
    if args.items_out:
        ip = resolve(args.items_out)
        prev = json.loads(ip.read_text()) if ip.exists() else {}
        ip.write_text(json.dumps({**prev, **items_report}))
    print(f"total {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
