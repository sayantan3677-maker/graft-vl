"""Noise-aware text evaluation with vLLM (roadmap §7.1–7.2). Runs in the `vlm-eval` env.

For every task × thinking mode it samples `--seeds` completions per question with the
Qwen3 model-card sampling settings, and reports accuracy mean/std across seeds.

Usage:
    CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/vlm-eval/bin/python scripts/eval_text.py \
        --model Qwen/Qwen3-1.7B --tasks mmlu_pro gsm8k ifeval math500 aime24 --seeds 4 \
        --limit mmlu_pro=1000 --out reports/text/qwen3-1.7b-base.json
"""

import argparse
import os
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# FlashInfer's JIT sampler misdetects sm_120 (Blackwell) as unsupported; use the PyTorch sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(ROOT / "src"))

from vlm.eval.text_tasks import TASKS  # noqa: E402

# Qwen3 model card recommendations.
SAMPLING = {
    True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local Qwen3ForCausalLM dir (e.g. exported VLM LLM)")
    ap.add_argument("--tokenizer")
    ap.add_argument("--tasks", nargs="+", default=["mmlu_pro", "gsm8k", "ifeval", "math500", "aime24"])
    ap.add_argument("--modes", nargs="+", default=["think", "nothink"], choices=["think", "nothink"])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--limit", nargs="*", default=[], help="task=N subsets, e.g. mmlu_pro=1000")
    ap.add_argument("--max-tokens-think", type=int, default=32768)
    ap.add_argument("--max-tokens-nothink", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=36864)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items-out", help="optional JSON with per-item per-seed scores (for paired comparisons)")
    args = ap.parse_args()
    limits = {k: int(v) for k, v in (x.split("=") for x in args.limit)}

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, seed=0)
    tok = llm.get_tokenizer()

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(out_path.read_text()) if out_path.exists() else {"model": args.model, "results": {}}

    for task_name in args.tasks:
        task = TASKS[task_name]
        items = task.load(limits.get(task_name), 0)
        for mode in args.modes:
            thinking = mode == "think"
            key = f"{task_name}/{mode}"
            prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=thinking) for it in items]
            params = SamplingParams(n=args.seeds, seed=1234,
                                    max_tokens=args.max_tokens_think if thinking else args.max_tokens_nothink,
                                    **SAMPLING[thinking])
            t0 = time.time()
            outputs = llm.generate(prompts, params)
            per_seed = [[] for _ in range(args.seeds)]
            item_scores = {}
            lengths, truncated = [], 0
            for it, out in zip(items, outputs):
                for s, comp in enumerate(out.outputs):
                    per_seed[s].append(task.score(it, comp.text))
                    item_scores.setdefault(str(it["id"]), []).append(per_seed[s][-1])
                    lengths.append(len(comp.token_ids))
                    truncated += comp.finish_reason == "length"
            accs = [100 * sum(x) / len(x) for x in per_seed]
            report["results"][key] = {
                "n_items": len(items), "seeds": args.seeds, "acc_mean": statistics.mean(accs),
                "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0, "acc_per_seed": accs,
                "mean_gen_tokens": statistics.mean(lengths), "truncated_frac": truncated / len(lengths),
                "sampling": SAMPLING[thinking], "max_tokens": params.max_tokens, "minutes": (time.time() - t0) / 60,
            }
            r = report["results"][key]
            print(f"{key:24s} acc {r['acc_mean']:6.2f} ± {r['acc_std']:.2f}  (n={len(items)}, gen {r['mean_gen_tokens']:.0f} tok, "
                  f"trunc {100 * r['truncated_frac']:.1f}%, {r['minutes']:.1f} min)", flush=True)
            out_path.write_text(json.dumps(report, indent=2))
            if args.items_out:
                ip = Path(args.items_out) if Path(args.items_out).is_absolute() else ROOT / args.items_out
                items_report = json.loads(ip.read_text()) if ip.exists() else {}
                items_report[key] = item_scores
                ip.write_text(json.dumps(items_report))


if __name__ == "__main__":
    main()
