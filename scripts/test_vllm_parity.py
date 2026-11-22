"""Serving parity: HF training wrapper vs. vLLM plugin on the same images and prompts (roadmap §12.3).

Checks, per sample:
  1. prompt token ids built by vLLM (after image-placeholder expansion) == our VLMProcessor ids
  2. greedy continuation agreement (exact-prefix length out of N tokens)
  3. at the first divergence, whether it is a near-tie in vLLM's own distribution
     (|logprob(vLLM top-1) - logprob(HF token)| < NEAR_TIE nats) → BF16 kernel noise, or a real gap → bug.
Pass = all prompt ids equal AND every divergence is a near-tie.

Runs in the `vlm-eval` env (has both torch/transformers and vLLM).

Usage:
    CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/vlm-eval/bin/python scripts/test_vllm_parity.py \
        --export exports/<serving-dir> --run experiments/<run_id> [--step N] --eval-data manifests/s1-eval300
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_s1 import load_eval_samples, load_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from vlm.data.processor import ProcessorConfig, VLMProcessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--eval-data", default="manifests/s1-eval300")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--new-tokens", type=int, default=48)
    args = ap.parse_args()

    samples = load_eval_samples(ROOT / args.eval_data, None, 300)
    by_type: dict[str, list] = {}
    for s in samples:
        by_type.setdefault(s["meta"]["image_type"], []).append(s)
    picked = [s for t in sorted(by_type) for s in by_type[t][: args.n // len(by_type)]]
    questions = [s["messages"][0]["content"][-1]["text"] for s in picked]

    # ---- HF reference
    run_dir = ROOT / args.run
    model, cfg, ckpt = load_model(run_dir, args.step, "cuda")
    tok = AutoTokenizer.from_pretrained(cfg["model"]["llm_name"])
    proc = VLMProcessor(tok, ProcessorConfig(tile_size=model.vision.image_size, **cfg["processor"]))
    ref = []
    for s, q in zip(picked, questions):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        p = proc.prompt(msgs, s["images"], enable_thinking=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(p["input_ids"][None].cuda(), p["pixel_values"].cuda(), max_new_tokens=args.new_tokens, stop_ids=())
        ref.append({"prompt_ids": p["input_ids"].tolist(), "gen": out})
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(ROOT / args.export), dtype="bfloat16", gpu_memory_utilization=0.35, max_model_len=8192,
              limit_mm_per_prompt={"image": 1}, seed=0)
    chats = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": s["images"][0]}, {"type": "text", "text": q}]}]
             for s, q in zip(picked, questions)]
    outs = llm.chat(chats, SamplingParams(temperature=0.0, max_tokens=args.new_tokens, ignore_eos=True, logprobs=5),
                    chat_template_kwargs={"enable_thinking": False})

    NEAR_TIE = 0.5
    rows, prompt_ok, agree, ties = [], 0, [], []
    for s, r, o in zip(picked, ref, outs):
        same_prompt = list(o.prompt_token_ids) == r["prompt_ids"]
        v = list(o.outputs[0].token_ids)
        prefix = next((i for i, (a, b) in enumerate(zip(r["gen"], v)) if a != b), min(len(r["gen"]), len(v)))
        prompt_ok += same_prompt
        agree.append(prefix)
        gap = None
        if prefix < min(len(r["gen"]), len(v)):
            step_lp = o.outputs[0].logprobs[prefix]  # {token_id: Logprob} for vLLM's top-5 at the divergence
            top_lp = step_lp[v[prefix]].logprob
            hf_tok = r["gen"][prefix]
            gap = top_lp - step_lp[hf_tok].logprob if hf_tok in step_lp else float("inf")
            ties.append(gap < NEAR_TIE)
        rows.append({"key": s["meta"]["key"], "type": s["meta"]["image_type"], "prompt_ids_equal": same_prompt,
                     "prompt_len_hf": len(r["prompt_ids"]), "prompt_len_vllm": len(o.prompt_token_ids),
                     "greedy_prefix_agree": prefix, "divergence_gap_nats": gap, "hf": tok.decode(r["gen"])[:160], "vllm": tok.decode(v)[:160]})
    report = {"export": args.export, "checkpoint": ckpt, "n": len(rows), "prompt_ids_equal": prompt_ok,
              "greedy_prefix_mean": sum(agree) / len(agree), "greedy_full_match": sum(a >= args.new_tokens for a in agree),
              "divergences": len(ties), "near_tie_divergences": sum(ties),
              "new_tokens": args.new_tokens, "rows": rows}
    out = ROOT / "reports" / "serving" / f"parity-{Path(args.export).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    for r in rows[:6]:
        print(f"[{r['type']}] prompt_equal={r['prompt_ids_equal']} agree={r['greedy_prefix_agree']}\n  HF:   {r['hf'][:120]}\n  vLLM: {r['vllm'][:120]}")
    ok = prompt_ok == len(rows) and all(ties)
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
