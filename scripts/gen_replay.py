"""Generate self-distilled text replay with the ORIGINAL Qwen3 (roadmap §4.8). Runs in `vlm-eval`.

Prompts: a stratified sample of first user turns from allenai/tulu-3-sft-mixture (ODC-BY),
grouped by source into coding / math / reasoning / instruction-following / writing / chat.
Each prompt is answered once, alternately in thinking and non-thinking mode, with the
model-card sampling settings. Truncated generations and empty answers are dropped.
Output is a text-only WebDataset manifest usable by the S2 loader.

Usage:
    CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/vlm-eval/bin/python scripts/gen_replay.py \
        --model Qwen/Qwen3-1.7B --n 20000 --name replay-q3-1b7-20k
"""

import argparse
import os
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# FlashInfer's JIT sampler misdetects sm_120 (Blackwell) as unsupported; use the PyTorch sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(ROOT / "src"))

SAMPLING = {
    True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
}

# tulu-3-sft-mixture `source` substrings → replay category, with target shares.
CATEGORIES = {
    "coding": (["code", "Code", "evol_codealpaca"], 0.18),
    "math": (["math", "Math", "gsm", "numina"], 0.22),
    "instruction_following": (["ifdata", "persona_ifeval", "instruction_following"], 0.18),
    "safety": (["wildguard", "wildjailbreak", "coconot"], 0.08),  # refusal / over-refusal behavior
    "reasoning": (["flan", "sciriff", "table_gpt", "aya"], 0.10),
    "writing_chat": (["wildchat", "oasst", "no_robots", "hard_coded"], 0.24),
}


def categorize(source: str) -> str | None:
    for cat, (keys, _) in CATEGORIES.items():
        if any(k in source for k in keys):
            return cat
    return None


def split_thinking(text: str) -> tuple[str | None, str]:
    m = re.match(r"\s*<think>\n?(.*?)\n?</think>\s*(.*)", text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--think-frac", type=float, default=0.5)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=6144)  # prompt + answer stays within the 8K replay cap
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--categories", nargs="*", help="generate only these categories (top-ups); shares renormalized")
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    pools: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    seen = set()
    for r in ds.shuffle(seed=args.seed).select(range(min(len(ds), 600_000))):
        cat = categorize(r["source"])
        first = next((m["content"] for m in r["messages"] if m["role"] == "user"), None)
        if cat is None or not first or len(first) < 10:
            continue
        h = hashlib.sha1(first.strip().lower().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        pools[cat].append(first.strip())
    # Oversample prompts by 15% to absorb drops (truncation, empty answers).
    cats = args.categories or list(CATEGORIES)
    total_share = sum(CATEGORIES[c][1] for c in cats)
    want = {c: int(CATEGORIES[c][1] / total_share * args.n * 1.15) for c in cats}
    chosen = [(c, p) for c in cats for p in rng.sample(pools[c], min(want[c], len(pools[c])))]
    rng.shuffle(chosen)
    print("pool sizes:", {c: len(v) for c, v in pools.items()}, "| chosen:", Counter(c for c, _ in chosen))

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_prompt_tokens + args.max_new_tokens + 64, seed=args.seed)
    tok = llm.get_tokenizer()

    jobs = []
    for i, (cat, prompt) in enumerate(chosen):
        thinking = rng.random() < args.think_frac
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=thinking)
        if len(tok(text)["input_ids"]) > args.max_prompt_tokens:
            continue
        jobs.append({"cat": cat, "prompt": prompt, "thinking": thinking, "text": text})

    outs = {}
    for thinking in (True, False):
        idx = [i for i, j in enumerate(jobs) if j["thinking"] == thinking]
        params = SamplingParams(max_tokens=args.max_new_tokens, seed=args.seed, **SAMPLING[thinking])
        for i, o in zip(idx, llm.generate([jobs[i]["text"] for i in idx], params)):
            outs[i] = o.outputs[0]

    import webdataset as wds

    out_dir = ROOT / "manifests" / args.name
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    writer = wds.ShardWriter(str(out_dir / "shards" / "shard-%06d.tar"), maxcount=2000, verbose=0)
    records, drops = [], Counter()
    for i, j in enumerate(jobs):
        if len(records) >= args.n:
            break
        o = outs[i]
        if o.finish_reason == "length":
            drops["truncated"] += 1
            continue
        # With enable_thinking=True the completion starts with the reasoning; add the opening tag back.
        reasoning, answer = split_thinking(("<think>\n" + o.text) if j["thinking"] and "<think>" not in o.text[:20] else o.text)
        if j["thinking"] and not reasoning:
            drops["no_reasoning"] += 1
            continue
        if not answer:
            drops["empty_answer"] += 1
            continue
        final = {"role": "assistant", "content": answer}
        if j["thinking"]:
            final["reasoning"] = reasoning
        key = f"replay-{len(records):08d}"
        meta = {"key": key, "source": "tulu3_prompts+self_generated", "task": j["cat"], "thinking": j["thinking"],
                "teacher": args.model, "image_type": "text", "origin": "self_distilled",
                "license": "prompts ODC-BY; responses generated by Qwen3 (Apache-2.0)", "commercial_ok": None,
                "n_gen_tokens": len(o.token_ids), "split": "train"}
        writer.write({"__key__": key, "json": json.dumps({"messages": [{"role": "user", "content": j["prompt"]}, final],
                                                          "meta": meta}).encode()})
        records.append(meta)
    writer.close()
    with open(out_dir / "records.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    manifest = {"name": args.name, "teacher": args.model, "written": len(records), "drops": dict(drops),
                "by_task": dict(Counter(r["task"] for r in records)),
                "thinking_frac": sum(r["thinking"] for r in records) / max(1, len(records)),
                "mean_gen_tokens": sum(r["n_gen_tokens"] for r in records) / max(1, len(records))}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
