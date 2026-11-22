"""Serving parity for the hybrid (stock Qwen3VLForConditionalGeneration checkpoint): HF transformers vs. vLLM.

Same export directory, same images, same prompts (roadmap §12.3):
  1. prompt token ids: HF (chat template on a string prompt with the training-format image placeholder +
     Qwen3VLProcessor expansion, exactly as scripts/eval_hybrid.py generate()) vs. vLLM's chat path
     (OpenAI-style content list [image, text], enable_thinking=False via chat_template_kwargs) → must be equal.
  2. greedy continuation agreement (exact-prefix length).
  3. at the first divergent token: gap = logprob(own top-1) - logprob(other engine's token), in both engines'
     distributions. gap < NEAR_TIE nats → BF16 numerics; larger → a real bug.
Two prompts per image: the short-answer QA prompt used by eval_hybrid.py and a free-form description
(longer generations). Images: --n images from manifests/eval-official, round-robin over sources/image types.

Runs in the vlm-eval env (vLLM + transformers):
  flock locks/gpu1-slotB.lock env CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
      ~/.conda/envs/vlm-eval/bin/python scripts/test_hybrid_vllm_parity.py --export exports/serve-hyb-1b7-s2000
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
NEAR_TIE = 0.5
DESCRIBE = "Describe this image in detail."
SHORT = "\nAnswer the question using a single word or phrase."


def pick_samples(n: int, offset: int):
    from eval_s1 import load_eval_samples

    samples = load_eval_samples(ROOT / "manifests/eval-official", None, 1400)
    by: dict[str, list] = {}
    for s in samples:
        by.setdefault(s["meta"]["source"], []).append(s)
    srcs, picked, i = sorted(by), [], 0
    while len(picked) < n:
        src = srcs[i % len(srcs)]
        picked.append(by[src][offset + i // len(srcs)])
        i += 1
    return picked


@torch.no_grad()
def teacher_forced(model, cases, ref):
    """Log-probs of HF-BF16's greedy tokens (and the argmax) at every generated position, teacher-forced."""
    res = []
    for c, r in zip(cases, ref):
        inp = {k: v.cuda() for k, v in c["inputs"].items()}
        gen = torch.tensor([r["gen"]], device="cuda")
        ids = torch.cat([inp["input_ids"], gen], 1)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        for k, v in list(kw.items()):  # per-token side inputs (e.g. mm_token_type_ids): text type for generated tokens
            if v.dim() == 2 and v.shape == inp["input_ids"].shape:
                kw[k] = torch.cat([v, torch.zeros_like(gen)], 1)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), **kw).logits[0, inp["input_ids"].shape[1] - 1:-1].float()
        lp = torch.log_softmax(logits, -1)
        res.append({"lp": lp.gather(1, gen[0][:, None])[:, 0].cpu().tolist(), "argmax": lp.argmax(-1).cpu().tolist()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="exports/serve-hyb-1b7-s2000")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--offset", type=int, default=100, help="index into each source (avoid the QA-eval head)")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    export = ROOT / args.export
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(export)
    tok = proc.tokenizer
    max_pixels = proc.image_processor.size["longest_edge"]
    picked = pick_samples(args.n, args.offset)
    cases = []
    for s in picked:
        q = s["messages"][0]["content"][-1]["text"]
        for kind, text in (("short", q + SHORT), ("describe", DESCRIBE)):
            cases.append({"s": s, "kind": kind, "text": text, "image": s["images"][0].convert("RGB")})

    # ---- HF reference (as eval_hybrid.generate, greedy, with scores)
    model = Qwen3VLForConditionalGeneration.from_pretrained(export, dtype=torch.bfloat16).cuda().eval()
    ref = []
    with torch.no_grad():
        for c in cases:
            prompt = tok.apply_chat_template([{"role": "user", "content": IMAGE + c["text"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            inp = proc(text=[prompt], images=[c["image"]], size={"longest_edge": max_pixels, "shortest_edge": 65536},
                       return_tensors="pt").to("cuda")
            out = model.generate(**inp, max_new_tokens=args.new_tokens, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True)
            gen = out.sequences[0, inp["input_ids"].shape[1]:].tolist()
            lps = [torch.log_softmax(sc[0].float(), -1).cpu() for sc in out.scores]
            ref.append({"prompt_ids": inp["input_ids"][0].tolist(), "gen": gen, "lps": lps, "prompt": prompt})
            c["inputs"] = {k: v.cpu() for k, v in inp.items()}
    tf = {"hf_bf16": teacher_forced(model, cases, ref)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # FP32 reference = noise floor yardstick: is vLLM-BF16 as close to FP32 as HF-BF16 is?
    model = Qwen3VLForConditionalGeneration.from_pretrained(export, dtype=torch.float32).cuda().eval()
    tf["hf_fp32"] = teacher_forced(model, cases, ref)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- vLLM, OpenAI-style content lists through the chat path
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(export), dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=16384,
              limit_mm_per_prompt={"image": 1}, seed=0)
    chats = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": c["image"]}, {"type": "text", "text": c["text"]}]}]
             for c in cases]
    sp = SamplingParams(temperature=0.0, max_tokens=args.new_tokens, logprobs=20)
    outs = llm.chat(chats, sp, chat_template_kwargs={"enable_thinking": False})

    # teacher-forced vLLM scoring of HF-BF16's sequences (prompt_logprobs over the generated span)
    tf_prompts = [{"prompt": r["prompt"] + tok.decode(r["gen"]), "multi_modal_data": {"image": c["image"]}} for c, r in zip(cases, ref)]
    tf_outs = llm.generate(tf_prompts, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1))
    tf["vllm_bf16"] = []
    for r, o in zip(ref, tf_outs):
        ids = list(o.prompt_token_ids)
        if ids != r["prompt_ids"] + r["gen"]:  # decode/re-encode did not round-trip; skip
            tf["vllm_bf16"].append(None)
            continue
        P = len(r["prompt_ids"])
        plp = o.prompt_logprobs[P:]
        tf["vllm_bf16"].append({"lp": [d[t].logprob for d, t in zip(plp, r["gen"])],
                                "argmax": [next(k for k, v in d.items() if v.rank == 1) for d in plp]})

    # chat-template checks: thinking default ON (no empty think block) and the original Qwen3 text template
    c0 = chats[0]
    think_on = llm.chat([c0], SamplingParams(temperature=0.0, max_tokens=1))[0]
    orig_tpl = (export / "chat_template.qwen3_text_original.jinja")
    orig = None
    if orig_tpl.exists():
        o = llm.chat([c0], SamplingParams(temperature=0.0, max_tokens=1), chat_template=orig_tpl.read_text(),
                     chat_template_kwargs={"enable_thinking": False})[0]
        orig = {"prompt_ids_equal_hf": list(o.prompt_token_ids) == ref[0]["prompt_ids"],
                "prompt_text_head": tok.decode([t for t in o.prompt_token_ids if t != 151655])[:160]}
    template_checks = {
        "nothink_has_empty_think_block": "<think>\n\n</think>" in tok.decode(outs[0].prompt_token_ids),
        "default_thinking_prompt_ends": repr(tok.decode(think_on.prompt_token_ids[-4:])),
        "default_thinking_has_no_empty_think_block": "<think>\n\n</think>" not in tok.decode(think_on.prompt_token_ids),
        "original_qwen3_template_via_vllm": orig,
    }

    rows, ties = [], []
    for c, r, o in zip(cases, ref, outs):
        v = list(o.outputs[0].token_ids)
        h = r["gen"]
        same_prompt = list(o.prompt_token_ids) == r["prompt_ids"]
        n = min(len(h), len(v))
        prefix = next((i for i in range(n) if h[i] != v[i]), n)
        full = h == v
        gap_v = gap_h = None
        if not full and prefix < n:
            lp = o.outputs[0].logprobs[prefix]
            gap_v = lp[v[prefix]].logprob - lp[h[prefix]].logprob if h[prefix] in lp else float("inf")
            hl = r["lps"][prefix]
            gap_h = float(hl[h[prefix]] - hl[v[prefix]])
            ties.append(max(gap_v, gap_h) < NEAR_TIE)
        elif not full:  # one sequence is a strict prefix of the other (EOS on one side)
            ties.append(False)
        rows.append({"key": c["s"]["meta"]["key"], "source": c["s"]["meta"]["source"], "type": c["s"]["meta"]["image_type"],
                     "kind": c["kind"], "prompt_ids_equal": same_prompt, "prompt_len": len(r["prompt_ids"]),
                     "prompt_len_vllm": len(o.prompt_token_ids), "hf_len": len(h), "vllm_len": len(v),
                     "greedy_prefix_agree": prefix, "full_match": full,
                     "gap_nats_vllm": None if gap_v is None else round(gap_v, 4),
                     "gap_nats_hf": None if gap_h is None else round(gap_h, 4),
                     "hf": tok.decode(h, skip_special_tokens=True)[:200], "vllm": tok.decode(v, skip_special_tokens=True)[:200]})
    prompt_ok = sum(r["prompt_ids_equal"] for r in rows)
    report = {"export": args.export, "max_pixels": max_pixels, "n_images": len(picked), "n_cases": len(rows),
              "prompt_ids_equal": prompt_ok, "full_greedy_match": sum(r["full_match"] for r in rows),
              "full_greedy_match_short": sum(r["full_match"] for r in rows if r["kind"] == "short"),
              "full_greedy_match_describe": sum(r["full_match"] for r in rows if r["kind"] == "describe"),
              "divergences": len(ties), "near_tie_divergences": sum(ties),
              "max_gap_nats": max([max(r["gap_nats_vllm"], r["gap_nats_hf"]) for r in rows if r["gap_nats_vllm"] is not None], default=None),
              "mean_prefix_agree_describe": sum(r["greedy_prefix_agree"] for r in rows if r["kind"] == "describe") / len(picked),
              "new_tokens": args.new_tokens, "near_tie_threshold": NEAR_TIE, "template_checks": template_checks, "rows": rows}
    def tf_stats(name):
        d, agree, n, used = [], 0, 0, 0
        for i, r in enumerate(ref):
            a, b = tf[name][i], tf["hf_fp32"][i]
            if a is None:
                continue
            used += 1
            d += [abs(x - y) for x, y in zip(a["lp"], b["lp"])]
            agree += sum(x == y for x, y in zip(a["argmax"], b["argmax"]))
            n += len(b["argmax"])
        d.sort()
        return {"cases": used, "tokens": n, "mean_abs_dlogprob_vs_fp32": round(sum(d) / len(d), 4),
                "p99_abs_dlogprob_vs_fp32": round(d[int(0.99 * (len(d) - 1))], 4), "max_abs_dlogprob_vs_fp32": round(d[-1], 4),
                "top1_agree_with_fp32": round(agree / n, 4)}
    report["teacher_forced"] = {k: tf_stats(k) for k in ("hf_bf16", "vllm_bf16")}
    report["pass_strict"] = prompt_ok == len(rows) and all(ties)
    # numerics criterion: identical prompts AND vLLM-BF16 no further from FP32 than HF-BF16 (within 1.5x)
    th, tv = report["teacher_forced"]["hf_bf16"], report["teacher_forced"]["vllm_bf16"]
    report["pass_numerics"] = (prompt_ok == len(rows) and tv["mean_abs_dlogprob_vs_fp32"] <= 1.5 * th["mean_abs_dlogprob_vs_fp32"]
                               and tv["top1_agree_with_fp32"] >= th["top1_agree_with_fp32"] - 0.01)
    report["pass"] = report["pass_strict"] or report["pass_numerics"]
    out = Path(args.out) if args.out else ROOT / "reports/serving" / f"parity-{export.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    for r in rows:
        if not r["full_match"]:
            print(f"[{r['source']}/{r['kind']}] agree={r['greedy_prefix_agree']} gap_v={r['gap_nats_vllm']} gap_h={r['gap_nats_hf']}\n"
                  f"  HF:   {r['hf'][:150]!r}\n  vLLM: {r['vllm'][:150]!r}")
    print("PARITY", "PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
