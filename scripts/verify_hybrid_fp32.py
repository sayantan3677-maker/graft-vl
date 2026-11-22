"""FP32 text-only equivalence check of a built hybrid vs its source text-only Qwen3 LM.

Loads the saved hybrid (exports/<name>, Qwen3-VL architecture) and the original Qwen3 in FP32 and
compares logits on text-only prompts (raw and chat-templated, thinking and non-thinking). The two
vision boundary rows (<|vision_start|>/<|vision_end|>) are excluded (deliberately borrowed from
Qwen3-VL). Also checks greedy continuations match.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/verify_hybrid_fp32.py --hybrid exports/hybrid-q3-4b-vit4b [--llm Qwen/Qwen3-4B]
"""

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VS, VE = 151652, 151653

TEXTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:",
    "Solve: 17 * 23 =",
    "In 1905, Albert Einstein published four papers that changed physics. The first one explained " * 8,
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", required=True)
    ap.add_argument("--llm")
    ap.add_argument("--out")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3VLForConditionalGeneration

    hdir = ROOT / args.hybrid
    llm = args.llm or json.loads((hdir / "hybrid_info.json").read_text())["language_model_from"]
    tok = AutoTokenizer.from_pretrained(llm)
    h = Qwen3VLForConditionalGeneration.from_pretrained(hdir, dtype=torch.float32, attn_implementation="eager").cuda().eval()
    q = AutoModelForCausalLM.from_pretrained(llm, dtype=torch.float32, attn_implementation="eager").cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prompts = list(TEXTS)
    for think in (True, False):
        prompts.append(tok.apply_chat_template([{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=think))
    rows, worst = [], 0.0
    for t in prompts:
        ids = torch.tensor([tok(t, add_special_tokens=False)["input_ids"]], device="cuda")
        a, b = h(input_ids=ids).logits, q(input_ids=ids).logits
        keep = torch.ones(a.shape[-1], dtype=torch.bool, device="cuda")
        keep[[VS, VE]] = False
        d = (a[..., keep] - b[..., keep]).abs().max().item()
        ga = h.generate(input_ids=ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1]:].tolist()
        gb = q.generate(input_ids=ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1]:].tolist()
        rows.append({"len": ids.shape[1], "max_abs_dlogit": d, "greedy_equal": ga == gb})
        worst = max(worst, d)
    tied = (h.lm_head.weight.data_ptr() == h.get_input_embeddings().weight.data_ptr())
    res = {"hybrid": args.hybrid, "llm": llm, "dtype": "float32", "max_abs_dlogit": worst, "rows": rows,
           "hybrid_lm_head_tied": tied, "hybrid_tie_cfg": h.config.text_config.tie_word_embeddings,
           "llm_tie_cfg": q.config.tie_word_embeddings,
           "rope_theta_hybrid": h.config.text_config.rope_parameters.get("rope_theta")}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
