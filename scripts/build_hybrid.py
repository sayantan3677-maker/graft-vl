"""Build the hybrid: Qwen3-VL architecture + Qwen3-VL ViT/merger/DeepStack + TEXT-ONLY Qwen3 language model.

The language model is the original text-only Qwen3 (all weights), with RoPE θ kept at Qwen3's
value; interleaved MRoPE gives text tokens equal (t, h, w) positions, which reduces exactly to
Qwen3's 1D RoPE, so text-only behaviour must be unchanged (verified here by logits).
Only the two vision boundary rows are borrowed from Qwen3-VL (they are untrained in Qwen3).

Usage: python scripts/build_hybrid.py --vl Qwen/Qwen3-VL-2B-Instruct --llm Qwen/Qwen3-1.7B --out exports/hybrid-q3-1b7
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VS, VE = 151652, 151653


def build(vl_name: str, llm_name: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, Qwen3VLForConditionalGeneration

    vl = Qwen3VLForConditionalGeneration.from_pretrained(vl_name, dtype=dtype)
    q = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dtype)
    cfg = copy.deepcopy(vl.config)
    theta = q.config.rope_theta if getattr(q.config, "rope_theta", None) else q.config.rope_parameters["rope_theta"]
    for field in ("rope_parameters", "rope_scaling"):
        if isinstance(getattr(cfg.text_config, field, None), dict):
            getattr(cfg.text_config, field)["rope_theta"] = theta
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"):
        assert getattr(cfg.text_config, k) == getattr(q.config, k), f"text config mismatch on {k}"

    hybrid = Qwen3VLForConditionalGeneration(cfg).to(dtype)
    state = {k: v for k, v in vl.state_dict().items() if "visual" in k}
    for k, v in q.state_dict().items():
        state["lm_head.weight" if k == "lm_head.weight" else k.replace("model.", "model.language_model.", 1)] = v
    missing, unexpected = hybrid.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "lm_head.weight" or not cfg.text_config.tie_word_embeddings]
    assert not unexpected and not missing, (missing[:5], unexpected[:5])
    if cfg.text_config.tie_word_embeddings:
        hybrid.tie_weights()
    with torch.no_grad():
        E = hybrid.get_input_embeddings().weight
        E[[VS, VE]] = vl.get_input_embeddings().weight[[VS, VE]]
    return hybrid.eval(), q.eval()


@torch.no_grad()
def text_equivalence(hybrid, q, tok, device="cuda"):
    texts = ["The capital of France is", "def fibonacci(n):\n    if n < 2:", "Solve: 17 * 23 ="]
    worst = 0.0
    for t in texts:
        ids = torch.tensor([tok(t)["input_ids"]], device=device)
        a = hybrid(input_ids=ids).logits.float()
        b = q(input_ids=ids).logits.float()
        keep = torch.ones(a.shape[-1], dtype=torch.bool, device=device)
        keep[[VS, VE]] = False  # deliberately changed rows
        worst = max(worst, (a[..., keep] - b[..., keep]).abs().max().item())
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoProcessor, AutoTokenizer

    hybrid, q = build(args.vl, args.llm)
    tok = AutoTokenizer.from_pretrained(args.llm)
    hybrid.cuda(), q.cuda()
    diff = text_equivalence(hybrid, q, tok)
    print(f"text-only max |Δlogit| vs {args.llm}: {diff:.4f} (bf16)")
    out = ROOT / args.out
    hybrid.save_pretrained(out, safe_serialization=True)
    proc = AutoProcessor.from_pretrained(args.vl)
    proc.tokenizer = tok  # Qwen3 tokenizer + its hybrid-thinking chat template
    proc.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "hybrid_info.json").write_text(json.dumps({"vision_from": args.vl, "language_model_from": args.llm,
                                                      "rope_theta": hybrid.config.text_config.rope_parameters.get("rope_theta"),
                                                      "text_only_max_abs_logit_diff_bf16": diff}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
