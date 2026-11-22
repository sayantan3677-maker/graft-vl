"""CPU-only checks of the Qwen3 facts the roadmap depends on (§2.1, §3.3, §4.5).

- hidden size, tied embeddings, vocab size
- special tokens exist as single ids; their embedding-row norms vs. the matrix
- chat template: non-thinking inserts an empty think block; earlier assistant
  turns have their <think> content stripped

Usage: python scripts/check_qwen_assumptions.py Qwen/Qwen3-1.7B [Qwen/Qwen3-4B ...]
"""

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|box_start|>", "<|box_end|>", "<think>", "</think>"]


def embedding_matrix(model_dir: Path) -> torch.Tensor:
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(f, "pt") as st:
            if "model.embed_tokens.weight" in st.keys():
                return st.get_tensor("model.embed_tokens.weight").float()
    raise FileNotFoundError("model.embed_tokens.weight not found")


def check(model_id: str) -> dict:
    model_dir = Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    cfg = AutoConfig.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    E = embedding_matrix(model_dir)
    norms = E.norm(dim=-1)
    trained = norms[: len(tok) - len(tok.added_tokens_encoder)]  # regular vocab rows

    rows = {}
    for t in SPECIAL:
        ids = tok.encode(t, add_special_tokens=False)
        tid = ids[0]
        n = norms[tid].item()
        rows[t] = {
            "single_id": len(ids) == 1,
            "id": tid,
            "norm": round(n, 4),
            "z_vs_vocab": round((n - trained.mean().item()) / trained.std().item(), 2),
        }

    msgs = [{"role": "user", "content": "Hi"}]
    no_think = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    think = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    multi = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<think>\nSECRET_REASONING\n</think>\n\nA1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<think>\nFINAL_REASONING\n</think>\n\nA2"},
    ]
    multi_txt = tok.apply_chat_template(multi, tokenize=False)

    return {
        "model": model_id,
        "revision": model_dir.name,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "embedding_rows": E.shape[0],
        "vocab_norm_mean": round(trained.mean().item(), 4),
        "vocab_norm_std": round(trained.std().item(), 4),
        "special_tokens": rows,
        "template": {
            "non_thinking_generation_prompt_tail": no_think[-40:],
            "non_thinking_inserts_empty_think": "<think>\n\n</think>" in no_think,
            "thinking_generation_prompt_tail": think[-40:],
            "earlier_turn_reasoning_stripped": "SECRET_REASONING" not in multi_txt,
            "final_turn_reasoning_kept": "FINAL_REASONING" in multi_txt,
        },
    }


def main():
    out_dir = ROOT / "reports" / "day0"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_id in sys.argv[1:] or ["Qwen/Qwen3-1.7B"]:
        res = check(model_id)
        print(json.dumps(res, indent=2))
        (out_dir / f"qwen_assumptions-{model_id.split('/')[-1]}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
