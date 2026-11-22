"""Batch processed samples: padded batches or packed rows (roadmap §4, §1.3)."""

import torch

from ..training.losses import IGNORE_INDEX
from .packing import block_causal_mask, pack_samples


def _concat_pixels(samples: list[dict]) -> torch.Tensor | None:
    pixels = [s["pixel_values"] for s in samples if s["pixel_values"] is not None]
    return torch.cat(pixels) if pixels else None


def pad_collate(samples: list[dict], pad_id: int) -> dict:
    """Right-padded batch. Tiles are concatenated in sample order, matching <|image_pad|> order."""
    max_len = max(len(s["input_ids"]) for s in samples)
    n = len(samples)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    labels = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, s in enumerate(samples):
        t = len(s["input_ids"])
        input_ids[i, :t] = s["input_ids"]
        labels[i, :t] = s["labels"]
        attention_mask[i, :t] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "pixel_values": _concat_pixels(samples)}


def packed_collate(samples: list[dict], pad_id: int, mask_dtype: torch.dtype | None = torch.bool) -> dict:
    """One packed row with per-sample position ids and sample lengths.

    mask_dtype=None skips the dense [T, T] mask; the trainer then builds a sparse
    FlexAttention BlockMask from `seq_lens` on the GPU (much faster, see §1.4).
    """
    packed = pack_samples(samples, pad_id=pad_id)
    return {
        "input_ids": packed["input_ids"],
        "labels": packed["labels"],
        "position_ids": packed["position_ids"],
        "seq_lens": packed["seq_lens"],
        "attention_mask": None if mask_dtype is None else block_causal_mask(packed["seq_lens"], mask_dtype),
        "pixel_values": _concat_pixels(samples),
    }
