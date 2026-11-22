"""Memory-safe causal-LM losses.

Qwen3's vocabulary is 151,936 tokens, so materializing FP32 logits for long
sequences costs ~5 GB at 8K tokens and ~10 GB at 16K. Both functions here take
final hidden states plus the LM-head weight and never build the full
[tokens, vocab] tensor.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _ce_sum(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = (hidden @ weight.t()).float()
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Mean CE over non-ignored tokens, computed chunk by chunk.

    Each chunk is recomputed in backward (activation checkpointing), so peak
    memory is one chunk of logits instead of the whole sequence.

    hidden: [N, H] already shifted so that hidden[i] predicts labels[i].
    weight: [V, H] LM-head weight.
    labels: [N] with IGNORE_INDEX for masked positions.
    """
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    n_valid = (labels != IGNORE_INDEX).sum()
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[0], chunk_size):
        h = hidden[start : start + chunk_size]
        y = labels[start : start + chunk_size]
        if (y != IGNORE_INDEX).any():
            total = total + checkpoint(_ce_sum, h, weight, y, use_reentrant=False)
    return total / n_valid.clamp(min=1)


def fused_linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Liger fused linear CE when available and working, else chunked fallback."""
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except ImportError:
        return chunked_linear_cross_entropy(hidden, weight, labels)
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX)
    return loss_fn(weight, hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1))


def shift_for_causal_lm(hidden: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden [B, T, H], labels [B, T] -> position t predicts token t+1."""
    return hidden[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
