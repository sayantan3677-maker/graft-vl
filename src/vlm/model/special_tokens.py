"""Learnable deltas for Qwen3's untrained special-token rows (roadmap §3.3).

In Qwen3, <|vision_start|>, <|vision_end|>, <|image_pad|>, <|box_start|> and
<|box_end|> share one bit-identical, untrained embedding row. We keep the
embedding matrix frozen and learn a delta for selected rows:

    E_eff[id] = E[id] + delta[id]

The same deltas are applied to the output projection, so with tied embeddings
(1.7B/4B) input and output stay consistent. For untied models (8B) the output
rows get their own delta. `merge_into` writes the deltas into the weights for
export, after which serving needs no special handling.
"""

import torch
from torch import nn

QWEN3_VISION_START = 151652
QWEN3_VISION_END = 151653
QWEN3_IMAGE_PAD = 151655
QWEN3_BOX_START = 151648
QWEN3_BOX_END = 151649
QWEN3_N_REGULAR = 151643  # ids below this are ordinary BPE tokens


def _distinct_targets(weight: torch.Tensor, n: int, n_regular: int, seed: int) -> torch.Tensor:
    """n distinct rows with the vocabulary's mean norm: mean direction + random direction."""
    vocab = weight[:n_regular].float()
    target_norm = vocab.norm(dim=-1).mean()
    mean = vocab.mean(0)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, weight.shape[1], generator=g)
    noise = noise / noise.norm(dim=-1, keepdim=True) * target_norm
    rows = mean.unsqueeze(0) + noise
    return rows / rows.norm(dim=-1, keepdim=True) * target_norm


class SpecialRowDelta(nn.Module):
    def __init__(
        self,
        token_ids: list[int],
        input_weight: torch.Tensor,
        output_weight: torch.Tensor | None = None,
        n_regular: int = QWEN3_N_REGULAR,
        seed: int = 0,
    ):
        """output_weight=None means tied embeddings."""
        super().__init__()
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long), persistent=True)
        with torch.no_grad():
            ids = self.token_ids
            targets = _distinct_targets(input_weight, len(token_ids), n_regular, seed)
            self.in_delta = nn.Parameter(targets - input_weight[ids].float())
            if output_weight is None:
                self.out_delta = None
            else:
                out_targets = _distinct_targets(output_weight, len(token_ids), n_regular, seed + 1)
                self.out_delta = nn.Parameter(out_targets - output_weight[ids].float())

    def embed(self, input_ids: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        onehot = (input_ids.unsqueeze(-1) == self.token_ids).to(embeds.dtype)  # [B, T, k]
        return embeds + onehot @ self.in_delta.to(embeds.dtype)

    def output_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Out-of-place effective LM-head weight (gradients reach the delta)."""
        delta = self.in_delta if self.out_delta is None else self.out_delta
        return weight.index_add(0, self.token_ids, delta.to(weight.dtype))

    @torch.no_grad()
    def merge_into(self, input_weight: torch.Tensor, output_weight: torch.Tensor | None = None) -> None:
        input_weight[self.token_ids] += self.in_delta.to(input_weight.dtype)
        if self.out_delta is not None:
            output_weight[self.token_ids] += self.out_delta.to(output_weight.dtype)
            self.out_delta.zero_()
        self.in_delta.zero_()
