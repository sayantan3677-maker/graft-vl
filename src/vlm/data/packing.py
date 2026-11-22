"""Sequence packing with strict per-sample attention isolation.

Several samples are concatenated into one row. Attention must stay causal
*within* each sample and must never cross sample boundaries; position ids
restart at 0 for every sample. For the SDPA backend we express this as an
explicit 4D block-diagonal causal mask.
"""

import torch


def pack_samples(samples: list[dict[str, torch.Tensor]], pad_id: int, ignore_index: int = -100) -> dict[str, torch.Tensor]:
    """samples: list of {"input_ids": [T_i], "labels": [T_i]} -> one packed row (batch size 1)."""
    input_ids = torch.cat([s["input_ids"] for s in samples])
    labels = torch.cat([s["labels"] for s in samples]).clone()
    position_ids = torch.cat([torch.arange(len(s["input_ids"])) for s in samples])
    seq_lens = torch.tensor([len(s["input_ids"]) for s in samples])

    # The first token of each sample must not be predicted from the previous
    # sample's last token: with causal shifting, labels[t+1] is predicted at t,
    # so block every boundary target.
    starts = torch.cumsum(seq_lens, 0)[:-1]
    labels[starts] = ignore_index

    return {
        "input_ids": input_ids.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
        "seq_lens": seq_lens,
    }


def block_causal_mask(seq_lens: torch.Tensor, dtype: torch.dtype = torch.bool, device: torch.device | str = "cpu") -> torch.Tensor:
    """[1, 1, T, T] mask. bool (default): True where attention is allowed, which works for any
    compute dtype. Floating dtype: additive mask, 0 where allowed and dtype-min elsewhere."""
    doc_ids = torch.repeat_interleave(torch.arange(len(seq_lens)), seq_lens).to(device)
    total = int(seq_lens.sum())
    idx = torch.arange(total, device=device)
    allowed = (doc_ids[:, None] == doc_ids[None, :]) & (idx[:, None] >= idx[None, :])
    if dtype == torch.bool:
        return allowed[None, None]
    mask = torch.zeros(total, total, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


def flex_block_mask(seq_lens: torch.Tensor, device: torch.device | str = "cuda"):
    """Sparse document-causal BlockMask for FlexAttention (same semantics as block_causal_mask).

    Blocks that cross sample boundaries are skipped entirely, so packed attention costs
    roughly the sum of per-sample attention instead of the full T×T.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    doc_ids = torch.repeat_interleave(torch.arange(len(seq_lens)), seq_lens.cpu()).to(device)
    total = int(seq_lens.sum())

    def document_causal(b, h, q, kv):
        return (doc_ids[q] == doc_ids[kv]) & (q >= kv)

    return create_block_mask(document_causal, None, None, total, total, device=device)
