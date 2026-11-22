"""CPU unit tests for packing isolation and memory-safe CE, on a tiny random Qwen3."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm.data.packing import block_causal_mask, pack_samples  # noqa: E402
from vlm.training.losses import IGNORE_INDEX, chunked_linear_cross_entropy  # noqa: E402


@pytest.fixture(scope="module")
def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
    )
    return Qwen3ForCausalLM(cfg).eval()


def _samples(lens, vocab=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(0, vocab, (n,), generator=g), "labels": torch.randint(0, vocab, (n,), generator=g)} for n in lens]


def _packed_logits(model, samples, mask_dtype=torch.bool):
    p = pack_samples(samples, pad_id=0)
    mask = block_causal_mask(p["seq_lens"], mask_dtype)
    with torch.no_grad():
        return model(input_ids=p["input_ids"], attention_mask=mask, position_ids=p["position_ids"]).logits[0]


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_packed_equals_unpacked(tiny_qwen3, mask_dtype):
    samples = _samples([17, 33, 9])
    packed = _packed_logits(tiny_qwen3, samples, mask_dtype)
    offset = 0
    for s in samples:
        with torch.no_grad():
            alone = tiny_qwen3(input_ids=s["input_ids"][None]).logits[0]
        n = len(s["input_ids"])
        torch.testing.assert_close(packed[offset: offset + n], alone, atol=1e-5, rtol=1e-4)
        offset += n


def test_no_cross_sample_leakage(tiny_qwen3):
    samples = _samples([20, 20, 20])
    base = _packed_logits(tiny_qwen3, samples)
    samples[1] = _samples([20], seed=123)[0]
    pert = _packed_logits(tiny_qwen3, samples)
    assert torch.equal(base[:20], pert[:20])
    assert torch.equal(base[40:], pert[40:])
    assert not torch.equal(base[20:40], pert[20:40])


def test_boundary_targets_masked():
    p = pack_samples(_samples([5, 7, 3]), pad_id=0)
    assert p["labels"][0, 5] == IGNORE_INDEX and p["labels"][0, 12] == IGNORE_INDEX
    assert p["position_ids"][0].tolist() == list(range(5)) + list(range(7)) + list(range(3))


def test_chunked_ce_matches_naive():
    torch.manual_seed(0)
    h = torch.randn(1000, 32, requires_grad=True)
    w = torch.randn(300, 32)
    y = torch.randint(0, 300, (1000,))
    y[:250] = IGNORE_INDEX
    naive = F.cross_entropy(h @ w.t(), y, ignore_index=IGNORE_INDEX)
    (g_naive,) = torch.autograd.grad(naive, h)
    chunked = chunked_linear_cross_entropy(h, w, y, chunk_size=128)
    (g_chunk,) = torch.autograd.grad(chunked, h)
    torch.testing.assert_close(chunked, naive, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g_chunk, g_naive, atol=1e-6, rtol=1e-4)
