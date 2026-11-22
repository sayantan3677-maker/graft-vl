"""Tiny-model tests for the VLM wrapper (roadmap §3, §5.2)."""

import pytest
import torch
from PIL import Image

from conftest import make_tiny_vlm
from vlm.data.collator import packed_collate
from vlm.model.special_tokens import QWEN3_BOX_END, QWEN3_BOX_START, QWEN3_VISION_END, QWEN3_VISION_START


def _sample(tiny_processor, size=(200, 120), answer="A small red square."):
    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]},
        {"role": "assistant", "content": answer},
    ]
    return tiny_processor({"messages": msgs, "images": [Image.new("RGB", size, "red")]})


@torch.no_grad()
def test_text_only_equivalence(tiny_vlm, tokenizer):
    """On text-only input every logit equals the original LLM's, except the columns of the
    special tokens whose output rows we deliberately made distinct (§3.3)."""
    ids = torch.tensor([tokenizer("Hello there, how are you?")["input_ids"]])
    ours = tiny_vlm(ids, return_logits=True)["logits"]
    ref = tiny_vlm.llm(input_ids=ids).logits
    special = torch.cat([m.token_ids for m in tiny_vlm.special_rows.values()])
    keep = torch.ones(ours.shape[-1], dtype=torch.bool)
    keep[special] = False
    torch.testing.assert_close(ours[..., keep], ref[..., keep])
    assert not torch.allclose(ours[..., special], ref[..., special])


@torch.no_grad()
def test_visual_tokens_fill_image_slots(tiny_vlm, tiny_processor):
    s = _sample(tiny_processor)
    ids = s["input_ids"][None]
    embeds = tiny_vlm.build_inputs_embeds(ids, s["pixel_values"])
    visual = tiny_vlm.encode_images(s["pixel_values"])
    torch.testing.assert_close(embeds[ids == tiny_vlm.config.image_pad_id], visual)
    assert visual.shape[0] == s["n_image_tokens"]


def test_slot_count_mismatch_raises(tiny_vlm, tiny_processor):
    s = _sample(tiny_processor)
    with pytest.raises(ValueError):
        tiny_vlm(s["input_ids"][None], pixel_values=s["pixel_values"][:0])
    extra = torch.cat([s["pixel_values"], s["pixel_values"][:1]])
    with pytest.raises(ValueError):
        tiny_vlm(s["input_ids"][None], pixel_values=extra)


@torch.no_grad()
def test_special_rows_become_distinct(tiny_vlm):
    E = tiny_vlm.llm.get_input_embeddings().weight
    ids = torch.tensor([[QWEN3_VISION_START, QWEN3_VISION_END, QWEN3_BOX_START, QWEN3_BOX_END]])
    assert torch.equal(E[QWEN3_BOX_START], E[QWEN3_BOX_END])  # like real Qwen3
    embeds = tiny_vlm.build_inputs_embeds(ids, None)[0]
    assert len({tuple(r.tolist()) for r in embeds}) == 4
    W = tiny_vlm.output_weight()
    h = torch.randn(64)
    assert (h @ W[QWEN3_BOX_START]) != (h @ W[QWEN3_BOX_END])  # box tokens separable at the output


@pytest.mark.parametrize("tied", [True, False])
@torch.no_grad()
def test_merge_special_rows_preserves_outputs(tied):
    vlm = make_tiny_vlm(tied=tied)
    ids = torch.tensor([[100, QWEN3_VISION_START, 200, QWEN3_VISION_END, QWEN3_BOX_START, 300, QWEN3_BOX_END]])
    before = vlm(ids, return_logits=True)["logits"]
    vlm.merge_special_rows()
    after_wrapper = vlm(ids, return_logits=True)["logits"]
    after_plain_llm = vlm.llm(input_ids=ids).logits
    torch.testing.assert_close(after_wrapper, before, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(after_plain_llm, before, atol=1e-5, rtol=1e-5)


def test_stage1_trainability_and_grads(tiny_vlm, tiny_processor):
    tiny_vlm.set_trainable()  # S1 preset
    s = _sample(tiny_processor)
    loss = tiny_vlm(s["input_ids"][None], labels=s["labels"][None], pixel_values=s["pixel_values"])["loss"]
    loss.backward()
    trainable = {n for n, p in tiny_vlm.named_parameters() if p.requires_grad}
    assert trainable == {"projector.fc1.weight", "projector.fc1.bias", "projector.fc2.weight",
                         "projector.fc2.bias", "special_rows.vision.in_delta"}
    assert tiny_vlm.projector.fc1.weight.grad is not None
    assert tiny_vlm.special_rows["vision"].in_delta.grad.abs().sum() > 0
    assert all(p.grad is None for p in tiny_vlm.llm.parameters())
    assert all(p.grad is None for p in tiny_vlm.vision.parameters())


def test_loss_impls_agree(tiny_vlm, tiny_processor):
    s = _sample(tiny_processor)
    args = (s["input_ids"][None],)
    kw = {"labels": s["labels"][None], "pixel_values": s["pixel_values"]}
    tiny_vlm.config.loss_impl = "naive"
    naive = tiny_vlm(*args, **kw)["loss"]
    tiny_vlm.config.loss_impl = "chunked"
    chunked = tiny_vlm(*args, **kw)["loss"]
    torch.testing.assert_close(chunked, naive, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_packed_images_match_individual(tiny_vlm, tiny_processor, tokenizer):
    samples = [_sample(tiny_processor, (200, 120), "Red."), _sample(tiny_processor, (90, 300), "A tall red bar.")]
    batch = packed_collate(samples, pad_id=tokenizer.pad_token_id)
    packed = tiny_vlm(batch["input_ids"], pixel_values=batch["pixel_values"], attention_mask=batch["attention_mask"],
                      position_ids=batch["position_ids"], return_logits=True)["logits"][0]
    offset = 0
    for s in samples:
        alone = tiny_vlm(s["input_ids"][None], pixel_values=s["pixel_values"], return_logits=True)["logits"][0]
        n = len(s["input_ids"])
        torch.testing.assert_close(packed[offset: offset + n], alone, atol=1e-4, rtol=1e-4)
        offset += n


@torch.no_grad()
def test_generate_matches_full_forward(tiny_vlm, tiny_processor):
    """Greedy KV-cached generation must agree with argmax of a full forward pass."""
    s = _sample(tiny_processor)
    p = tiny_processor.prompt(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}],
        [Image.new("RGB", (200, 120), "red")],
    )
    gen = tiny_vlm.generate(p["input_ids"][None], p["pixel_values"], max_new_tokens=5, stop_ids=())
    ids = p["input_ids"][None]
    for tok_id in gen:
        logits = tiny_vlm(ids, pixel_values=p["pixel_values"], return_logits=True)["logits"]
        assert int(logits[0, -1].argmax()) == tok_id
        ids = torch.cat([ids, torch.tensor([[tok_id]])], dim=1)


def _lora_vlm(seed=0):
    vlm = make_tiny_vlm(tied=True, seed=seed)
    from vlm.model.multimodal_qwen import QwenVLM

    vlm.config.lora = {"r": 4, "alpha": 8, "dropout": 0.0}
    vlm._inject_lora(vlm.config.lora)
    with torch.no_grad():  # LoRA B starts at zero; make the adapter non-trivial
        for n, p in vlm.llm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    return vlm


def test_lora_trainability():
    vlm = _lora_vlm()
    vlm.set_trainable(llm=True)
    trainable = [n for n, p in vlm.named_parameters() if p.requires_grad]
    assert trainable and all(("lora_" in n) or n.startswith(("projector.", "special_rows.vision")) for n in trainable)
    assert not any(p.requires_grad for n, p in vlm.llm.named_parameters() if "lora_" not in n)


@torch.no_grad()
def test_lora_merge_equivalence_and_scaling(tokenizer):
    ids = torch.tensor([tokenizer("LoRA merge check.")["input_ids"]])
    vlm = _lora_vlm()
    with_adapter = vlm(ids, return_logits=True)["logits"]
    assert vlm.merge_lora(1.0) == 14  # 7 targets × 2 layers
    torch.testing.assert_close(vlm(ids, return_logits=True)["logits"], with_adapter, atol=1e-4, rtol=1e-4)

    base = make_tiny_vlm(tied=True, seed=0)
    zero = _lora_vlm()
    zero.merge_lora(0.0)  # strength 0 → original weights
    torch.testing.assert_close(zero(ids, return_logits=True)["logits"], base(ids, return_logits=True)["logits"], atol=1e-5, rtol=1e-5)


def test_vision_top_layers_trainability(tiny_vlm):
    tiny_vlm.set_trainable(llm=False, vision=True, vision_top_layers=1)
    layers = tiny_vlm.vision.tower.encoder.layers
    assert all(p.requires_grad for p in layers[-1].parameters())
    assert not any(p.requires_grad for p in layers[0].parameters())
    assert not any(p.requires_grad for p in tiny_vlm.vision.tower.embeddings.parameters())
