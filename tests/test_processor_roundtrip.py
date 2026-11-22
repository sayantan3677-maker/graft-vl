"""Training samples must tokenize exactly like the real Qwen3 chat template (roadmap §4.5)."""

import torch
from PIL import Image

from vlm.training.losses import IGNORE_INDEX


def _full_template(tokenizer, messages):
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    assert text.endswith("<|im_end|>\n")
    return text[:-1]  # we stop at <|im_end|>


def test_non_thinking_matches_template(tokenizer, processor):
    msgs = [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]
    out = processor({"messages": msgs})
    expected = _full_template(tokenizer, msgs)
    assert "<think>\n\n</think>\n\n4<|im_end|>" in expected
    assert tokenizer.decode(out["input_ids"]) == expected
    assert out["input_ids"].tolist() == tokenizer(expected, add_special_tokens=False)["input_ids"]


def test_thinking_matches_template(tokenizer, processor):
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning": "Two plus two is four."},
    ]
    out = processor({"messages": msgs})
    ref = [msgs[0], {"role": "assistant", "content": "<think>\nTwo plus two is four.\n</think>\n\n4"}]
    expected = _full_template(tokenizer, ref)
    assert tokenizer.decode(out["input_ids"]) == expected
    assert out["input_ids"].tolist() == tokenizer(expected, add_special_tokens=False)["input_ids"]
    assert out["thinking"]


def test_multi_turn_strips_earlier_reasoning(tokenizer, processor):
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<think>\nOLD_REASONING\n</think>\n\nA1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    text = tokenizer.decode(processor({"messages": msgs})["input_ids"])
    assert "OLD_REASONING" not in text and "A1" in text and text.endswith("A2<|im_end|>")


def test_image_tokens_and_tiles(tokenizer, processor):
    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read the page."}]},
        {"role": "assistant", "content": "It says hello."},
    ]
    out = processor({"messages": msgs, "images": [Image.new("RGB", (1600, 400))]})
    n_tiles = out["pixel_values"].shape[0]
    assert n_tiles == 5  # thumbnail + 4×1 grid
    assert out["n_image_tokens"] == n_tiles * 144
    ids = out["input_ids"]
    vs = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ve = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    assert int((ids == vs).sum()) == int((ids == ve).sum()) == n_tiles


def test_loss_mask_covers_only_target(tokenizer, processor):
    for final in [{"role": "assistant", "content": "A red bus."},
                  {"role": "assistant", "content": "A red bus.", "reasoning": "I see a bus."}]:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}, final]
        out = processor({"messages": msgs, "images": [Image.new("RGB", (200, 200))]})
        supervised = out["input_ids"][out["labels"] != IGNORE_INDEX]
        text = tokenizer.decode(supervised)
        if "reasoning" in final:
            assert text == "<think>\nI see a bus.\n</think>\n\nA red bus.<|im_end|>"
        else:
            assert text == "A red bus.<|im_end|>"
        pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        assert not bool((supervised == pad).any())


def test_image_count_mismatch_raises(processor):
    msgs = [{"role": "user", "content": [{"type": "image"}]}, {"role": "assistant", "content": "x"}]
    try:
        processor({"messages": msgs, "images": []})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_prompt_for_inference(processor):
    p = processor.prompt([{"role": "user", "content": "Hi"}], enable_thinking=False)
    assert p["text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert p["pixel_values"] is None and p["input_ids"].dtype == torch.long
