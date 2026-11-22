"""Merged hybrid chat template (scripts/prepare_hybrid_serving.py): string content renders exactly like Qwen3's template,
OpenAI-style content lists render in the training format, and vLLM detects the 'openai' content format."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    except Exception as e:  # offline without cache
        pytest.skip(str(e))


def test_merged_template(tok):
    from prepare_hybrid_serving import merge_template

    orig = tok.chat_template
    new = merge_template(orig)
    nothink = "{%- set enable_thinking = false %}\n" + new
    convs = [
        [{"role": "user", "content": "hi"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "a"},
         {"role": "assistant", "content": "<think>\nr\n</think>\n\nb"}, {"role": "user", "content": "c"}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]},
         {"role": "tool", "content": "r"}],
    ]
    for c in convs:
        for th in (True, False):
            kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=th)
            assert tok.apply_chat_template(c, chat_template=new, **kw) == tok.apply_chat_template(c, chat_template=orig, **kw)
    mm = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What?"}]}]
    ref = tok.apply_chat_template([{"role": "user", "content": IMAGE + "What?"}], chat_template=orig, tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    assert tok.apply_chat_template(mm, chat_template=new, tokenize=False, add_generation_prompt=True, enable_thinking=False) == ref
    assert tok.apply_chat_template(mm, chat_template=nothink, tokenize=False, add_generation_prompt=True) == ref
    img_url = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "What?"}]}]
    assert tok.apply_chat_template(img_url, chat_template=nothink, tokenize=False, add_generation_prompt=True) == ref
