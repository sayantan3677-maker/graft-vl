"""Make a hybrid export (scripts/export_hybrid.py output) production-servable in vLLM / OpenAI-style clients.

export_hybrid.py saves Qwen3's text chat template, which renders non-string message content as '' (images and text of an
OpenAI content list are dropped by the template; vLLM then falls back to "string" content format and prepends the
placeholder with a '\n' separator — a prompt format the model never saw in training). This script writes:
  * chat_template.jinja         merged template: Qwen3 hybrid-thinking logic + Qwen3-VL content-list handling; image parts
                                → <|vision_start|><|image_pad|><|vision_end|> concatenated with text parts without a
                                separator (= training format, src/vlm/hybrid/data.py). String content renders byte-identically.
  * chat_template_nothink.jinja same, with enable_thinking forced to false (lmms-eval / non-thinking serving default)
  * chat_template.qwen3_text_original.jinja  the original template (kept for reference/tests)
  * processor_config.json       image max_pixels set to the training cap (default 1.6M; Qwen3-VL's default is 16.7M,
                                which yields >16K-token prompts on long pages)
  * generation_config.json      Qwen3 base LM's (eos [<|im_end|>, <|endoftext|>], recommended sampling defaults)

Usage: python scripts/prepare_hybrid_serving.py exports/<name> [--max-pixels 1600000]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = """{#- Merged chat template (serving-bench, 2026-11-01): Qwen3-1.7B hybrid-thinking logic (enable_thinking) +
    Qwen3-VL content-list handling. Image/video parts become <|vision_start|><|image_pad|><|vision_end|> and are
    concatenated with text parts WITHOUT separators, exactly as training rendered them (src/vlm/hybrid/data.py).
    String content renders byte-identically to the original Qwen3 template (chat_template.qwen3_text_original.jinja). -#}
"""
_SYS_TEXT = "{%- if messages[0].content is string %}{{- messages[0].content }}{%- else %}{%- for item in messages[0].content %}{%- if 'text' in item %}{{- item.text }}{%- endif %}{%- endfor %}{%- endif %}"


def merge_template(src: str) -> str:
    def sub(old, new):
        nonlocal src
        assert old in src, f"unexpected Qwen3 template (missing: {old[:60]!r})"
        src = src.replace(old, new)

    # system message (tools / no tools) may be a content list
    sub("        {{- messages[0].content + '\\n\\n' }}", f"        {_SYS_TEXT}\n        {{{{- '\\n\\n' }}}}")
    sub("        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}",
        f"        {{{{- '<|im_start|>system\\n' }}}}\n        {_SYS_TEXT}\n        {{{{- '<|im_end|>\\n' }}}}")
    # user messages with list content count as queries (multi-step tool detection)
    sub("""message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>'))""",
        """message.role == "user" and (message.content is not string or not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')))""")
    # content lists → placeholders + text, no separators
    sub("""    {%- else %}
        {%- set content = '' %}
    {%- endif %}""", """    {%- else %}
        {%- set cns = namespace(text='') %}
        {%- for item in message.content %}
            {%- if item.type == 'image' or 'image' in item or 'image_url' in item %}
                {%- set image_count.value = image_count.value + 1 %}
                {%- if add_vision_id %}{%- set cns.text = cns.text + 'Picture ' + image_count.value|string + ': ' %}{%- endif %}
                {%- set cns.text = cns.text + '<|vision_start|><|image_pad|><|vision_end|>' %}
            {%- elif item.type == 'video' or 'video' in item %}
                {%- set video_count.value = video_count.value + 1 %}
                {%- if add_vision_id %}{%- set cns.text = cns.text + 'Video ' + video_count.value|string + ': ' %}{%- endif %}
                {%- set cns.text = cns.text + '<|vision_start|><|video_pad|><|vision_end|>' %}
            {%- elif 'text' in item %}
                {%- set cns.text = cns.text + item.text %}
            {%- endif %}
        {%- endfor %}
        {%- set content = cns.text %}
    {%- endif %}""")
    old = "{%- for message in messages %}\n    {%- if message.content is string %}"
    sub(old, "{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n" + old)
    return HEADER + src


def is_merged(template: str) -> bool:
    """True for a template this script already wrote (exports initialised from a prepared export inherit it)."""
    return template.startswith(HEADER) or "set image_count = namespace(value=0)" in template


def original_template(d: Path, base: str) -> str:
    """The text-only Qwen3 template to merge from: the saved original if it is genuine, else the base LM's own
    template (re-merging an already merged template is impossible; see is_merged)."""
    orig = d / "chat_template.qwen3_text_original.jinja"
    for cand in (orig, d / "chat_template.jinja"):
        if cand.exists() and not is_merged(cand.read_text()):
            text = cand.read_text()
            break
    else:
        from transformers import AutoTokenizer

        text = AutoTokenizer.from_pretrained(base).chat_template
        assert text and not is_merged(text), f"no usable text-only chat template for {d} (base {base})"
    orig.write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--max-pixels", type=int, default=1_600_000)
    ap.add_argument("--base-lm", default=None, help="HF id/dir of the text LM (default: export_info.json base_llm, else the 1.7B hybrid_info.json)")
    args = ap.parse_args()
    d = ROOT / args.export
    info = d / "export_info.json"  # written by export_hybrid.py (base_llm resolved from the init chain)
    base = args.base_lm or (json.loads(info.read_text()).get("base_llm") if info.exists() else None) \
        or json.loads((ROOT / "exports/hybrid-q3-1b7-vit2b/hybrid_info.json").read_text())["language_model_from"]
    merged = merge_template(original_template(d, base))
    (d / "chat_template.jinja").write_text(merged)
    (d / "chat_template_nothink.jinja").write_text("{%- set enable_thinking = false %}\n" + merged)

    pc = json.loads((d / "processor_config.json").read_text())
    pc["image_processor"]["size"]["longest_edge"] = args.max_pixels
    (d / "processor_config.json").write_text(json.dumps(pc, indent=2))

    from transformers import GenerationConfig

    GenerationConfig.from_pretrained(base).save_pretrained(d)
    print(f"serving files written to {d} (max_pixels {args.max_pixels}, generation_config from {base})")


if __name__ == "__main__":
    main()
