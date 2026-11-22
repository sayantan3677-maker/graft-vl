"""Gradio demo for the hybrid Qwen3 vision-language model, served by vLLM (OpenAI-compatible API).

The UI is a thin client: images are sent as base64 data URLs to `vllm serve`, exactly as a production
integration would call the model. Answers stream; time-to-first-token and decode speed are shown per request.

Usage:
  demo/launch.sh exports/<name>                       # starts vLLM + this app
  VLM_API_BASE=http://host:8000/v1 VLM_MODEL=<served-name> python demo/app.py   # app only, existing server
"""

import base64
import io
import os
import time
from pathlib import Path

import gradio as gr
from openai import OpenAI
from PIL import Image

API_BASE = os.environ.get("VLM_API_BASE", "http://localhost:8011/v1")
MODEL = os.environ.get("VLM_MODEL", "vlm")
TITLE = os.environ.get("VLM_TITLE", "Graft-VL-4B")
MAX_SIDE = 4096  # client-side guard only; the server's processor enforces the trained pixel cap
EXAMPLES = Path(__file__).parent / "examples"

client = OpenAI(base_url=API_BASE, api_key=os.environ.get("VLM_API_KEY", "EMPTY"))

TASKS = {
    "Ask a question": "",
    "Read all text (OCR)": "Read all the text in the image. Preserve the reading order and line breaks.",
    "Describe the image": "Describe this image in detail.",
    "Chart / table → data": "Convert the chart or table in the image into a Markdown table with all its values.",
    "Extract key fields (JSON)": "Extract the key fields from this document as a flat JSON object. Output only JSON.",
}


def _data_url(img: Image.Image) -> str:
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _split_thinking(text: str) -> tuple[str, str]:
    """Inline <think>…</think> (server without a reasoning parser) → (thinking, answer)."""
    if "<think>" not in text and "</think>" not in text:
        return "", text
    head, sep, tail = text.partition("</think>")
    thought = head.replace("<think>", "").strip()
    return (thought, tail.strip()) if sep else (thought, "")


def answer(image, task, question, thinking, temperature, max_tokens):
    prompt = (question or "").strip() or TASKS.get(task, "")
    if image is None and not prompt:
        yield "", "", "Upload an image and/or type a question."
        return
    content = ([{"type": "image_url", "image_url": {"url": _data_url(image)}}] if image is not None else [])
    content.append({"type": "text", "text": prompt or "Describe this image."})
    t0 = time.perf_counter()
    ttft, n_chunks, raw, reasoning = None, 0, "", ""
    try:
        stream = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": content}], stream=True,
            temperature=float(temperature), top_p=0.95 if thinking else 0.8, max_tokens=int(max_tokens),
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(thinking)}, "top_k": 20})
        usage = None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = (delta.content or "")
            rpiece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
            if piece or rpiece:
                ttft = ttft if ttft is not None else time.perf_counter() - t0
                n_chunks += 1
                raw += piece
                reasoning += rpiece
                thought, ans = _split_thinking(raw)
                yield ans, (reasoning + thought), f"streaming… TTFT {ttft * 1000:.0f} ms"
    except Exception as e:  # surface server errors in the UI instead of a stack trace
        yield "", "", f"⚠️ request failed: {type(e).__name__}: {e}"
        return
    total = time.perf_counter() - t0
    thought, ans = _split_thinking(raw)
    n_out = usage.completion_tokens if usage else n_chunks
    n_in = usage.prompt_tokens if usage else None
    decode = (n_out / (total - ttft)) if ttft is not None and total > ttft and n_out else 0.0
    stats = f"TTFT {ttft * 1000:.0f} ms · {n_out} tokens · {decode:.0f} tok/s" if ttft is not None else "no output"
    if n_in:
        stats += f" · prompt {n_in} tokens (image + text)"
    yield ans, (reasoning + thought), stats


def build() -> gr.Blocks:
    examples = [[str(EXAMPLES / f), t, q] for f, t, q in [
        ("invoice.png", "Ask a question", "What is the total amount due, and by when?"),
        ("invoice.png", "Extract key fields (JSON)", ""),
        ("chart.png", "Ask a question", "Which quarter had the highest revenue, and by how much did it beat Q1?"),
        ("chart.png", "Chart / table → data", ""),
        ("settings.png", "Ask a question", "Which settings are currently turned off?"),
        ("settings.png", "Read all text (OCR)", ""),
    ] if (EXAMPLES / f).exists()]
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\nA text-only **Qwen3-4B** turned into a vision-language model: the Qwen3-VL vision "
                    "encoder is grafted on, and the LM keeps its text ability (avg −0.9 pts on MMLU-Pro / GSM8K / "
                    "IFEval / MATH-500). Works on screenshots, documents, charts and photos. Served with vLLM.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Image", height=420)
                task = gr.Dropdown(list(TASKS), value="Ask a question", label="Task")
                question = gr.Textbox(label="Question / instruction", lines=2,
                                      placeholder="Leave empty to use the task's default instruction")
                with gr.Accordion("Generation settings", open=False):
                    thinking = gr.Checkbox(False, label="Thinking mode (slower; usually no gain on image questions)")
                    temperature = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Temperature")
                    max_tokens = gr.Slider(32, 4096, 1024, step=32, label="Max new tokens")
                run = gr.Button("Run", variant="primary")
            with gr.Column(scale=1):
                out = gr.Markdown(label="Answer", min_height=200)
                stats = gr.Markdown()
                with gr.Accordion("Thinking", open=False):
                    thought = gr.Markdown()
        inputs = [image, task, question, thinking, temperature, max_tokens]
        run.click(answer, inputs, [out, thought, stats])
        question.submit(answer, inputs, [out, thought, stats])
        if examples:
            gr.Examples(examples, [image, task, question], label="Examples (synthetic, generated by demo/make_examples.py)")
        gr.Markdown("Research / non-commercial use. Trained on public datasets with mixed licenses — see the model card.")
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=16).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)))
