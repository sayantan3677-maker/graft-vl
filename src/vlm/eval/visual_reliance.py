"""Image-reliance evaluation (roadmap §7.4).

For each held-out sample the target text is fixed and we measure its per-token
loss under different visual conditions:

    real     the correct image
    blind    image removed from the prompt
    swapped  another sample's image (of the same image type)
    zeroed   correct image layout, but all visual embeddings set to zero

If the model uses the image, `real` must be clearly lower than all the others.
"""

from collections import defaultdict
from contextlib import contextmanager

import torch

from ..data.processor import VLMProcessor
from ..model.multimodal_qwen import QwenVLM
from ..training.losses import IGNORE_INDEX

CONDITIONS = ("real", "blind", "swapped", "zeroed")


def _strip_images(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if isinstance(m["content"], list):
            text = "\n".join(p["text"] for p in m["content"] if p["type"] == "text")
            out.append({**m, "content": text})
        else:
            out.append(m)
    return out


@contextmanager
def _zero_visual(model: QwenVLM):
    original = model.encode_images
    model.encode_images = lambda pv: torch.zeros_like(original(pv))
    try:
        yield
    finally:
        model.encode_images = original


@torch.no_grad()
def _target_nll(model: QwenVLM, p: dict, device: str) -> tuple[float, int]:
    """Summed NLL over target tokens and their count."""
    ids = p["input_ids"][None].to(device)
    labels = p["labels"][None].to(device)
    pv = None if p["pixel_values"] is None else p["pixel_values"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        loss = model(ids, labels=labels, pixel_values=pv)["loss"]
    n = int((labels[:, 1:] != IGNORE_INDEX).sum())
    return loss.item() * n, n


def evaluate_reliance(model: QwenVLM, processor: VLMProcessor, samples: list[dict], device: str = "cuda") -> dict:
    """samples: [{"messages", "images", "meta": {"image_type": ...}}]; returns mean NLL per condition and type."""
    model.eval()
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_type[s["meta"]["image_type"]].append(i)

    totals = {c: defaultdict(lambda: [0.0, 0]) for c in CONDITIONS}
    for image_type, idxs in by_type.items():
        for j, i in enumerate(idxs):
            s = samples[i]
            other = samples[idxs[(j + 1) % len(idxs)]] if len(idxs) > 1 else samples[(i + 1) % len(samples)]
            variants = {
                "real": s,
                "blind": {"messages": _strip_images(s["messages"]), "images": []},
                "swapped": {"messages": s["messages"], "images": other["images"]},
            }
            for cond, v in variants.items():
                nll, n = _target_nll(model, processor(v), device)
                for key in (image_type, "all"):
                    totals[cond][key][0] += nll
                    totals[cond][key][1] += n
            with _zero_visual(model):
                nll, n = _target_nll(model, processor(s), device)
            for key in (image_type, "all"):
                totals["zeroed"][key][0] += nll
                totals["zeroed"][key][1] += n

    result = {}
    for key in ["all", *sorted(by_type)]:
        row = {c: totals[c][key][0] / max(1, totals[c][key][1]) for c in CONDITIONS}
        row["n_samples"] = len(samples) if key == "all" else len(by_type[key])
        for c in ("blind", "swapped", "zeroed"):
            row[f"gap_{c}"] = row[c] - row["real"]
        result[key] = row
    return result
