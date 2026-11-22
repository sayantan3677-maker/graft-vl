"""Data path for the hybrid (Qwen3-VL architecture + text-only Qwen3 LM).

Prompt rules are the same as the SigLIP path (roadmap §4.5): the prompt is rendered with the
Qwen3 chat template (thinking / non-thinking), the target is appended, only the final assistant
turn is supervised. Each image is a single native-resolution placeholder
`<|vision_start|><|image_pad|><|vision_end|>` that the Qwen3-VL processor expands to the image's
token count (dynamic resolution, capped at `max_pixels`).

Qwen3-VL derives MRoPE positions per sample, so samples are not packed: one sample per micro-step.
"""

import random
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..data.loader import MixedSource, _read_samples
from ..training.losses import IGNORE_INDEX
from .formats import normalize_sample, pick_turn

IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
IMAGE_PAD_ID = 151655
IM_END = "<|im_end|>"


class HybridProcessor:
    def __init__(self, processor, max_pixels: int = 1_200_000, max_length: int = 6144):
        self.proc = processor
        self.tok = processor.tokenizer
        self.size = {"longest_edge": max_pixels, "shortest_edge": 65536}
        self.max_length = max_length

    def _render(self, content) -> tuple[str, int]:
        if isinstance(content, str):
            return content, 0
        parts, n = [], 0
        for p in content:
            if p["type"] == "image":
                parts.append(IMAGE)
                n += 1
            else:
                parts.append(p["text"])
        return "".join(parts), n

    def prompt_text(self, messages: list[dict], thinking: bool) -> tuple[str, int]:
        rendered, n_img = [], 0
        for m in messages:
            text, n = self._render(m["content"])
            rendered.append({"role": m["role"], "content": text})
            n_img += n
        text = self.tok.apply_chat_template(rendered, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return text, n_img

    def prepare(self, sample: dict) -> dict:
        """CPU side (DataLoader workers): render prompt/target text; images stay decoded PIL."""
        messages = sample["messages"]
        final = messages[-1]
        thinking = final.get("reasoning") is not None
        prompt, n_img = self.prompt_text(messages[:-1], thinking)
        answer = final["content"].strip()
        target = (f"<think>\n{final['reasoning'].strip()}\n</think>\n\n{answer}" if thinking else answer) + IM_END
        images = [im.convert("RGB") for im in (sample.get("images") or [])]
        if n_img != len(images):
            raise ValueError(f"{n_img} image placeholders but {len(images)} images")
        return {"prompt": prompt, "target": target, "images": images, "thinking": thinking}

    def finalize(self, prep: dict, device: str = "cuda") -> dict | None:
        """GPU side (main process): image preprocessing on `device`, placeholder expansion, tokenization.
        Matches Qwen3VLProcessor: each <|image_pad|> expands to grid.prod() // merge_size² tokens."""
        prompt, target = prep["prompt"], prep["target"]
        out = {"thinking": prep["thinking"]}
        if prep["images"]:
            ip = self.proc.image_processor
            vis = ip(images=prep["images"], size=self.size, return_tensors="pt", device=device)
            counts = (vis["image_grid_thw"].prod(-1) // ip.merge_size**2).tolist()
            for n in counts:  # expand placeholders in order (prompt holds every image placeholder)
                prompt = prompt.replace(IMAGE_PAD_TOKEN, "<|placeholder|>" * n, 1)
            prompt = prompt.replace("<|placeholder|>", IMAGE_PAD_TOKEN)
            out["pixel_values"] = vis["pixel_values"]
            out["image_grid_thw"] = vis["image_grid_thw"]
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        ids = torch.tensor(p_ids + t_ids)
        if len(ids) > self.max_length:
            return None
        labels = ids.clone()
        labels[: len(p_ids)] = IGNORE_INDEX
        out.update(input_ids=ids, labels=labels, n_image_tokens=int((ids == IMAGE_PAD_ID).sum()),
                   mm_token_type_ids=(ids == IMAGE_PAD_ID).long())
        return out

    def __call__(self, sample: dict, device: str = "cpu") -> dict:
        out = self.finalize(self.prepare(sample), device=device)
        if out is None:
            raise ValueError("sample longer than max_length")
        return out


class HybridStream(IterableDataset):
    """Weighted, infinite, single-sample stream over several manifests."""

    def __init__(self, sources: list[MixedSource], processor: HybridProcessor, seed: int = 0, shuffle_buffer: int = 256,
                 normalize: list[bool | str] | None = None):
        self.sources, self.processor, self.seed, self.shuffle_buffer = sources, processor, seed, shuffle_buffer
        self.normalize = normalize or [False] * len(sources)  # per source: answer-format normalization (formats.py)

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        n_workers = worker.num_workers if worker else 1
        rng = random.Random(self.seed * 7919 + (worker.id if worker else 0))

        def forever(src: MixedSource, k: int):
            epoch = 0
            while True:
                n = 0
                for s in _read_samples(src.shards, self.seed + 1000 * k + epoch, len(src.shards) >= n_workers):
                    n += 1
                    yield s
                if n == 0:
                    return
                epoch += 1

        iters = [forever(s, k) for k, s in enumerate(self.sources)]
        weights = [s.weight for s in self.sources]
        norm = list(self.normalize)
        buf = []
        while iters:
            k = rng.choices(range(len(iters)), weights=weights)[0]
            try:
                raw = next(iters[k])
            except StopIteration:
                del iters[k], weights[k], norm[k]
                continue
            if len(raw["messages"]) > 2 and raw["messages"][-1].get("reasoning") is None and \
                    (raw.get("meta") or {}).get("n_turns"):
                raw = pick_turn(raw, rng)  # --all-turns QA manifests: one random question per draw
            if norm[k]:  # True: instruction-driven fixes; "qa": also strip '.' from unprompted short answers
                raw = normalize_sample(raw, rng, strip_unprompted=norm[k] == "qa")
            try:
                p = self.processor.prepare(raw)
            except ValueError:
                continue
            buf.append(p)
            if len(buf) >= self.shuffle_buffer:
                yield buf.pop(rng.randrange(len(buf)))


def make_hybrid_loader(data: list[dict], processor: HybridProcessor, seed: int = 0, num_workers: int = 12) -> DataLoader:
    sources = [MixedSource(d["path"], d["weight"]) for d in data]
    normalize = [d.get("normalize_format") or False for d in data]
    return DataLoader(HybridStream(sources, processor, seed=seed, normalize=normalize), batch_size=None, num_workers=num_workers,
                      persistent_workers=num_workers > 0, prefetch_factor=8 if num_workers > 0 else None)


def pad_collate(items: list[dict], pad_id: int) -> dict:
    """Right-pad a micro-batch for Qwen3-VL (images concatenated in sample order, grids stacked)."""
    n, L = len(items), max(len(b["input_ids"]) for b in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    labels = torch.full((n, L), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    mm = torch.zeros((n, L), dtype=torch.long)
    for i, b in enumerate(items):
        t = len(b["input_ids"])
        ids[i, :t], labels[i, :t], attn[i, :t], mm[i, :t] = b["input_ids"], b["labels"], 1, b["mm_token_type_ids"]
    out = {"input_ids": ids, "labels": labels, "attention_mask": attn, "mm_token_type_ids": mm}
    with_img = [b for b in items if "pixel_values" in b]
    if with_img:
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in with_img])
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in with_img])
    return out
