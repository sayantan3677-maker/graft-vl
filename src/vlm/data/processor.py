"""Turns a conversation + images into model inputs (roadmap §3.3, §4.5).

Sample format:

    {"messages": [
        {"role": "system", "content": "..."},                       # optional
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]},
        {"role": "assistant", "content": "A cat on a sofa.", "reasoning": "..."},  # reasoning optional
     ],
     "images": [PIL.Image, ...]}                                    # one per {"type": "image"}, in order

Only the final assistant turn is supervised. The prompt is rendered with the
tokenizer's own chat template and `add_generation_prompt=True`, exactly as at
inference, then the target is appended:

    non-thinking: prompt ends "<|im_start|>assistant\n<think>\n\n</think>\n\n"; target = answer + <|im_end|>
    thinking:     prompt ends "<|im_start|>assistant\n";  target = "<think>\n{reasoning}\n</think>\n\n{answer}" + <|im_end|>

Earlier assistant turns are rendered (and reasoning-stripped) by the template itself.

Image layout in text (one image):
    single tile:  <|vision_start|><|image_pad|>*T<|vision_end|>
    tiled:        [thumbnail block] "\n" [row 1 tile blocks] "\n" [row 2 tile blocks] ...
"""

from dataclasses import dataclass

import torch
from PIL import Image

from ..training.losses import IGNORE_INDEX
from .image_tiling import tile_image, to_pixel_values

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
IM_END = "<|im_end|>"


@dataclass
class ProcessorConfig:
    tile_size: int = 384
    patch_size: int = 16
    reduce_factor: int = 2
    max_tiles: int = 6
    thumbnail: bool = True
    max_length: int = 8192

    @property
    def tokens_per_tile(self) -> int:
        return (self.tile_size // self.patch_size // self.reduce_factor) ** 2


class VLMProcessor:
    def __init__(self, tokenizer, config: ProcessorConfig | None = None):
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    # ------------------------------------------------------------------ images

    def _block(self) -> str:
        return VISION_START + IMAGE_PAD * self.config.tokens_per_tile + VISION_END

    def image_text(self, grid: tuple[int, int], has_thumbnail: bool) -> str:
        cols, rows = grid
        if cols * rows == 1:
            return self._block()
        lines = [self._block()] if has_thumbnail else []
        lines += [self._block() * cols for _ in range(rows)]
        return "\n".join(lines)

    def _render_content(self, content, images: list[Image.Image], tiles_out: list) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append(part["text"])
            elif part["type"] == "image":
                if not images:
                    raise ValueError("more image placeholders than images")
                tiles, grid = tile_image(images.pop(0), self.config.tile_size, self.config.max_tiles, self.config.thumbnail)
                tiles_out.extend(tiles)
                parts.append(self.image_text(grid, self.config.thumbnail and len(tiles) > grid[0] * grid[1]))
            else:
                raise ValueError(f"unknown content type {part['type']!r}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ text

    def render_messages(self, messages: list[dict], images: list[Image.Image]) -> tuple[list[dict], list[Image.Image]]:
        """Replace image placeholders with image-token text; returns (text messages, tiles)."""
        remaining, tiles = list(images), []
        rendered = []
        for m in messages:
            msg = {"role": m["role"], "content": self._render_content(m["content"], remaining, tiles)}
            rendered.append(msg)
        if remaining:
            raise ValueError(f"{len(remaining)} images without a placeholder")
        return rendered, tiles

    def prompt(self, messages: list[dict], images: list[Image.Image] | None = None, enable_thinking: bool = False) -> dict:
        """Inference inputs: messages end with a user turn."""
        rendered, tiles = self.render_messages(messages, images or [])
        text = self.tokenizer.apply_chat_template(
            rendered, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "pixel_values": to_pixel_values(tiles) if tiles else None,
            "text": text,
        }

    def __call__(self, sample: dict) -> dict:
        """Training inputs with assistant-only labels."""
        messages = sample["messages"]
        final = messages[-1]
        if final["role"] != "assistant":
            raise ValueError("the last message must be the supervised assistant turn")
        reasoning = final.get("reasoning")
        thinking = reasoning is not None
        p = self.prompt(messages[:-1], sample.get("images"), enable_thinking=thinking)

        answer = final["content"].strip()
        target = (f"<think>\n{reasoning.strip()}\n</think>\n\n{answer}" if thinking else answer) + IM_END
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        input_ids = torch.cat([p["input_ids"], torch.tensor(target_ids, dtype=torch.long)])
        labels = input_ids.clone()
        labels[: len(p["input_ids"])] = IGNORE_INDEX
        if len(input_ids) > self.config.max_length:
            raise ValueError(f"sample length {len(input_ids)} > max_length {self.config.max_length}")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": p["pixel_values"],
            "thinking": thinking,
            "n_image_tokens": int((input_ids == self.tokenizer.convert_tokens_to_ids(IMAGE_PAD)).sum()),
        }
