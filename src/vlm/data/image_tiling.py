"""Deterministic aspect-ratio-aware tiling (roadmap §3.2).

An image is laid out on a cols×rows grid of square tiles (cols*rows <= max_tiles)
whose aspect ratio best matches the image. The image is scaled to fit that grid
*without distortion* and padded with white, so text keeps its shape. When more
than one tile is used, a global thumbnail of the whole image comes first.

This logic must be reimplemented exactly in the vLLM processor, so keep it simple.
"""

import math

import numpy as np
import torch
from PIL import Image

RESAMPLE = Image.Resampling.BILINEAR  # SigLIP2 preprocessor: resample=2
PAD_COLOR = (255, 255, 255)
MEAN = 0.5
STD = 0.5


def candidate_grids(max_tiles: int) -> list[tuple[int, int]]:
    """All (cols, rows) with cols*rows <= max_tiles, ordered by area then cols."""
    grids = [(c, r) for c in range(1, max_tiles + 1) for r in range(1, max_tiles + 1) if c * r <= max_tiles]
    return sorted(grids, key=lambda g: (g[0] * g[1], g[0]))


def choose_grid(width: int, height: int, tile: int, max_tiles: int) -> tuple[int, int]:
    if width <= tile and height <= tile:
        return 1, 1
    log_ratio = math.log(width / height)
    best, best_diff = (1, 1), float("inf")
    for cols, rows in candidate_grids(max_tiles):
        diff = abs(log_ratio - math.log(cols / rows))
        if diff < best_diff - 1e-9:
            best, best_diff = (cols, rows), diff
        elif abs(diff - best_diff) <= 1e-9 and width * height > 0.5 * cols * rows * tile * tile:
            # Same shape, more tiles: only worth it if the image has the pixels to fill them.
            best = (cols, rows)
    return best


def fit_and_pad(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    new_w, new_h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    canvas = Image.new("RGB", (width, height), PAD_COLOR)
    canvas.paste(image.resize((new_w, new_h), RESAMPLE), (0, 0))
    return canvas


def tile_image(image: Image.Image, tile: int, max_tiles: int, thumbnail: bool = True) -> tuple[list[Image.Image], tuple[int, int]]:
    """Returns (tiles, (cols, rows)); tiles = [thumbnail?] + row-major grid tiles."""
    image = image.convert("RGB")
    cols, rows = choose_grid(image.width, image.height, tile, max_tiles)
    if cols * rows == 1:
        return [fit_and_pad(image, tile, tile)], (1, 1)
    canvas = fit_and_pad(image, cols * tile, rows * tile)
    tiles = [canvas.crop((c * tile, r * tile, (c + 1) * tile, (r + 1) * tile)) for r in range(rows) for c in range(cols)]
    if thumbnail:
        tiles.insert(0, fit_and_pad(image, tile, tile))
    return tiles, (cols, rows)


def to_pixel_values(tiles: list[Image.Image]) -> torch.Tensor:
    """List of RGB tiles -> [N, 3, S, S] float32, SigLIP normalization."""
    arr = np.stack([np.asarray(t, dtype=np.float32) for t in tiles]) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
