"""Image validation and duplicate hashing for the data pipeline (roadmap §4.1)."""

import hashlib

import numpy as np
from PIL import Image

MIN_SIDE = 32
MAX_ASPECT = 20.0
MAX_PIXELS = 40_000_000


def validate(image: Image.Image) -> tuple[Image.Image | None, str | None]:
    """Returns (RGB image, None) or (None, rejection reason)."""
    try:
        w, h = image.size
        if min(w, h) < MIN_SIDE:
            return None, "too_small"
        if max(w, h) / min(w, h) > MAX_ASPECT:
            return None, "extreme_aspect"
        if w * h > MAX_PIXELS:
            return None, "too_large"
        rgb = image.convert("RGB")
        if np.asarray(rgb.resize((16, 16))).std() < 1.0:
            return None, "blank"
        return rgb, None
    except Exception as e:
        return None, f"decode_error:{type(e).__name__}"


def exact_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def dhash(image: Image.Image, size: int = 8) -> str:
    """64-bit difference hash; equal hashes ≈ near-duplicate images (resized/recompressed copies)."""
    g = np.asarray(image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"
