import torch
from PIL import Image

from vlm.data.image_tiling import choose_grid, tile_image, to_pixel_values


def test_small_image_single_tile():
    tiles, grid = tile_image(Image.new("RGB", (300, 200), "red"), 384, 6)
    assert grid == (1, 1) and len(tiles) == 1 and tiles[0].size == (384, 384)


def test_wide_screenshot_grid_with_thumbnail():
    tiles, grid = tile_image(Image.new("RGB", (1600, 400)), 384, 6)
    assert grid == (4, 1)
    assert len(tiles) == 1 + 4  # thumbnail first
    assert all(t.size == (384, 384) for t in tiles)


def test_tall_page_is_capped_and_not_distorted():
    img = Image.new("RGB", (800, 8000), "black")
    tiles, grid = tile_image(img, 384, 6)
    assert grid == (1, 6)
    # Content is scaled to fit (0.288×) and padded white on the right, not stretched.
    first = tiles[1]
    assert first.getpixel((10, 10)) == (0, 0, 0)
    assert first.getpixel((380, 10)) == (255, 255, 255)


def test_same_shape_prefers_more_tiles_only_with_enough_pixels():
    assert choose_grid(400, 400, 384, 6) == (1, 1)
    assert choose_grid(1000, 1000, 384, 6) == (2, 2)


def test_max_tiles_respected_and_deterministic():
    for w, h in [(1920, 1080), (1080, 1920), (3000, 500), (500, 5000), (1234, 777)]:
        c, r = choose_grid(w, h, 384, 6)
        assert c * r <= 6
        assert choose_grid(w, h, 384, 6) == (c, r)


def test_pixel_values_normalization():
    px = to_pixel_values([Image.new("RGB", (64, 64), (255, 0, 128))])
    assert px.shape == (1, 3, 64, 64)
    assert torch.allclose(px[0, 0], torch.ones(64, 64))
    assert torch.allclose(px[0, 1], -torch.ones(64, 64))
