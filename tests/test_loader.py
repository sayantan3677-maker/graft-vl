import io
import json

import torch
import webdataset as wds
from PIL import Image

from vlm.data.loader import PackedStream


def _write_shard(path, n):
    with wds.TarWriter(str(path)) as w:
        for i in range(n):
            buf = io.BytesIO()
            Image.new("RGB", (100 + 40 * i, 90), (i * 40, 0, 0)).save(buf, format="PNG")
            msgs = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]},
                {"role": "assistant", "content": f"Sample number {i}."},
            ]
            w.write({"__key__": f"s{i:03d}", "img.png": buf.getvalue(),
                     "json": json.dumps({"messages": msgs, "meta": {"source": "test"}}).encode()})


def test_packed_stream_covers_all_samples(tmp_path, tiny_processor, tokenizer):
    shard = tmp_path / "shard-000000.tar"
    _write_shard(shard, 7)
    stream = PackedStream([str(shard)], tiny_processor, max_tokens=256, max_tiles=6,
                          pad_id=tokenizer.pad_token_id, shuffle_buffer=4, epochs=1)
    batches = list(stream)
    assert sum(b["n_samples"] for b in batches) == 7
    pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    seen = set()
    for b in batches:
        ids = b["input_ids"][0]
        assert len(ids) <= 256
        assert b["pixel_values"].shape[0] * tiny_processor.config.tokens_per_tile == int((ids == pad_id).sum())
        assert b["pixel_values"].shape[0] <= 6
        assert b["attention_mask"].shape == (1, 1, len(ids), len(ids))
        text = tokenizer.decode(ids[b["labels"][0] != -100])
        seen.update(int(t) for t in text.replace("<|im_end|>", " ").replace(".", " ").split() if t.isdigit())
    assert seen == set(range(7))


def test_multiworker_loader_yields_from_every_shard(tmp_path, tiny_processor, tokenizer):
    """Regression: shards were split twice across workers, starving all but worker 0 (deadlock)."""
    import itertools

    from vlm.data.loader import make_loader

    (tmp_path / "shards").mkdir()
    for i in range(3):
        _write_shard(tmp_path / "shards" / f"shard-{i:06d}.tar", 4)
    loader = make_loader(str(tmp_path), tiny_processor, max_tokens=256, max_tiles=6,
                         pad_id=tokenizer.pad_token_id, num_workers=3, epochs=1)
    batches = list(itertools.islice(iter(loader), 100))
    assert sum(b["n_samples"] for b in batches) == 12


def test_text_only_samples_in_stream(tmp_path, tiny_processor, tokenizer):
    shard = tmp_path / "shard-000000.tar"
    with wds.TarWriter(str(shard)) as w:
        for i in range(3):
            msgs = [{"role": "user", "content": f"Question {i}?"},
                    {"role": "assistant", "content": f"Answer {i}.", "reasoning": "Think." if i % 2 else None}]
            if not i % 2:
                msgs[1].pop("reasoning")
            w.write({"__key__": f"t{i}", "json": json.dumps({"messages": msgs, "meta": {"source": "replay"}}).encode()})
    batches = list(PackedStream([str(shard)], tiny_processor, max_tokens=512, max_tiles=6,
                                pad_id=tokenizer.pad_token_id, epochs=1))
    assert sum(b["n_samples"] for b in batches) == 3
    assert all(b["pixel_values"] is None for b in batches)
    text = tokenizer.decode(batches[0]["input_ids"][0])
    assert "<think>\nThink.\n</think>" in text and "<think>\n\n</think>" in text


def test_weighted_mixture_respects_weights(tmp_path, tiny_processor, tokenizer):
    import itertools

    from vlm.data.loader import make_loader

    img_dir, txt_dir = tmp_path / "img", tmp_path / "txt"
    (img_dir / "shards").mkdir(parents=True)
    (txt_dir / "shards").mkdir(parents=True)
    for i in range(2):
        _write_shard(img_dir / "shards" / f"shard-{i:06d}.tar", 5)
    with wds.TarWriter(str(txt_dir / "shards" / "shard-000000.tar")) as w:  # 1 shard < 2 workers: read by all
        for i in range(4):
            msgs = [{"role": "user", "content": f"Q{i}?"}, {"role": "assistant", "content": f"A{i}."}]
            w.write({"__key__": f"t{i}", "json": json.dumps({"messages": msgs, "meta": {"source": "replay"}}).encode()})
    loader = make_loader([{"path": str(img_dir), "weight": 0.8}, {"path": str(txt_dir), "weight": 0.2}],
                         tiny_processor, max_tokens=256, max_tiles=6, pad_id=tokenizer.pad_token_id, num_workers=2)
    batches = list(itertools.islice(iter(loader), 60))
    n = sum(b["n_samples"] for b in batches)
    frac_text = sum(b["n_text_only"] for b in batches) / n
    assert 0.08 < frac_text < 0.35, frac_text
