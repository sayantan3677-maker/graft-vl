"""Streaming training data: WebDataset shards → processor → token-budget packed batches."""

import io
import json
import random
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .collator import packed_collate
from .processor import VLMProcessor


def _read_samples(shards: list[str], seed: int, split_by_worker: bool = True) -> Iterator[dict]:
    import webdataset as wds

    # Single node: no node splitting. Shards are split across DataLoader workers by WebDataset's
    # default workersplitter (passing split_by_worker as nodesplitter would split twice).
    # Small sources (fewer shards than workers) are read in full by every worker instead.
    kwargs = {} if split_by_worker else {"workersplitter": lambda urls: urls}
    ds = wds.WebDataset(shards, shardshuffle=len(shards), nodesplitter=None, seed=seed, empty_check=False, **kwargs)
    for item in ds:
        rec = json.loads(item["json"])
        img_key = next((k for k in item if k.startswith("img.")), None)
        images = []
        if img_key is not None:  # text-only samples (e.g. replay) have no image
            image = Image.open(io.BytesIO(item[img_key]))
            image.load()
            images = [image]  # shards currently hold at most one image per sample
        yield {"messages": rec["messages"], "images": images, "meta": rec["meta"]}


class MixedSource:
    """One manifest in a weighted mixture."""

    def __init__(self, manifest_dir: str, weight: float):
        self.dir = manifest_dir
        self.weight = weight
        self.shards = sorted(str(p) for p in (Path(manifest_dir) / "shards").glob("*.tar"))
        if not self.shards:
            raise FileNotFoundError(f"no shards in {manifest_dir}/shards")


class PackedStream(IterableDataset):
    """Greedy packing into rows of at most `max_tokens` tokens (and `max_tiles` tiles)."""

    def __init__(self, shards: list[str], processor: VLMProcessor, max_tokens: int, max_tiles: int,
                 pad_id: int, seed: int = 0, shuffle_buffer: int = 512, epochs: int | None = None,
                 dense_mask: bool = True, sources: list[MixedSource] | None = None):
        """`sources` (weighted manifests) replaces `shards` for mixed streams; mixed streams are infinite."""
        self.shards, self.processor = shards, processor
        self.sources = sources
        self.dense_mask = dense_mask
        self.max_tokens, self.max_tiles, self.pad_id = max_tokens, max_tiles, pad_id
        self.seed, self.shuffle_buffer, self.epochs = seed, shuffle_buffer, epochs

    def _mixed_raw(self) -> Iterator[dict]:
        """Weighted per-sample interleaving of several manifests, each repeated forever."""
        worker = get_worker_info()
        n_workers = worker.num_workers if worker else 1
        rng = random.Random(self.seed * 7919 + (worker.id if worker else 0))

        def forever(src: MixedSource, k: int):
            epoch = 0
            while True:
                n = 0
                for sample in _read_samples(src.shards, self.seed + 1000 * k + epoch, len(src.shards) >= n_workers):
                    n += 1
                    yield sample
                if n == 0:
                    return
                epoch += 1

        iters = [forever(src, k) for k, src in enumerate(self.sources)]
        weights = [src.weight for src in self.sources]
        while iters:
            k = rng.choices(range(len(iters)), weights=weights)[0]
            try:
                yield next(iters[k])
            except StopIteration:  # this worker has no shards of that source
                del iters[k], weights[k]

    def _processed(self) -> Iterator[dict]:
        if self.sources is not None:
            yield from self._processed_from(self._mixed_raw(), single_pass=True)
            return
        worker = get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))
        epoch = 0
        while self.epochs is None or epoch < self.epochs:
            n_read = 0

            def counted(it):
                nonlocal n_read
                for x in it:
                    n_read += 1
                    yield x

            yield from self._processed_from(counted(_read_samples(self.shards, self.seed + epoch)), rng=rng)
            if n_read == 0:
                return  # this worker was assigned no shards; stop instead of spinning
            epoch += 1

    def _processed_from(self, raw: Iterator[dict], rng: random.Random | None = None, single_pass: bool = False) -> Iterator[dict]:
        worker = get_worker_info()
        rng = rng or random.Random(self.seed + 17 + (worker.id if worker else 0))
        buf = []
        for sample in raw:
            try:
                p = self.processor(sample)
            except ValueError:
                continue  # over-long or malformed; counted by the data pipeline, not here
            if len(p["input_ids"]) > self.max_tokens:
                continue
            p["source"] = sample.get("meta", {}).get("source", "unknown")
            buf.append(p)
            if len(buf) >= self.shuffle_buffer:
                yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf

    def __iter__(self) -> Iterator[dict]:
        row, n_tok, n_tiles = [], 0, 0
        for p in self._processed():
            t = len(p["input_ids"])
            k = 0 if p["pixel_values"] is None else p["pixel_values"].shape[0]
            if row and (n_tok + t > self.max_tokens or n_tiles + k > self.max_tiles):
                yield self._emit(row)
                row, n_tok, n_tiles = [], 0, 0
            row.append(p)
            n_tok += t
            n_tiles += k
        if row:
            yield self._emit(row)

    def _emit(self, row: list[dict]) -> dict:
        batch = packed_collate(row, self.pad_id, mask_dtype=torch.bool if self.dense_mask else None)
        batch["n_samples"] = len(row)
        batch["n_image_tokens"] = sum(p["n_image_tokens"] for p in row)
        batch["n_target_tokens"] = int((batch["labels"] != -100).sum())
        batch["n_thinking"] = sum(bool(p["thinking"]) for p in row)
        batch["n_text_only"] = sum(p["pixel_values"] is None for p in row)
        return batch


def make_loader(data, processor: VLMProcessor, max_tokens: int, max_tiles: int, pad_id: int,
                seed: int = 0, num_workers: int = 8, epochs: int | None = None, dense_mask: bool = True) -> DataLoader:
    """data: a manifest dir, or a list of {"path": dir, "weight": w} for a weighted mixture."""
    if isinstance(data, (list, tuple)):
        sources = [MixedSource(d["path"], d["weight"]) for d in data]
        stream = PackedStream([], processor, max_tokens, max_tiles, pad_id, seed=seed, dense_mask=dense_mask, sources=sources)
        workers = num_workers
    else:
        shards = sorted(str(p) for p in (Path(data) / "shards").glob("*.tar"))
        if not shards:
            raise FileNotFoundError(f"no shards in {data}/shards")
        stream = PackedStream(shards, processor, max_tokens, max_tiles, pad_id, seed=seed, epochs=epochs, dense_mask=dense_mask)
        workers = min(num_workers, len(shards))  # every worker must own at least one shard
    return DataLoader(stream, batch_size=None, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
