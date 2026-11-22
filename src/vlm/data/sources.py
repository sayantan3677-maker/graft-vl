"""Data-source registry and converters to the common sample format (roadmap §4).

Every source declares what the manifest needs: image type, task, origin
(real/synthetic) and license status. `commercial_ok=None` means "not yet
verified"; the product checkpoint only trains on sources marked True (§4.2).
"""

import hashlib
import io
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import requests
from PIL import Image

CAPTION_PROMPTS = [
    "Describe this image in detail.",
    "What do you see in this image? Describe it thoroughly.",
    "Give a detailed description of the picture.",
    "Describe everything visible in this image.",
    "Write a detailed caption for this image.",
]


@dataclass(frozen=True)
class Source:
    name: str
    hf_id: str
    config: str | None
    kind: str  # "finevision" | "pixmo_url"
    image_type: str  # "photo" | "screenshot" | "page"
    task: str  # "caption" | "ocr" | ...
    origin: str  # "real" | "synthetic"
    license: str
    commercial_ok: bool | None
    min_ratings: int = 3  # FineVision quality-rating floor (relevance, correspondence, formatting)

    def to_dict(self) -> dict:
        return asdict(self)


FV = "HuggingFaceM4/FineVision"
FV_LICENSE = "per-subset (FineVision aggregates sources with their own licenses) - verify"

SOURCES: dict[str, Source] = {s.name: s for s in [
    Source("pixmo_cap", "allenai/pixmo-cap", None, "pixmo_url", "photo", "caption", "real", "ODC-BY (captions); images: web", None),
    Source("localized_narratives", FV, "localized_narratives", "finevision", "photo", "caption", "real", FV_LICENSE, None),
    Source("textcaps", FV, "textcaps", "finevision", "photo", "caption", "real", FV_LICENSE, None),
    Source("screen2words", FV, "screen2words", "finevision", "screenshot", "caption", "real", FV_LICENSE, None),
    # OCR transcriptions get low "image correspondence" ratings from FineVision's judge even when correct.
    Source("olmocr_docs", FV, "olmOCR-mix-0225-documents", "finevision", "page", "ocr", "real", FV_LICENSE, None, min_ratings=1),
    Source("ureader_cap", FV, "ureader_cap", "finevision", "page", "caption", "real", FV_LICENSE, None),
    Source("websight", FV, "websight", "finevision", "screenshot", "caption", "synthetic", FV_LICENSE, None),
    # ---------------------------------------------------------------- Stage 2 (roadmap §5.4)
    # natural photos: VQA, counting, spatial, knowledge
    Source("vqav2", FV, "vqav2", "finevision", "photo", "vqa", "real", FV_LICENSE, None),
    Source("aokvqa", FV, "aokvqa", "finevision", "photo", "vqa", "real", FV_LICENSE, None),
    Source("visual7w", FV, "visual7w", "finevision", "photo", "vqa", "real", FV_LICENSE, None),
    Source("tallyqa", FV, "tallyqa", "finevision", "photo", "counting", "real", FV_LICENSE, None),
    Source("vsr", FV, "vsr", "finevision", "photo", "spatial", "real", FV_LICENSE, None),
    Source("lnqa", FV, "lnqa", "finevision", "photo", "vqa", "real", FV_LICENSE, None),
    Source("textvqa", FV, "textvqa", "finevision", "photo", "scene_text", "real", FV_LICENSE, None),
    Source("st_vqa", FV, "st_vqa", "finevision", "photo", "scene_text", "real", FV_LICENSE, None),
    # dense captions on real images
    Source("densefusion", FV, "densefusion_1m", "finevision", "photo", "caption", "real", FV_LICENSE, None),
    # screenshots / UI
    Source("screenqa", FV, "screenqa", "finevision", "screenshot", "ui_qa", "real", FV_LICENSE, None),
    Source("visualwebinstruct", FV, "visualwebinstruct(filtered)", "finevision", "screenshot", "web_qa", "real", FV_LICENSE, None),
    # page snapshots / documents / OCR
    Source("docvqa", FV, "docvqa", "finevision", "page", "doc_qa", "real", FV_LICENSE, None),
    Source("infographic_vqa", FV, "infographic_vqa", "finevision", "page", "doc_qa", "real", FV_LICENSE, None),
    Source("ureader_qa", FV, "ureader_qa_processed", "finevision", "page", "doc_qa", "real", FV_LICENSE, None),
    Source("ureader_ie", FV, "ureader_ie", "finevision", "page", "doc_ie", "real", FV_LICENSE, None),
    Source("ocrvqa", FV, "ocrvqa", "finevision", "page", "ocr_qa", "real", FV_LICENSE, None),
    Source("doclingmatix", FV, "DoclingMatix", "finevision", "page", "doc_conversion", "real", FV_LICENSE, None, min_ratings=1),
    Source("slidevqa", FV, "slidevqa", "finevision", "page", "doc_qa", "real", FV_LICENSE, None),
    Source("sroie", FV, "sroie", "finevision", "page", "doc_ie", "real", FV_LICENSE, None),
    # charts / tables
    Source("chartqa", FV, "chartqa", "finevision", "chart", "chart_qa", "real", FV_LICENSE, None),
    Source("plotqa", FV, "plotqa", "finevision", "chart", "chart_qa", "synthetic", FV_LICENSE, None),
    Source("chart2text", FV, "chart2text", "finevision", "chart", "chart_caption", "real", FV_LICENSE, None),
    Source("tat_qa", FV, "tat_qa", "finevision", "chart", "table_qa", "real", FV_LICENSE, None),
    Source("hitab", FV, "hitab", "finevision", "chart", "table_qa", "real", FV_LICENSE, None),
    # visual reasoning / math / diagrams / STEM
    Source("ai2d", FV, "ai2d_merged", "finevision", "diagram", "diagram_qa", "real", FV_LICENSE, None),
    Source("scienceqa", FV, "scienceqa", "finevision", "diagram", "science_qa", "real", FV_LICENSE, None),
    Source("tqa", FV, "tqa", "finevision", "diagram", "science_qa", "real", FV_LICENSE, None),
    Source("iconqa", FV, "iconqa", "finevision", "diagram", "visual_math", "synthetic", FV_LICENSE, None),
    Source("geo170k_qa", FV, "geo170k(qa)", "finevision", "diagram", "visual_math", "synthetic", FV_LICENSE, None),
    Source("arxivqa", FV, "arxivqa", "finevision", "diagram", "figure_qa", "real", FV_LICENSE, None),
    # S3 additions (doc / chart / table / OCR emphasis, roadmap §16 2026-11-20)
    Source("pdfvqa", FV, "pdfvqa", "finevision", "page", "doc_qa", "real", FV_LICENSE, None),
    Source("visualmrc", FV, "visualmrc", "finevision", "screenshot", "doc_qa", "real", FV_LICENSE, None),
    Source("funsd", FV, "funsd", "finevision", "page", "doc_ie", "real", FV_LICENSE, None),
    Source("invoices_receipts", FV, "invoices_receipts", "finevision", "page", "doc_ie", "real", FV_LICENSE, None),
    Source("ureader_kg", FV, "ureader_kg_processed", "finevision", "page", "doc_qa", "real", FV_LICENSE, None),
    Source("olmocr_books", FV, "olmOCR-mix-0225-books", "finevision", "page", "ocr", "real", FV_LICENSE, None, min_ratings=1),
    Source("llavar", FV, "llavar_gpt4_20k", "finevision", "photo", "scene_text", "real", FV_LICENSE, None),
    Source("textocr_gpt4v", FV, "textocr(gpt4v)", "finevision", "photo", "ocr", "real", FV_LICENSE, None),
    Source("cocotext", FV, "cocotext", "finevision", "photo", "ocr", "real", FV_LICENSE, None),
    Source("rendered_text", FV, "rendered_text", "finevision", "page", "ocr", "synthetic", FV_LICENSE, None),
    Source("iam", FV, "iam", "finevision", "page", "ocr", "real", FV_LICENSE, None),
    Source("infographic_gpt4v", FV, "infographic(gpt4v)", "finevision", "page", "caption", "real", FV_LICENSE, None),
    Source("dvqa", FV, "dvqa", "finevision", "chart", "chart_qa", "synthetic", FV_LICENSE, None),
    Source("figureqa", FV, "figureqa", "finevision", "chart", "chart_qa", "synthetic", FV_LICENSE, None),
    Source("unichart", FV, "Unichart", "finevision", "chart", "chart_qa", "real", FV_LICENSE, None),
    Source("mmc_instruct", FV, "mmc_instruct", "finevision", "chart", "chart_qa", "real", FV_LICENSE, None),
    Source("vistext", FV, "vistext", "finevision", "chart", "chart_caption", "real", FV_LICENSE, None),
    Source("cosyn_chart", FV, "CoSyn_400k_chart", "finevision", "chart", "chart_qa", "synthetic", FV_LICENSE, None),
    Source("cosyn_table", FV, "CoSyn_400k_table", "finevision", "chart", "table_qa", "synthetic", FV_LICENSE, None),
    Source("cosyn_document", FV, "CoSyn_400k_document", "finevision", "page", "doc_qa", "synthetic", FV_LICENSE, None),
    Source("robut_wtq", FV, "robut_wtq", "finevision", "chart", "table_qa", "real", FV_LICENSE, None),
    Source("tat_dqa", FV, "tat_dqa", "finevision", "page", "table_qa", "real", FV_LICENSE, None),
    Source("multihiertt", FV, "multihiertt", "finevision", "chart", "table_qa", "real", FV_LICENSE, None),
    Source("mapqa", FV, "mapqa", "finevision", "chart", "chart_qa", "synthetic", FV_LICENSE, None),
]}


def _stable_choice(key: str, options: list[str]) -> str:
    return options[int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(options)]


def _sample(user: str, assistant: str, image: Image.Image, source: Source, extra: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user.strip()}]},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "images": [image],
        "meta": {
            "source": source.name, "image_type": source.image_type, "task": source.task,
            "origin": source.origin, "license": source.license, "commercial_ok": source.commercial_ok,
            "thinking": False, "split": "train", **extra,
        },
    }


def iter_finevision(source: Source, seed: int, buffer_size: int = 1_000, all_turns: bool = False) -> Iterator[dict]:
    """One sample per image. all_turns: keep every QA turn of the row as one multi-turn sample (training draws one
    turn per use, src/vlm/hybrid/formats.py:pick_turn); otherwise the first turn only."""
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, source.config, split="train", streaming=True).shuffle(seed=seed, buffer_size=buffer_size)
    for row in ds:
        if len(row["images"]) != 1 or not row["texts"]:
            continue  # S1 is single-image, single-turn
        ratings = [row.get(k) for k in ("relevance_min", "image_correspondence_min", "formatting_min")]
        if any(r is not None and r < source.min_ratings for r in ratings):
            continue
        turn = row["texts"][0]
        sample = _sample(turn["user"], turn["assistant"], row["images"][0], source, {"ratings": ratings})
        if all_turns and len(row["texts"]) > 1:
            for t in row["texts"][1:]:
                sample["messages"] += [{"role": "user", "content": t["user"].strip()},
                                       {"role": "assistant", "content": t["assistant"].strip()}]
            sample["meta"]["n_turns"] = len(row["texts"])
        yield sample


def _fetch(url: str, timeout: float = 10.0) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img.load()
        return img
    except Exception:
        return None


def iter_pixmo_urls(source: Source, seed: int, buffer_size: int = 1_000, workers: int = 32, batch: int = 256) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, split="train", streaming=True).shuffle(seed=seed, buffer_size=buffer_size)
    rows = iter(ds)
    with ThreadPoolExecutor(workers) as pool:
        while True:
            chunk = [r for _, r in zip(range(batch), rows)]
            if not chunk:
                return
            for row, img in zip(chunk, pool.map(lambda r: _fetch(r["image_url"]), chunk)):
                if img is None:
                    continue
                prompt = _stable_choice(row["image_url"], CAPTION_PROMPTS)
                yield _sample(prompt, row["caption"], img, source, {"image_url": row["image_url"]})


def iter_source(source: Source, seed: int, all_turns: bool = False) -> Iterator[dict]:
    if source.kind == "finevision":
        return iter_finevision(source, seed, all_turns=all_turns)
    if source.kind == "pixmo_url":
        return iter_pixmo_urls(source, seed)
    raise ValueError(source.kind)


def manifest_hash(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()[:16]
