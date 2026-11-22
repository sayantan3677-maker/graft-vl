"""vLLM out-of-tree model for the Qwen3 + SigLIP2 VLM (roadmap §12.3).

Preprocessing reuses `vlm.data.image_tiling` and `VLMProcessor.image_text`, so serving
produces exactly the tiles and image-token layout used in training.

Prompt contract: the chat placeholder for one image is
    <|vision_start|><|image_pad|><|vision_end|>
and vLLM replaces it with the full tiled layout (thumbnail block + rows of tile blocks);
only the <|image_pad|> positions receive visual embeddings.
"""

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors

from ..data.image_tiling import choose_grid, tile_image, to_pixel_values
from ..model.projector import MLPProjector, pixel_unshuffle_tokens
from .config import QwenSiglipVLMConfig

PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _layout_text(cfg: QwenSiglipVLMConfig, width: int, height: int) -> tuple[str, int]:
    """Image-token text for one image (same as VLMProcessor.image_text) and its tile count."""
    cols, rows = choose_grid(width, height, cfg.tile_size, cfg.max_tiles)
    block = "<|vision_start|>" + "<|image_pad|>" * cfg.tokens_per_tile + "<|vision_end|>"
    if cols * rows == 1:
        return block, 1
    lines = ([block] if cfg.thumbnail else []) + [block * cols for _ in range(rows)]
    return "\n".join(lines), cols * rows + int(cfg.thumbnail)


class TilingImageProcessor:
    """Minimal HF-processor-like callable: images → flat tiles + tiles per image."""

    def __init__(self, cfg: QwenSiglipVLMConfig):
        self.cfg = cfg

    def __call__(self, images=None, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        images = images if isinstance(images, (list, tuple)) else [images]
        tiles, counts = [], []
        for img in images:
            t, _ = tile_image(img, self.cfg.tile_size, self.cfg.max_tiles, self.cfg.thumbnail)
            tiles.extend(t)
            counts.append(len(t))
        return BatchFeature({"pixel_values_flat": to_pixel_values(tiles), "image_num_tiles": torch.tensor(counts)})


class QwenSiglipProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> QwenSiglipVLMConfig:
        return self.ctx.get_hf_config(QwenSiglipVLMConfig)

    def get_hf_processor(self, **kwargs: object) -> TilingImageProcessor:
        return TilingImageProcessor(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        cfg = self.get_hf_config()
        _, n_tiles = _layout_text(cfg, image_width, image_height)
        return n_tiles * cfg.tokens_per_tile

    def get_image_size_with_most_features(self) -> ImageSize:
        cfg = self.get_hf_config()
        best, best_tokens = ImageSize(cfg.tile_size, cfg.tile_size), 0
        for cols in range(1, cfg.max_tiles + 1):
            for rows in range(1, cfg.max_tiles // cols + 1):
                w, h = cols * cfg.tile_size, rows * cfg.tile_size
                n = self.get_num_image_tokens(image_width=w, image_height=h)
                if n > best_tokens:
                    best, best_tokens = ImageSize(width=w, height=h), n
        return best

    def get_max_image_tokens(self) -> int:
        size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(image_width=size.width, image_height=size.height)


class QwenSiglipDummyInputsBuilder(BaseDummyInputsBuilder[QwenSiglipProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int],
                          mm_options: Mapping[str, BaseDummyOptions]) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        return {"image": self._get_dummy_images(width=size.width, height=size.height,
                                                num_images=mm_counts.get("image", 0),
                                                overrides=mm_options.get("image"))}


class QwenSiglipMultiModalProcessor(BaseMultiModalProcessor[QwenSiglipProcessingInfo]):
    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
                              ) -> Mapping[str, MultiModalFieldConfig]:
        num_tiles = hf_inputs.get("image_num_tiles", torch.empty(0, dtype=torch.long))
        return {
            "pixel_values_flat": MultiModalFieldConfig.flat_from_sizes("image", num_tiles),
            "image_num_tiles": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }

    def _get_prompt_updates(self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs: Mapping[str, object],
                            out_mm_kwargs: MultiModalKwargsItems) -> Sequence[PromptUpdate]:
        cfg = self.info.get_hf_config()
        tok = self.info.get_tokenizer()
        placeholder_ids = tok.encode(PLACEHOLDER, add_special_tokens=False)

        def replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            size = images.get_image_size(item_idx)
            text, _ = _layout_text(cfg, size.width, size.height)
            ids = tok.encode(text, add_special_tokens=False)
            return PromptUpdateDetails.select_token_id(ids, cfg.image_pad_id)

        return [PromptReplacement(modality="image", target=placeholder_ids, replacement=replacement)]


@MULTIMODAL_REGISTRY.register_processor(QwenSiglipMultiModalProcessor, info=QwenSiglipProcessingInfo,
                                        dummy_inputs=QwenSiglipDummyInputsBuilder)
class QwenSiglipVLMForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return PLACEHOLDER
        raise ValueError("Only images are supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg: QwenSiglipVLMConfig = vllm_config.model_config.hf_config
        self.config = cfg
        quant_config = vllm_config.quant_config
        with self._mark_tower_model(vllm_config, {"image"}):
            # The exported vision_config already has num_hidden_layers truncated at the feature layer.
            self.vision_tower = SiglipVisionModel(cfg.vision_config, quant_config,
                                                  num_hidden_layers_override=cfg.vision_config.num_hidden_layers,
                                                  require_post_norm=False, prefix=maybe_prefix(prefix, "vision_tower"))
            d_in = cfg.vision_config.hidden_size * cfg.reduce_factor**2
            self.projector = MLPProjector(d_in, cfg.text_config.hidden_size, layernorm=cfg.projector_layernorm)
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(vllm_config=vllm_config, hf_config=cfg.text_config,
                                                             prefix=maybe_prefix(prefix, "language_model"))
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        pixel_values = kwargs.get("pixel_values_flat")
        num_tiles = kwargs.get("image_num_tiles")
        if pixel_values is None:
            return []
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat(list(pixel_values))
        num_tiles = torch.as_tensor(num_tiles).flatten().tolist()
        feats = self.vision_tower(pixel_values.to(self.vision_tower.dtype))
        feats = pixel_unshuffle_tokens(feats, self.config.reduce_factor)
        emb = self.projector(feats.to(self.projector.fc1.weight.dtype))  # [tiles, tokens_per_tile, d]
        per_image = torch.split(emb, num_tiles)
        return tuple(x.flatten(0, 1) for x in per_image)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None, **kwargs: object):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(input_ids=input_ids, positions=positions,
                                         intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

