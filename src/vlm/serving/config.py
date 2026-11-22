"""HF config for exported checkpoints (model_type "qwen_siglip_vlm"), shared by export and vLLM."""

from transformers import AutoConfig, PretrainedConfig, Qwen3Config, SiglipVisionConfig

MODEL_TYPE = "qwen_siglip_vlm"
ARCHITECTURE = "QwenSiglipVLMForConditionalGeneration"


class QwenSiglipVLMConfig(PretrainedConfig):
    model_type = MODEL_TYPE
    sub_configs = {"text_config": Qwen3Config, "vision_config": SiglipVisionConfig}

    def __init__(self, text_config=None, vision_config=None, image_pad_id: int = 151655, reduce_factor: int = 2,
                 projector_layernorm: bool = False, tile_size: int = 384, max_tiles: int = 6,
                 thumbnail: bool = True, **kwargs):
        if isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)
        if isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config or Qwen3Config()
        self.vision_config = vision_config or SiglipVisionConfig()
        self.image_pad_id = image_pad_id
        self.reduce_factor = reduce_factor
        self.projector_layernorm = projector_layernorm
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        self.thumbnail = thumbnail
        kwargs.setdefault("tie_word_embeddings", self.text_config.tie_word_embeddings)
        super().__init__(**kwargs)

    @property
    def tokens_per_tile(self) -> int:
        return (self.vision_config.image_size // self.vision_config.patch_size // self.reduce_factor) ** 2


def register_config() -> None:
    try:
        AutoConfig.register(MODEL_TYPE, QwenSiglipVLMConfig)
    except ValueError:
        pass  # already registered
