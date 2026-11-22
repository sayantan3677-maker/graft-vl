"""vLLM general-plugin entry point (lightweight: registers by import path)."""


def register() -> None:
    from vllm import ModelRegistry

    from .config import ARCHITECTURE, register_config

    register_config()
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, "vlm.serving.vllm_model:QwenSiglipVLMForConditionalGeneration")
