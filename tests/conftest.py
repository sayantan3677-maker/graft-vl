import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm.data.processor import ProcessorConfig, VLMProcessor  # noqa: E402
from vlm.model.multimodal_qwen import QwenVLM, VLMConfig  # noqa: E402
from vlm.model.vision_encoder import VisionEncoder  # noqa: E402

TINY_TILE = 64  # tiny SigLIP: 64 px / patch 16 = 4×4 grid → 2×2 unshuffle → 4 tokens per tile


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


@pytest.fixture(scope="session")
def processor(tokenizer):
    return VLMProcessor(tokenizer, ProcessorConfig(tile_size=384, max_tiles=6))


@pytest.fixture(scope="session")
def tiny_processor(tokenizer):
    return VLMProcessor(tokenizer, ProcessorConfig(tile_size=TINY_TILE, max_tiles=4))


def make_tiny_vlm(tied: bool = True, seed: int = 0) -> QwenVLM:
    from transformers import Qwen3Config, Qwen3ForCausalLM, SiglipVisionConfig

    torch.manual_seed(seed)
    llm_cfg = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
        tie_word_embeddings=tied,
    )
    llm = Qwen3ForCausalLM(llm_cfg).eval()
    if not tied:
        # Untrained special rows are identical in real Qwen3; mimic that on the output side too.
        with torch.no_grad():
            llm.lm_head.weight[151643:] = llm.lm_head.weight[151643]
    with torch.no_grad():
        llm.model.embed_tokens.weight[151643:] = llm.model.embed_tokens.weight[151643]
    vis_cfg = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
        image_size=TINY_TILE, patch_size=16,
    )
    vision = VisionEncoder.from_config(vis_cfg, feature_layer=-2).eval()
    return QwenVLM(VLMConfig(loss_impl="chunked"), llm, vision).eval()


@pytest.fixture
def tiny_vlm():
    return make_tiny_vlm(tied=True)


@pytest.fixture
def tiny_vlm_untied():
    return make_tiny_vlm(tied=False)
