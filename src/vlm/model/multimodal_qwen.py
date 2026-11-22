"""Qwen3 + SigLIP2 vision-language model (roadmap §3).

    image tiles → VisionEncoder → 2×2 unshuffle → MLPProjector ─┐
    input_ids → Qwen3 embeddings (+ special-row deltas) ─────────┴→ replace <|image_pad|> → Qwen3 decoder

The LM head is applied by the loss (fused/chunked CE), so full [T, V] logits are
only built when explicitly requested.
"""

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from ..training.losses import (
    IGNORE_INDEX,
    chunked_linear_cross_entropy,
    fused_linear_cross_entropy,
    shift_for_causal_lm,
)
from .projector import MLPProjector, pixel_unshuffle_tokens
from .special_tokens import (
    QWEN3_BOX_END,
    QWEN3_BOX_START,
    QWEN3_IMAGE_PAD,
    QWEN3_N_REGULAR,
    QWEN3_VISION_END,
    QWEN3_VISION_START,
    SpecialRowDelta,
)
from .vision_encoder import VisionEncoder


@dataclass
class VLMConfig:
    llm_name: str = "Qwen/Qwen3-1.7B"
    vision_name: str = "google/siglip2-so400m-patch16-384"
    feature_layer: int = -2
    reduce_factor: int = 2
    projector_layernorm: bool = False
    image_pad_id: int = QWEN3_IMAGE_PAD
    # Special rows that get learnable deltas, grouped so each group can be frozen separately.
    special_rows: dict[str, list[int]] = field(
        default_factory=lambda: {
            "vision": [QWEN3_VISION_START, QWEN3_VISION_END],
            "box": [QWEN3_BOX_START, QWEN3_BOX_END],
        }
    )
    n_regular_tokens: int = QWEN3_N_REGULAR
    loss_impl: str = "chunked"  # "chunked" | "liger" | "naive"
    # LoRA on the LLM (4B/8B, roadmap §17), e.g. {"r": 32, "alpha": 64, "dropout": 0.05}; None = no adapters.
    lora: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class QwenVLM(nn.Module):
    def __init__(self, config: VLMConfig, llm: PreTrainedModel, vision: VisionEncoder):
        super().__init__()
        self.config = config
        self.llm = llm
        self.vision = vision
        d_llm = llm.config.hidden_size
        d_in = vision.hidden_size * config.reduce_factor**2
        self.projector = MLPProjector(d_in, d_llm, layernorm=config.projector_layernorm)

        if config.lora:
            self._inject_lora(config.lora)

        in_w = llm.get_input_embeddings().weight
        tied = llm.config.tie_word_embeddings
        out_w = None if tied else llm.get_output_embeddings().weight
        self.special_rows = nn.ModuleDict(
            {
                name: SpecialRowDelta(ids, in_w.detach(), None if out_w is None else out_w.detach(),
                                      n_regular=config.n_regular_tokens, seed=i)
                for i, (name, ids) in enumerate(config.special_rows.items())
            }
        )

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(cls, config: VLMConfig, dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "sdpa") -> "QwenVLM":
        llm = AutoModelForCausalLM.from_pretrained(config.llm_name, dtype=dtype, attn_implementation=attn_implementation)
        vision = VisionEncoder.from_pretrained(config.vision_name, config.feature_layer, dtype=dtype)
        return cls(config, llm, vision)

    LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def _inject_lora(self, lora: dict) -> None:
        """In-place LoRA on the LLM's linear layers; the module tree (and our forward) is unchanged."""
        from peft import LoraConfig, inject_adapter_in_model

        cfg = LoraConfig(r=lora.get("r", 32), lora_alpha=lora.get("alpha", 64), lora_dropout=lora.get("dropout", 0.05),
                         target_modules=list(lora.get("targets", self.LORA_TARGETS)), bias="none")
        inject_adapter_in_model(cfg, self.llm.model)
        for p in self.lora_parameters():  # FP32 adapters on a BF16 base (roadmap §1.5)
            p.data = p.data.float()

    def lora_parameters(self):
        return [p for n, p in self.llm.named_parameters() if "lora_" in n]

    @torch.no_grad()
    def merge_lora(self, scale: float = 1.0) -> int:
        """Merge adapters into the base weights (optionally scaled by `scale`, the LoRA-strength
        sweep of roadmap §5.7) and remove the LoRA wrappers. Returns the number of merged layers."""
        from peft.tuners.lora import LoraLayer

        merged = 0
        for name, module in list(self.llm.model.named_modules()):
            if isinstance(module, LoraLayer):
                for adapter in module.scaling:
                    module.scaling[adapter] *= scale
                module.merge()
                parent_name, _, child = name.rpartition(".")
                parent = self.llm.model.get_submodule(parent_name) if parent_name else self.llm.model
                setattr(parent, child, module.get_base_layer())
                merged += 1
        self.config.lora = None
        return merged

    @property
    def tokens_per_tile(self) -> int:
        return (self.vision.grid // self.config.reduce_factor) ** 2

    # ------------------------------------------------------------------ trainability

    def set_trainable(self, llm: bool = False, vision: bool = False, projector: bool = True,
                      special_rows: tuple[str, ...] = ("vision",), vision_top_layers: int | None = None) -> None:
        """Stage presets: S1 = defaults. S2 full FT = llm=True. LoRA is applied by the trainer on top."""
        self.llm.requires_grad_(llm and not self.config.lora)
        if self.config.lora:  # with adapters, `llm=True` trains only the LoRA weights
            for p in self.lora_parameters():
                p.requires_grad_(llm)
        self.vision.requires_grad_(vision and vision_top_layers is None)
        if vision and vision_top_layers is not None:  # train only the last k encoder layers
            for layer in self.vision.tower.encoder.layers[-vision_top_layers:]:
                layer.requires_grad_(True)
        self.projector.requires_grad_(projector)
        for name, module in self.special_rows.items():
            module.requires_grad_(name in special_rows)

    # ------------------------------------------------------------------ forward

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[N_tiles, 3, S, S] -> [N_tiles * tokens_per_tile, d_llm], tile-major order."""
        if any(p.requires_grad for p in self.vision.parameters()):
            feats = self.vision(pixel_values)
        else:
            with torch.no_grad():  # frozen tower: no activations kept for backward
                feats = self.vision(pixel_values)
        feats = pixel_unshuffle_tokens(feats, self.config.reduce_factor)
        proj_dtype = self.projector.fc1.weight.dtype
        return self.projector(feats.to(proj_dtype)).flatten(0, 1)

    def build_inputs_embeds(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None) -> torch.Tensor:
        embeds = self.llm.get_input_embeddings()(input_ids)
        for module in self.special_rows.values():
            embeds = module.embed(input_ids, embeds)
        image_mask = input_ids == self.config.image_pad_id
        n_slots = int(image_mask.sum())
        if pixel_values is None or pixel_values.numel() == 0:
            if n_slots:
                raise ValueError(f"{n_slots} <|image_pad|> tokens but no pixel_values")
            return embeds
        visual = self.encode_images(pixel_values)
        if visual.shape[0] != n_slots:
            raise ValueError(f"visual tokens ({visual.shape[0]}) != <|image_pad|> slots ({n_slots})")
        return embeds.masked_scatter(image_mask.unsqueeze(-1), visual.to(embeds.dtype))

    def output_weight(self) -> torch.Tensor:
        weight = self.llm.get_output_embeddings().weight
        for module in self.special_rows.values():
            weight = module.output_weight(weight)
        return weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        embeds = self.build_inputs_embeds(input_ids, pixel_values)
        hidden = self.llm.model(
            inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids
        ).last_hidden_state
        weight = self.output_weight()
        out: dict[str, torch.Tensor] = {}
        if labels is not None:
            h, y = shift_for_causal_lm(hidden, labels)
            if self.config.loss_impl == "liger":
                out["loss"] = fused_linear_cross_entropy(h, weight, y)
            elif self.config.loss_impl == "chunked":
                out["loss"] = chunked_linear_cross_entropy(h, weight, y)
            else:
                logits = (h @ weight.t()).float()
                out["loss"] = nn.functional.cross_entropy(
                    logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE_INDEX
                )
        if return_logits:
            out["logits"] = hidden @ weight.t()
        return out

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None,
                 max_new_tokens: int = 256, stop_ids: tuple[int, ...] = (151645,)) -> list[int]:
        """Greedy decoding for one prompt ([1, T]); stops at <|im_end|>. For eval/debug, not serving."""
        from transformers import DynamicCache

        embeds = self.build_inputs_embeds(input_ids, pixel_values)
        weight = self.output_weight()
        cache = DynamicCache()
        out = self.llm.model(inputs_embeds=embeds, past_key_values=cache, use_cache=True)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_id = int((out.last_hidden_state[:, -1] @ weight.t()).argmax(-1))
            generated.append(next_id)
            if next_id in stop_ids:
                break
            ids = torch.tensor([[next_id]], device=input_ids.device)
            out = self.llm.model(inputs_embeds=self.build_inputs_embeds(ids, None), past_key_values=cache, use_cache=True)
        return generated

    # ------------------------------------------------------------------ export

    @torch.no_grad()
    def merge_special_rows(self) -> None:
        """Write the deltas into the LLM weights (tied: once; untied: input and output)."""
        in_w = self.llm.get_input_embeddings().weight
        out_w = None if self.llm.config.tie_word_embeddings else self.llm.get_output_embeddings().weight
        for module in self.special_rows.values():
            module.merge_into(in_w, out_w)
