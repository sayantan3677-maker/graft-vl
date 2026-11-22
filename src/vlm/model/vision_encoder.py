"""SigLIP2 vision tower returning dense patch features (roadmap §3.1).

The tower is truncated at the selected feature layer, so layers after it and the
pooling head are never computed and never hold optimizer state.
"""

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel


class VisionEncoder(nn.Module):
    def __init__(self, model: SiglipVisionModel, feature_layer: int = -2):
        super().__init__()
        # transformers 5 puts the tower at the top level; older versions nest it in .vision_model
        vm = getattr(model, "vision_model", model)
        n_layers = len(vm.encoder.layers)
        # feature_layer indexes encoder outputs: -1 = last layer, -2 = penultimate, ...
        keep = n_layers + feature_layer + 1 if feature_layer < 0 else feature_layer
        if not 1 <= keep <= n_layers:
            raise ValueError(f"feature_layer={feature_layer} invalid for {n_layers} layers")
        vm.encoder.layers = vm.encoder.layers[:keep]
        # The MAP pooling head is unused: we take dense patch features.
        if getattr(vm, "head", None) is not None:
            vm.head = None
        vm.use_head = False
        self.model = model
        self.tower = vm
        self.feature_layer = feature_layer
        self.hidden_size = model.config.hidden_size
        self.image_size = model.config.image_size
        self.patch_size = model.config.patch_size

    @classmethod
    def from_pretrained(cls, name: str, feature_layer: int = -2, dtype: torch.dtype = torch.bfloat16) -> "VisionEncoder":
        return cls(SiglipVisionModel.from_pretrained(name, dtype=dtype), feature_layer)

    @classmethod
    def from_config(cls, config: SiglipVisionConfig, feature_layer: int = -2) -> "VisionEncoder":
        return cls(SiglipVisionModel(config), feature_layer)

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[N, 3, S, S] -> [N, grid*grid, hidden]; output of the kept last layer, before post-layernorm."""
        embeds = self.tower.embeddings(pixel_values.to(self.model.dtype))
        return self.tower.encoder(inputs_embeds=embeds).last_hidden_state
