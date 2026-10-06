"""Observe branch tensors without changing ProposedModel or fusion behavior."""

import copy

import torch
from torch import nn

from models.proposed_model import ProposedModel


class FrozenBranches(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        if not isinstance(baseline, ProposedModel):
            raise TypeError("baseline must be a loaded ProposedModel.")
        if not all((baseline.use_feature_attention, baseline.use_temporal_attention,
                    baseline.use_scalar_gated_fusion)):
            raise ValueError("Requires the finalized ProposedModel, not an ablation.")
        head = baseline.mlp_head
        if (head.hidden_layer.in_features, head.hidden_layer.out_features,
                head.output_layer.out_features) != (128, 64, 3):
            raise ValueError("Requires the finalized 128 -> 64 -> 3 head.")
        self.baseline = copy.deepcopy(baseline).requires_grad_(False).eval()
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def sequences(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1:] != (24, 7):
            raise ValueError("Expected historical inputs [B,24,7].")
        captured = {}

        def spatial_hook(module, args, output):
            captured["S"] = output

        def temporal_hook(module, args):
            captured["T"] = args[1]

        def fused_hook(module, args):
            captured["F"] = args[0]

        hooks = []
        try:
            fusion = self.baseline.scalar_gated_fusion
            hooks.append(fusion.spatial_projection.register_forward_hook(spatial_hook))
            hooks.append(fusion.register_forward_pre_hook(temporal_hook))
            hooks.append(self.baseline.mlp_head.register_forward_pre_hook(fused_hook))
            prediction = self.baseline(inputs)
        finally:
            for hook in hooks:
                hook.remove()
        if any(captured[name].shape != (len(inputs), 24, 128) for name in ("S", "T", "F")):
            raise ValueError("Expected S, T, and F to have shape [B,24,128].")
        if prediction.shape != (len(inputs), 3):
            raise ValueError("Expected baseline predictions [B,3].")
        return captured["S"], captured["T"], captured["F"], prediction

    @torch.no_grad()
    def forward(self, inputs):
        spatial, temporal, fused, _ = self.sequences(inputs)
        return torch.stack((fused.mean(1), spatial.mean(1), temporal.mean(1)), dim=1)
