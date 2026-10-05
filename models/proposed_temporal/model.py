"""Independent A/B/C models with an unchanged, frozen Proposed backbone."""

import copy

import torch
from torch import nn

from models.proposed_model import ProposedModel
from .temporal import TemporalFeatures, TemporalSummary


class TemporalAugmentationModel(nn.Module):
    def __init__(self, baseline, arm="A", history=24, seed=42, quantum_seed=None):
        super().__init__()
        if not isinstance(baseline, ProposedModel):
            raise TypeError("baseline must be a loaded ProposedModel.")
        if arm not in ("A", "B", "C", "D", "E"):
            raise ValueError("arm must be A, B, C, D, or E.")
        if not all((baseline.use_feature_attention, baseline.use_temporal_attention,
                    baseline.use_scalar_gated_fusion)):
            raise ValueError("Requires the finalized ProposedModel, not an ablation.")
        original = baseline.mlp_head
        if (original.hidden_layer.in_features, original.hidden_layer.out_features,
                original.output_layer.out_features) != (128, 64, 3):
            raise ValueError("Requires the finalized 128 -> 64 -> 3 head.")
        self.arm = arm
        self.backbone = copy.deepcopy(baseline).requires_grad_(False).eval()
        self.head = copy.deepcopy(original).requires_grad_(True)
        self.summary = TemporalSummary(history)
        if arm != "A":
            self.features = TemporalFeatures(seed)
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Linear(134, 64, device=original.hidden_layer.weight.device,
                                     dtype=original.hidden_layer.weight.dtype)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :128].copy_(original.hidden_layer.weight)
                expanded.bias.copy_(original.hidden_layer.bias)
            self.head.hidden_layer = expanded
        if arm == "C":
            with torch.random.fork_rng(devices=[]):
                self.transform = nn.Sequential(nn.Linear(6, 8), nn.ReLU(),
                                               nn.Linear(8, 6), nn.Tanh())
            generator = torch.Generator().manual_seed(seed + 1000)
            for layer in self.transform:
                if isinstance(layer, nn.Linear):
                    bound = layer.in_features ** -0.5
                    nn.init.uniform_(layer.weight, -bound, bound, generator=generator)
                    nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
        if arm in ("D", "E"):
            from .quantum import QuantumFeatures
            self.transform = QuantumFeatures(seed + 10000 if quantum_seed is None
                                             else quantum_seed, trainable=arm == "E")

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def representations(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1:] != (24, 7):
            raise ValueError("Expected inputs [B,24,7].")
        captured = []
        hook = self.backbone.mlp_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        try:
            self.backbone(inputs)
        finally:
            hook.remove()
        fused = captured[0]
        return fused.mean(dim=1), self.summary(fused)

    def augmentation(self, moments):
        theta = self.features(moments)
        q = self.transform(theta) if hasattr(self, "transform") else theta
        if q.shape != (len(moments), 6):
            raise ValueError("Expected augmentation [B,6].")
        return theta, q

    def forward_cached(self, pooled, moments):
        if pooled.ndim != 2 or pooled.shape[1] != 128:
            raise ValueError("Expected pooled features [B,128].")
        if self.arm == "A":
            return self.head(pooled)
        if len(pooled) != len(moments):
            raise ValueError("Pooled features and moments must have matching batches.")
        _, q = self.augmentation(moments)
        return self.head(torch.cat((pooled, q), dim=1))

    def forward(self, inputs):
        return self.forward_cached(*self.representations(inputs))

    @torch.no_grad()
    def assert_initial_equivalence(self, inputs):
        mode = self.training
        self.eval()
        try:
            torch.testing.assert_close(self(inputs), self.backbone(inputs),
                                       rtol=1e-5, atol=1e-6)
        finally:
            self.train(mode)
