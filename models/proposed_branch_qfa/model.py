"""Separate A/B/C controls and optional D/E branch-interaction feature maps."""

import copy
import math

import torch
from torch import nn

from .extractor import FrozenBranches


class BranchFeatures(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(128, 3, bias=False)
        generator = torch.Generator().manual_seed(seed)
        bound = 128 ** -0.5
        nn.init.uniform_(self.projection.weight, -bound, bound, generator=generator)
        self.normalization = nn.BatchNorm1d(6, affine=False)

    def raw(self, spatial, temporal):
        if (spatial.ndim != 2 or spatial.shape[1] != 128 or temporal.shape != spatial.shape):
            raise ValueError("Expected matching S_bar and T_bar [B,128].")
        return torch.cat((self.projection(spatial), self.projection(temporal)), dim=1)

    def forward(self, spatial, temporal):
        z = self.normalization(self.raw(spatial, temporal))
        return (math.pi / 2) * (1 + torch.tanh(z / 2))


class BranchAugmentationModel(nn.Module):
    def __init__(self, baseline, arm="A", seed=42, cached=False, quantum_seed=None):
        super().__init__()
        if arm not in "ABCDE" or len(arm) != 1:
            raise ValueError("arm must be A, B, C, D, or E.")
        self.arm = arm
        self.cached = cached
        self.frozen = FrozenBranches(baseline)
        self.head = copy.deepcopy(baseline.mlp_head).requires_grad_(True)
        if arm != "A":
            self.features = BranchFeatures(seed)
            original = self.head.hidden_layer
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Linear(134, 64, dtype=original.weight.dtype)
            expanded = expanded.to(original.weight.device)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :128].copy_(original.weight)
                expanded.bias.copy_(original.bias)
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
        if arm in "DE":
            from .quantum import QuantumFeatures
            self.transform = QuantumFeatures(seed + 10000 if quantum_seed is None else quantum_seed,
                                             trainable=arm == "E")

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward_cached(self, pooled, spatial, temporal):
        if pooled.ndim != 2 or pooled.shape[1] != 128:
            raise ValueError("Expected pooled fused features [B,128].")
        if spatial.shape != pooled.shape or temporal.shape != pooled.shape:
            raise ValueError("Expected matching pooled branch features [B,128].")
        if self.arm == "A":
            return self.head(pooled)
        theta = self.features(spatial, temporal)
        q = self.transform(theta) if hasattr(self, "transform") else theta
        if q.shape != (len(pooled), 6):
            raise ValueError("Expected augmentation features [B,6].")
        return self.head(torch.cat((pooled, q), dim=1))

    def forward(self, inputs):
        if self.cached:
            if inputs.ndim != 3 or inputs.shape[1:] != (3, 128):
                raise ValueError("Expected cached [x,S_bar,T_bar] features [B,3,128].")
            packed = inputs
        else:
            packed = self.frozen(inputs)
        return self.forward_cached(packed[:, 0], packed[:, 1], packed[:, 2])

    @torch.no_grad()
    def assert_initial_equivalence(self, inputs):
        mode = self.training
        self.eval()
        try:
            expected = (self.frozen.baseline.mlp_head(inputs[:, 0]) if self.cached
                        else self.frozen.baseline(inputs))
            torch.testing.assert_close(self(inputs), expected, rtol=1e-5, atol=1e-6)
        finally:
            self.train(mode)
