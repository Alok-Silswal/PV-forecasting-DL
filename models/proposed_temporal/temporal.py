"""Fixed level/trend/curvature summaries and shared channel compression."""

import math

import torch
from torch import nn


class TemporalSummary(nn.Module):
    def __init__(self, history=24):
        super().__init__()
        if history not in (12, 24):
            raise ValueError("history must be 12 or 24.")
        self.history = history
        t = torch.linspace(-1, 1, history, dtype=torch.float64)
        level = torch.ones_like(t)
        trend = t
        curvature = t.square() - t.square().mean()
        basis = torch.stack((level, trend, curvature), dim=1)
        basis = basis / basis.norm(dim=0)
        self.register_buffer("basis", basis.float())

    def forward(self, fused):
        if fused.ndim != 3 or fused.shape[1:] != (24, 128):
            raise ValueError("Expected fused features [B,24,128].")
        return torch.einsum("tk,btc->bkc", self.basis, fused[:, -self.history:])


class TemporalFeatures(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(128, 2, bias=False)
        generator = torch.Generator().manual_seed(seed)
        nn.init.uniform_(self.projection.weight, -1 / math.sqrt(128),
                         1 / math.sqrt(128), generator=generator)
        self.normalization = nn.BatchNorm1d(6, affine=False)

    def forward(self, moments):
        if moments.ndim != 3 or moments.shape[1:] != (3, 128):
            raise ValueError("Expected temporal moments [B,3,128].")
        signals = self.projection(moments)
        # Explicit signal-major order: level, trend, curvature for each signal.
        ordered = torch.cat((signals[:, :, 0], signals[:, :, 1]), dim=1)
        z = self.normalization(ordered)
        return (math.pi / 2) * (1 + torch.tanh(z / 2))
