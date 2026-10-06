"""Sample-dependent, time-shared vector gate over matching branch channels."""

import torch
from torch import Tensor, nn


class VectorGatedFusion(nn.Module):
    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.gate_generator = nn.Linear(256, 128)
        generator = torch.Generator().manual_seed(seed + 2000)
        nn.init.normal_(self.gate_generator.weight, std=0.01, generator=generator)
        nn.init.zeros_(self.gate_generator.bias)

    def gate(self, spatial: Tensor, temporal: Tensor) -> Tensor:
        if spatial.ndim != 3 or spatial.shape[1:] != (24, 128) or temporal.shape != spatial.shape:
            raise ValueError("Expected matching S and T_q [B,24,128].")
        summary = torch.cat((spatial.mean(1), temporal.mean(1)), dim=1)
        return torch.sigmoid(self.gate_generator(summary))

    def forward(self, spatial: Tensor, temporal: Tensor) -> Tensor:
        gate = self.gate(spatial, temporal).unsqueeze(1)
        return gate * temporal + (1 - gate) * spatial
