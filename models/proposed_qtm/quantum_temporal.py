"""One sample-level circuit generates a sequence-wide temporal profile."""

import math

import torch
from torch import nn
from torch import Tensor

from .quantum_simulator import QuantumCircuit


class QuantumTemporalWeighting(nn.Module):
    def __init__(self, seed: int = 42, quantum_backend: str = "torch") -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.scorer = nn.Linear(128, 1, bias=False)
        generator = torch.Generator().manual_seed(seed)
        nn.init.xavier_uniform_(self.scorer.weight, generator=generator)
        time = torch.arange(24, dtype=torch.float64)[:, None]
        modes = torch.arange(1, 7, dtype=torch.float64)[None, :]
        basis = math.sqrt(2 / 24) * torch.cos(math.pi * (time + 0.5) * modes / 24)
        self.register_buffer("basis", basis.float())
        self.normalization = nn.BatchNorm1d(6, affine=False)
        self.quantum = QuantumCircuit(seed, quantum_backend)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def weighting_details(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if hidden.ndim != 3 or hidden.shape[1:] != (24, 128):
            raise ValueError("Expected Residual BiLSTM features [B,24,128].")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            scores = self.scorer(hidden.float()).squeeze(-1)
            z = scores @ self.basis.float()
            normalized = self.normalization(z)
            theta = (math.pi / 2) * (1 + torch.tanh(normalized / 2))
            q = self.quantum(theta)
            logits = self.gamma.float() * (q @ self.basis.float().T)
            weights = 24 * torch.softmax(logits, dim=1)
        return weights, q, theta

    def temporal_weights(self, hidden: Tensor) -> Tensor:
        return self.weighting_details(hidden)[0]

    def forward(self, hidden: Tensor) -> Tensor:
        weights = self.temporal_weights(hidden)
        return hidden * weights.to(hidden.dtype).unsqueeze(-1)

    def forward_with_diagnostics(self, hidden: Tensor) -> tuple[Tensor, dict]:
        from .diagnostics import summarize
        weights, q, theta = self.weighting_details(hidden)
        return hidden * weights.to(hidden.dtype).unsqueeze(-1), summarize(weights, q, theta, self.gamma)
