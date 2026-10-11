"""Paired zero-initialized corrections on the unchanged frozen backbone."""

import math
import torch
from torch import nn
from models.proposed_rvqc.frozen_baseline import FrozenBaseline
from .simulator import RVQCSimulator, pennylane_reference

ARMS = ("classical", "quantum")


def initial_state(seed: int) -> dict:
    common = torch.Generator().manual_seed(seed)
    quantum = torch.Generator().manual_seed(seed + 10000)
    classical = torch.Generator().manual_seed(seed + 11000)
    return {
        "projection.weight": torch.empty(6, 128).uniform_(-1 / math.sqrt(128), 1 / math.sqrt(128), generator=common),
        "projection.bias": torch.empty(6).uniform_(-1 / math.sqrt(128), 1 / math.sqrt(128), generator=common),
        "readout.weight": torch.zeros(3, 6), "readout.bias": torch.zeros(3),
        "quantum_angles": torch.empty(2, 6, 2).uniform_(-0.1, 0.1, generator=quantum),
        "transform.weight": torch.empty(6, 6).uniform_(-1 / math.sqrt(6), 1 / math.sqrt(6), generator=classical),
        "transform.bias": torch.empty(6).uniform_(-1 / math.sqrt(6), 1 / math.sqrt(6), generator=classical),
    }


class Correction(nn.Module):
    def __init__(self, arm: str, seed: int = 42, backend: str = "torch"):
        super().__init__()
        if arm not in ARMS or backend not in ("torch", "pennylane"):
            raise ValueError("Unknown correction arm or quantum backend.")
        self.arm, self.backend = arm, backend
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(128, 6)
            self.readout = nn.Linear(6, 3)
            if arm == "classical":
                self.transform = nn.Linear(6, 6)
        if arm == "quantum":
            self.quantum_angles = nn.Parameter(torch.empty(2, 6, 2))
            self.simulator = RVQCSimulator()
            self.reference = pennylane_reference() if backend == "pennylane" else None
        state = initial_state(seed)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(state[name])

    def forward(self, pooled):
        if pooled.ndim != 2 or pooled.shape[1] != 128:
            raise ValueError("Expected pooled fusion [B,128].")
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            theta = math.pi * torch.tanh(self.projection(pooled.float()))
            if self.arm == "classical":
                features = torch.tanh(self.transform(theta))
            elif self.backend == "torch":
                features = self.simulator(theta, self.quantum_angles)
            else:
                if pooled.device.type != "cpu":
                    raise ValueError("PennyLane reference backend is CPU-only; use torch on CUDA.")
                features = torch.stack(self.reference(theta, self.quantum_angles), dim=1).float()
            return self.readout(features)


class FrozenExtension(nn.Module):
    def __init__(self, run: int, arm: str, backend: str = "torch"):
        super().__init__()
        self.frozen = FrozenBaseline(run)
        self.correction = Correction(arm, 41 + run, backend)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, inputs):
        pooled, base = self.frozen(inputs)
        return base + self.correction(pooled)

    def forward_cached(self, pooled, base):
        return base + self.correction(pooled)
