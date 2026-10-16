"""Complete jointly trained quantum/classical residual forecasters."""

import math
import torch
from torch import Tensor, nn
from configs import config
from models.proposed_no_quantum import ProposedNoQuantum
from models.quantum_simulator import RVQCSimulator


class ResidualForecast(nn.Module):
    def __init__(self, kind: str = "quantum", seed: int | None = None) -> None:
        super().__init__()
        if kind not in ("quantum", "classical"):
            raise ValueError("kind must be quantum or classical.")
        if (config.NUM_FEATURES, config.LOOKBACK, config.BILSTM_HIDDEN_SIZE,
                config.MLP_HIDDEN_DIM, config.ACTIVE_HORIZON) != (7, 24, 64, 64, "15"):
            raise ValueError("Residual forecasters require [B,24,7], width 128, head 64 and horizon 15.")
        seed = config.RANDOM_SEED + config.RUN_NUMBER - 1 if seed is None else seed
        self.architecture_id = f"residual_{kind}_v1"
        self.kind = kind
        # Construction order cannot change paired backbone/head initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.backbone = ProposedNoQuantum()
            self.projection = nn.Linear(128, 6)
            self.readout = nn.Linear(6, 3)
            if kind == "classical":
                torch.random.default_generator.manual_seed(seed + 11000)
                self.transform = nn.Linear(6, 6)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            bound = 1 / math.sqrt(128)
            self.projection.weight.uniform_(-bound, bound, generator=generator)
            self.projection.bias.uniform_(-bound, bound, generator=generator)
            self.readout.weight.zero_()
            self.readout.bias.zero_()
        if kind == "quantum":
            generator = torch.Generator().manual_seed(seed + 10000)
            self.quantum_angles = nn.Parameter(
                torch.empty(2, 6, 2).uniform_(-0.1, 0.1, generator=generator))
            self.simulator = RVQCSimulator()

    def forward_components(self, inputs: Tensor) -> dict[str, Tensor]:
        if inputs.ndim != 3 or inputs.shape[1:] != (24, 7):
            raise ValueError("Expected input [B,24,7].")
        pooled = self.backbone.forward_features(inputs)
        base = self.backbone.mlp_head(pooled)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            theta = math.pi * torch.tanh(self.projection(pooled.float()))
            if self.kind == "quantum":
                features = self.simulator(theta, self.quantum_angles)
            else:
                features = torch.tanh(self.transform(theta))
            correction = self.readout(features)
        return {"pooled": pooled, "base": base, "correction": correction,
                "final": base + correction}

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_components(inputs)["final"]
