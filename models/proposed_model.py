"""End-to-end hybrid model: forecast-head output + quantum residual."""

import math
import torch
from torch import Tensor, nn
from configs import config
from models.proposed_no_quantum import ProposedNoQuantum
from models.quantum_simulator import RVQCSimulator


class ResidualForecast(nn.Module):
    def __init__(self, seed: int | None = None, *, use_temporal_attention: bool = True,
                 use_feature_attention: bool = True,
                 use_dcnn: bool = True, use_rbilstm: bool = True) -> None:
        super().__init__()
        if not use_dcnn and not use_rbilstm:
            raise ValueError("Only one encoder may be removed.")
        if (config.NUM_FEATURES, config.LOOKBACK, config.BILSTM_HIDDEN_SIZE,
                config.MLP_HIDDEN_DIM, config.ACTIVE_HORIZON) != (7, 24, 64, 64, "15"):
            raise ValueError("Hybrid forecasters require [B,24,7], width 128, head 64 and horizon 15.")
        seed = config.RANDOM_SEED + config.RUN_NUMBER - 1 if seed is None else seed
        self.architecture_id = "residual_quantum_v1"
        self.kind = "quantum"
        # Preserve initialization order and the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.backbone = ProposedNoQuantum()
            self.projection = nn.Linear(128, 6)
            self.readout = nn.Linear(6, 3)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            bound = 1 / math.sqrt(128)
            self.projection.weight.uniform_(-bound, bound, generator=generator)
            self.projection.bias.uniform_(-bound, bound, generator=generator)
            self.readout.weight.zero_()
            self.readout.bias.zero_()
        generator = torch.Generator().manual_seed(seed + 10000)
        self.quantum_angles = nn.Parameter(
            torch.empty(2, 6, 2).uniform_(-0.1, 0.1, generator=generator))
        self.simulator = RVQCSimulator()
        # Replace after construction so retained components keep the same initialization.
        if not use_dcnn or not use_rbilstm:
            with torch.random.fork_rng(devices=[]):
                if not use_dcnn:
                    torch.random.default_generator.manual_seed(seed + 20000)
                    self.backbone.dcnn = nn.Linear(config.NUM_FEATURES, config.DCNN_FILTERS)
                else:
                    torch.random.default_generator.manual_seed(seed + 30000)
                    self.backbone.residual_bilstm = nn.Linear(config.NUM_FEATURES, 2 * config.BILSTM_HIDDEN_SIZE)
        if not use_temporal_attention:
            self.backbone.temporal_attention = nn.Identity()
        if not use_feature_attention:
            self.backbone.feature_attention = nn.Identity()

    def forward_components(self, inputs: Tensor) -> dict[str, Tensor]:
        if inputs.ndim != 3 or inputs.shape[1:] != (24, 7):
            raise ValueError("Expected input [B,24,7].")
        pooled = self.backbone.forward_features(inputs)
        forecast = self.backbone.mlp_head(pooled)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            theta = math.pi * torch.tanh(self.projection(pooled.float()))
            features = self.simulator(theta, self.quantum_angles)
            quantum_residual = self.readout(features)
        # Legacy keys preserve the forward_components API.
        return {"pooled": pooled, "base": forecast, "correction": quantum_residual,
                "final": forecast + quantum_residual}

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_components(inputs)["final"]
