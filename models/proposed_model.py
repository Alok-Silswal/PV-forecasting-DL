"""Jointly trained quantum residual forecaster."""

import math
import torch
from torch import Tensor, nn
from configs import config
from models.proposed_no_quantum import ProposedNoQuantum
from models.quantum_simulator import RVQCSimulator


class ResidualForecast(nn.Module):
    def __init__(self, seed: int | None = None, *, use_temporal_attention: bool = True,
                 use_feature_attention: bool = True, use_scalar_gated_fusion: bool = True) -> None:
        super().__init__()
        if (config.NUM_FEATURES, config.LOOKBACK, config.BILSTM_HIDDEN_SIZE,
                config.MLP_HIDDEN_DIM, config.ACTIVE_HORIZON) != (7, 24, 64, 64, "15"):
            raise ValueError("Residual forecasters require [B,24,7], width 128, head 64 and horizon 15.")
        seed = config.RANDOM_SEED + config.RUN_NUMBER - 1 if seed is None else seed
        self.architecture_id = "residual_quantum_v1"
        self.kind = "quantum"
        self.use_scalar_gated_fusion = use_scalar_gated_fusion
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
        # Bypass after construction so retained components keep the same initialization.
        if not use_temporal_attention:
            self.backbone.temporal_attention = nn.Identity()
        if not use_feature_attention:
            self.backbone.feature_attention = nn.Identity()
        if not use_scalar_gated_fusion:
            self.backbone.scalar_gated_fusion.gate_generator = nn.Identity()

    def forward_components(self, inputs: Tensor) -> dict[str, Tensor]:
        if inputs.ndim != 3 or inputs.shape[1:] != (24, 7):
            raise ValueError("Expected input [B,24,7].")
        if self.use_scalar_gated_fusion:
            pooled = self.backbone.forward_features(inputs)
        else:
            spatial = self.backbone.feature_attention(self.backbone.dcnn(inputs))
            temporal = self.backbone.temporal_attention(self.backbone.residual_bilstm(inputs))
            pooled = (0.5 * self.backbone.scalar_gated_fusion.spatial_projection(spatial)
                      + 0.5 * temporal).mean(dim=1)
        base = self.backbone.mlp_head(pooled)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            theta = math.pi * torch.tanh(self.projection(pooled.float()))
            features = self.simulator(theta, self.quantum_angles)
            correction = self.readout(features)
        return {"pooled": pooled, "base": base, "correction": correction,
                "final": base + correction}

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_components(inputs)["final"]
