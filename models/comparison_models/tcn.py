"""Causal TCN: three 32-channel residual blocks with dilations 1, 2, 4."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from configs import config
from models.mlp_head import MLPHead


class TemporalBlock(nn.Module):
    def __init__(self, inputs: int, channels: int, dilation: int,
                 kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(inputs, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.residual = nn.Conv1d(inputs, channels, 1) if inputs != channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.dropout(F.relu(self.conv1(F.pad(inputs, (self.padding, 0)))))
        hidden = self.dropout(F.relu(self.conv2(F.pad(hidden, (self.padding, 0)))))
        return F.relu(hidden + self.residual(inputs))


class TCN(nn.Module):
    architecture_id = "tcn_v1"

    def __init__(self) -> None:
        super().__init__()
        if (config.NUM_FEATURES, config.ACTIVE_HORIZON) != (7, "15"):
            raise ValueError("TCN requires seven inputs and horizon 15 (three targets).")
        self.network = nn.Sequential(
            TemporalBlock(config.NUM_FEATURES, 32, 1),
            TemporalBlock(32, 32, 2), TemporalBlock(32, 32, 4))
        self.head = MLPHead(
            input_dim=32, hidden_dim=64, dropout_rate=0.2,
            output_dim=config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON])

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != config.NUM_FEATURES:
            raise ValueError("Expected inputs [B,T,7].")
        return self.head(self.network(inputs.transpose(1, 2))[:, :, -1])
