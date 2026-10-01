"""Small three-output residual regressor."""

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Map a pooled fusion latent to standardized residuals."""

    def __init__(self, hidden_size: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden_size must be positive and dropout in [0, 1).")
        self.network = nn.Sequential(
            nn.Linear(128, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)
