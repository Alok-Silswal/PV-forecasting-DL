"""Small nonlinear residual learner with an exactly six-dimensional interface."""

import torch
from torch import nn


class BottleneckResidualMLP(nn.Module):
    """Predict standardized centered residuals from a 128-dimensional latent."""

    def __init__(self, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim must be positive and dropout in [0, 1).")
        self.projection = nn.Linear(128, 6)
        self.head = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(self.projection(latent))
