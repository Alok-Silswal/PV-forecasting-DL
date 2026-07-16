"""
Scalar Gated Fusion module for short-term photovoltaic (PV) power forecasting.

This module adaptively fuses the spatial representations from the DCNN branch and the temporal representations from the Residual BiLSTM branch using a learnable scalar gate.

Inputs:
    Spatial Features  : (batch_size, sequence_length, embedding_dim)
    Temporal Features : (batch_size, sequence_length, embedding_dim)

Output:
    Fused Features    : (batch_size, sequence_length, embedding_dim)
"""

import torch
import torch.nn as nn
from torch import Tensor


class ScalarGatedFusion(nn.Module):
    """
    Scalar Gated Fusion.

    Parameters
    ----------
    embedding_dim : Feature dimension of both input branches.
    """

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
    ) -> None:
        super().__init__()

        self.spatial_projection = (
            nn.Identity()
            if spatial_dim == temporal_dim
            else nn.Linear(
                in_features=spatial_dim,
                out_features=temporal_dim,
            )
        )

        self.gate_generator = nn.Linear(
            in_features=2 * temporal_dim,
            out_features=1,
        )

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize Linear layer weights.

        Linear
        ------
        Xavier Uniform Initialization

        Bias
        ----
        Initialized to 0.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        spatial_features: Tensor,
        temporal_features: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        spatial_features : Tensor
            Spatial representations from the Feature Attention branch.

            Shape:
            (batch_size, sequence_length, embedding_dim)

        temporal_features : Tensor
            Temporal representations from the Temporal Attention branch.

            Shape:
            (batch_size, sequence_length, embedding_dim)

        Returns
        -------
        Tensor
            Fused representation.

            Shape:
            (batch_size, sequence_length, embedding_dim)
        """

        spatial_features = self.spatial_projection(spatial_features)

        # Global representation of each branch
        spatial_summary = spatial_features.mean(dim=1)

        temporal_summary = temporal_features.mean(dim=1)

        # Concatenate branch summaries
        fusion_summary = torch.cat(
            [spatial_summary, temporal_summary],
            dim=1,
        )

        # Learn scalar gate
        gate = self.gate_generator(fusion_summary)

        gate = self.sigmoid(gate)

        # (B, 1) -> (B, 1, 1)
        gate = gate.unsqueeze(-1)

        # Adaptive fusion
        fused_features = (
            gate * temporal_features
            + (1.0 - gate) * spatial_features
        )

        return fused_features