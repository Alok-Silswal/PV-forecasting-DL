"""
Channel-wise Feature Attention module for short-term photovoltaic (PV) power forecasting.

This module implements Squeeze-and-Excitation (SE) channel attention to adaptively recalibrate the importance of feature channels extracted by the DCNN.

Input Shape: (batch_size, sequence_length, num_features)

Output Shape: (batch_size, sequence_length, num_features)
"""

import torch
import torch.nn as nn
from torch import Tensor


class FeatureAttention(nn.Module):
    """
    Channel-wise Feature Attention using the Squeeze-and-Excitation (SE) mechanism.

    Parameters
    ----------
    num_features : Number of feature channels produced by the DCNN.

    reduction_ratio : Reduction ratio used in the excitation network.
    """

    def __init__(
        self,
        num_features: int,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()

        if reduction_ratio <= 0:
            raise ValueError(
                "reduction_ratio must be greater than 0."
            )

        hidden_features = max(1, num_features // reduction_ratio)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(
            in_features=num_features,
            out_features=hidden_features,
        )

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=num_features,
        )

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize Linear layer weights using Kaiming Normal initialization.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )

                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape:
            (batch_size, sequence_length, num_features)

        Returns
        -------
        Tensor
            Attention-refined tensor of shape:
            (batch_size, sequence_length, num_features)
        """

        identity = x

        # (B, L, F) -> (B, F, L)
        attention_weights = x.transpose(1, 2)

        # (B, F, L) -> (B, F, 1)
        attention_weights = self.global_avg_pool(attention_weights)

        # (B, F, 1) -> (B, F)
        attention_weights = attention_weights.squeeze(-1)

        # Excitation Network
        attention_weights = self.fc1(attention_weights)
        attention_weights = self.relu(attention_weights)

        attention_weights = self.fc2(attention_weights)
        attention_weights = self.sigmoid(attention_weights)

        # (B, F) -> (B, 1, F)
        attention_weights = attention_weights.unsqueeze(1)

        output = identity * attention_weights

        return output