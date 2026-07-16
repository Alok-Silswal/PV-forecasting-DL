"""
Shallow Multi-Layer Perceptron (MLP) prediction head for short-term photovoltaic (PV) power forecasting.

This module receives the fused spatio-temporal representation, performs global average pooling over the temporal dimension, and predicts the future PV power output.

Input Shape: (batch_size, sequence_length, embedding_dim)

Output Shape: (batch_size, 1)
"""

import torch
import torch.nn as nn
from torch import Tensor


class MLPHead(nn.Module):
    """
    Shallow MLP prediction head.

    Parameters
    ----------
    input_dim : Dimension of the fused feature representation.

    hidden_dim : Number of neurons in the hidden layer.

    dropout_rate : Dropout probability applied after the hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                "dropout_rate must satisfy 0.0 <= dropout_rate < 1.0."
            )

        self.hidden_layer = nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
        )

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(
            p=dropout_rate,
        )

        self.output_layer = nn.Linear(
            in_features=hidden_dim,
            out_features=1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize network weights.

        Hidden Layer
        ------------
        Kaiming Normal Initialization

        Output Layer
        ------------
        Xavier Uniform Initialization

        Bias
        ----
        Initialized to 0.
        """

        nn.init.kaiming_normal_(
            self.hidden_layer.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.constant_(self.hidden_layer.bias, 0.0)

        nn.init.xavier_uniform_(
            self.output_layer.weight
        )
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(
        self,
        fused_features: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        fused_features : Tensor
            Fused spatio-temporal representation.

            Shape:
            (batch_size, sequence_length, input_dim)

        Returns
        -------
        Tensor
            Predicted PV power.

            Shape:
            (batch_size, 1)
        """

        # Global Average Pooling over the temporal dimension
        pooled_features = fused_features.mean(dim=1)

        hidden_features = self.hidden_layer(
            pooled_features
        )

        hidden_features = self.relu(
            hidden_features
        )

        hidden_features = self.dropout(
            hidden_features
        )

        prediction = self.output_layer(
            hidden_features
        )

        return prediction