"""
Temporal Attention module for short-term photovoltaic (PV) power forecasting.

This module adaptively assigns an importance score to each historical timestep
and reweights the temporal representations produced by the Residual BiLSTM.

Input Shape:  (batch_size, sequence_length, embedding_dim)

Output Shape: (batch_size, sequence_length, embedding_dim)
"""

import torch
import torch.nn as nn
from torch import Tensor


class TemporalAttention(nn.Module):
    """
    Lightweight Learnable Temporal Attention.

    Parameters
    ----------
    embedding_dim : Feature dimension of the Residual BiLSTM output.
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.score = nn.Linear(
            in_features=embedding_dim,
            out_features=1,
        )

        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize network weights.

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
        x: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape:
            (batch_size, sequence_length, embedding_dim)

        Returns
        -------
        Tensor
            Attention-refined tensor of shape:
            (batch_size, sequence_length, embedding_dim)
        """

        attention_weights = self.score(x)

        attention_weights = self.softmax(attention_weights)

        attended_features = x * attention_weights

        return attended_features