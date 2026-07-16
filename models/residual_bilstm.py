"""
Residual Bidirectional Long Short-Term Memory (Residual BiLSTM) module for short-term photovoltaic (PV) power forecasting.

This module models temporal dependencies within the historical input sequence using a single-layer Bidirectional LSTM with a residual projection connection.

Input Shape: (batch_size, sequence_length, input_size)

Output Shape: (batch_size, sequence_length, hidden_size * 2)
"""

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBiLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size : Number of input features at each timestep.

    hidden_size : Number of hidden units in each LSTM direction.

    dropout_rate : Dropout probability applied after the BiLSTM output and before residual addition.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                "dropout_rate must satisfy 0.0 <= dropout_rate < 1.0."
            )

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            bias=True,
        )

        # Projection is required only when dimensions differ.
        if input_size == hidden_size * 2:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(
                in_features=input_size,
                out_features=hidden_size * 2,
            )

        self.dropout = nn.Dropout(
            p=dropout_rate,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize network weights.

        LSTM
        ----
        weight_ih : Xavier Uniform

        weight_hh : Orthogonal

        bias :
            Forget gate bias = 1
            Remaining biases = 0

        Linear
        ------
        Xavier Uniform
        Bias = 0
        """

        for module in self.modules():

            if isinstance(module, nn.LSTM):

                for name, parameter in module.named_parameters():

                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)

                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)

                    elif "bias" in name:
                        nn.init.constant_(parameter, 0.0)

                        hidden_size = parameter.shape[0] // 4

                        with torch.no_grad():
                            parameter[
                                hidden_size:2 * hidden_size
                            ].fill_(1.0)

            elif isinstance(module, nn.Linear):

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
            (batch_size, sequence_length, input_size)

        Returns
        -------
        Tensor
            Output tensor of shape:
            (batch_size, sequence_length, hidden_size * 2)
        """

        residual = self.projection(x)

        lstm_output, _ = self.bilstm(x)

        lstm_output = self.dropout(lstm_output)

        output = lstm_output + residual

        return output