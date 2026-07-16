"""
LSTM baseline architecture for short-term photovoltaic (PV) power
forecasting.

Architecture
------------
Input
│
Single-layer LSTM
│
Last Output Timestep
│
MLP Head
│
Forecast

This model is a temporal-only baseline: it has no CNN, no feature
attention, no temporal attention, no residual connections, no
bidirectional LSTM, and no feature fusion. It exists purely for
comparison against the proposed hybrid architecture.
"""

import torch.nn as nn
from torch import Tensor

from configs import config

from models.mlp_head import MLPHead


class LSTM(nn.Module):
    """
    Temporal baseline PV forecasting model.

    Composes a single-layer, unidirectional ``nn.LSTM`` for temporal
    feature extraction and ``MLPHead`` for the final forecast, using
    only the LSTM's last output timestep as the feature representation.

    Parameters
    ----------
    hidden_size : int, optional
        Hidden state size of the LSTM. Falls back to
        ``config.BILSTM_HIDDEN_SIZE`` if not provided.

    lstm_dropout_rate : float, optional
        Dropout probability applied to the LSTM's last output timestep,
        before ``MLPHead``. Falls back to ``config.BILSTM_DROPOUT_RATE``
        if not provided.

    mlp_hidden_dim : int, optional
        Hidden layer dimension of ``MLPHead``. Falls back to
        ``config.MLP_HIDDEN_DIM`` if not provided.

    mlp_dropout_rate : float, optional
        Dropout probability applied within ``MLPHead``. Falls back to
        ``config.MLP_DROPOUT_RATE`` if not provided.
    """

    def __init__(
        self,
        hidden_size: int | None = None,
        lstm_dropout_rate: float | None = None,
        mlp_hidden_dim: int | None = None,
        mlp_dropout_rate: float | None = None,
    ) -> None:

        super().__init__()

        # Any parameter left as None falls back to config.py, preserving
        # LSTM() as fully equivalent to prior behavior.
        hidden_size = hidden_size if hidden_size is not None else config.BILSTM_HIDDEN_SIZE
        lstm_dropout_rate = lstm_dropout_rate if lstm_dropout_rate is not None else config.BILSTM_DROPOUT_RATE
        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else config.MLP_HIDDEN_DIM
        mlp_dropout_rate = mlp_dropout_rate if mlp_dropout_rate is not None else config.MLP_DROPOUT_RATE

        input_size = config.NUM_FEATURES
        output_dim = config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON]

        # ------------------------------------------------------------
        # Temporal Encoder
        # ------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # nn.LSTM's own `dropout` argument only applies between stacked
        # layers, so it is a no-op for a single-layer LSTM. A separate
        # dropout layer is applied to the extracted last timestep
        # instead, so lstm_dropout_rate has an actual effect.
        self.lstm_dropout = nn.Dropout(p=lstm_dropout_rate)

        # ------------------------------------------------------------
        # Prediction Head
        # ------------------------------------------------------------
        self.mlp_head = MLPHead(
            input_dim=hidden_size,
            hidden_dim=mlp_hidden_dim,
            output_dim=output_dim,
            dropout_rate=mlp_dropout_rate,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize LSTM weights.

        Input-hidden weights:
            Xavier Uniform Initialization

        Hidden-hidden weights:
            Orthogonal Initialization

        Biases:
            Zero, then the forget-gate slice of ``bias_ih`` is set to 1
            (standard LSTM forget-gate bias initialization; since the
            effective forget-gate bias is ``bias_ih + bias_hh``, leaving
            ``bias_hh`` at zero yields an effective forget-gate bias of
            exactly 1).

        ``MLPHead`` already initializes itself and is not reinitialized
        here.
        """

        for name, param in self.lstm.named_parameters():

            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            elif "bias" in name:
                nn.init.constant_(param, 0.0)

                if "bias_ih" in name:
                    hidden_size = param.shape[0] // 4
                    param.data[hidden_size:2 * hidden_size].fill_(1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape
            ``(batch_size, sequence_length, input_size)``.

        Returns
        -------
        Tensor
            Predicted forecast of shape ``(batch_size, output_dim)``.
        """

        # (B, L, input_size) -> (B, L, hidden_size)
        output, _ = self.lstm(x)

        # (B, L, hidden_size) -> (B, hidden_size)
        features = output[:, -1, :]

        features = self.lstm_dropout(features)

        prediction = self.mlp_head(features)

        return prediction