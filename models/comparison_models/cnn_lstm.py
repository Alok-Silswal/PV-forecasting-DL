"""
CNN-LSTM baseline architecture for short-term photovoltaic (PV) power
forecasting.

Architecture
------------
Input
│
Conv1D → BatchNorm → ReLU → Dropout
│
Conv1D → BatchNorm → ReLU → Dropout
│
Single-layer LSTM
│
Last Output Timestep
│
LSTM Dropout
│
MLP Head
│
Forecast

This model is the classical CNN-LSTM baseline: two standard
(non-dilated) convolution blocks for local feature extraction,
followed by a single-layer unidirectional LSTM for temporal modeling.
It has no attention, no residual connections, no bidirectionality,
and no feature fusion. Like the other comparison models, it uses
standard convolutions only (dilation=1); the proposed architecture's
DCNN branch remains the only component in this project that uses
dilated convolutions.
"""

import torch.nn as nn
from torch import Tensor

from configs import config

from models.mlp_head import MLPHead


class CNNLSTM(nn.Module):
    """
    Classical CNN-LSTM baseline PV forecasting model.

    Composes two standard (non-dilated) Conv1D blocks for local
    temporal feature extraction, a single-layer unidirectional LSTM
    for sequence modeling, and ``MLPHead`` for the final forecast.

    Parameters
    ----------
    num_filters : int, optional
        Number of convolutional filters used by both Conv1D blocks.
        Falls back to ``config.DCNN_FILTERS`` if not provided.

    kernel_size : int, optional
        Temporal convolution kernel size used by both Conv1D blocks.
        Falls back to ``config.DCNN_KERNEL_SIZE`` if not provided.

    cnn_dropout_rate : float, optional
        Dropout probability applied after each convolution block.
        Falls back to ``config.DCNN_DROPOUT_RATE`` if not provided.

    hidden_size : int, optional
        Hidden size of the LSTM. Falls back to
        ``config.BILSTM_HIDDEN_SIZE`` if not provided.

    lstm_dropout_rate : float, optional
        Dropout probability applied to the LSTM's last output
        timestep. Falls back to ``config.BILSTM_DROPOUT_RATE`` if not
        provided.

    mlp_hidden_dim : int, optional
        Hidden layer dimension of ``MLPHead``. Falls back to
        ``config.MLP_HIDDEN_DIM`` if not provided.

    mlp_dropout_rate : float, optional
        Dropout probability applied within ``MLPHead``. Falls back to
        ``config.MLP_DROPOUT_RATE`` if not provided.
    """

    def __init__(
        self,
        num_filters: int | None = None,
        kernel_size: int | None = None,
        cnn_dropout_rate: float | None = None,
        hidden_size: int | None = None,
        lstm_dropout_rate: float | None = None,
        mlp_hidden_dim: int | None = None,
        mlp_dropout_rate: float | None = None,
    ) -> None:

        super().__init__()

        # Any parameter left as None falls back to config.py, preserving
        # CNNLSTM() as fully equivalent to prior behavior.
        num_filters = num_filters if num_filters is not None else config.DCNN_FILTERS
        kernel_size = kernel_size if kernel_size is not None else config.DCNN_KERNEL_SIZE
        cnn_dropout_rate = cnn_dropout_rate if cnn_dropout_rate is not None else config.DCNN_DROPOUT_RATE
        hidden_size = hidden_size if hidden_size is not None else config.BILSTM_HIDDEN_SIZE
        lstm_dropout_rate = lstm_dropout_rate if lstm_dropout_rate is not None else config.BILSTM_DROPOUT_RATE
        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else config.MLP_HIDDEN_DIM
        mlp_dropout_rate = mlp_dropout_rate if mlp_dropout_rate is not None else config.MLP_DROPOUT_RATE

        if not (0.0 <= cnn_dropout_rate < 1.0):
            raise ValueError(
                "cnn_dropout_rate must satisfy 0.0 <= cnn_dropout_rate < 1.0."
            )

        if not (0.0 <= lstm_dropout_rate < 1.0):
            raise ValueError(
                "lstm_dropout_rate must satisfy 0.0 <= lstm_dropout_rate < 1.0."
            )

        input_channels = config.NUM_FEATURES
        output_dim = config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON]

        self.hidden_size = hidden_size

        # ------------------------------------------------------------------
        # First Convolution Block
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            dilation=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(num_filters)

        # ------------------------------------------------------------------
        # Second Convolution Block
        # ------------------------------------------------------------------
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            dilation=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_filters)

        # ------------------------------------------------------------------
        # Shared CNN Layers
        # ------------------------------------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.cnn_dropout = nn.Dropout(p=cnn_dropout_rate)

        # ------------------------------------------------------------------
        # LSTM
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm_dropout = nn.Dropout(p=lstm_dropout_rate)

        # ------------------------------------------------------------------
        # Prediction Head
        # ------------------------------------------------------------------
        self.mlp_head = MLPHead(
            input_dim=hidden_size,
            hidden_dim=mlp_hidden_dim,
            output_dim=output_dim,
            dropout_rate=mlp_dropout_rate,
        )

        # ------------------------------------------------------------------
        # Weight Initialization
        # ------------------------------------------------------------------
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize network weights.

        Conv1D:
            Kaiming Normal Initialization

        BatchNorm:
            Weight = 1
            Bias = 0

        LSTM:
            Input-hidden weights (``weight_ih``): Xavier Uniform
            Hidden-hidden weights (``weight_hh``): Orthogonal
            Biases: 0, except the forget-gate segment of ``bias_ih``,
            set to 1.

        ``MLPHead`` is not reinitialized here; it manages its own
        weight initialization.
        """

        for module in self.modules():

            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        for name, parameter in self.lstm.named_parameters():

            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)

            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)

            elif "bias_ih" in name:
                nn.init.constant_(parameter, 0.0)
                hidden_size = parameter.shape[0] // 4
                parameter.data[hidden_size:2 * hidden_size].fill_(1.0)

            elif "bias_hh" in name:
                nn.init.constant_(parameter, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape
            ``(batch_size, sequence_length, input_channels)``.

        Returns
        -------
        Tensor
            Predicted forecast of shape ``(batch_size, output_dim)``.
        """

        # (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        # ------------------------- Block 1 -------------------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cnn_dropout(x)

        # ------------------------- Block 2 -------------------------
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cnn_dropout(x)

        # (B, C, L) -> (B, L, F)
        x = x.transpose(1, 2)

        # ---------------------------- LSTM ----------------------------
        lstm_output, _ = self.lstm(x)

        # Last output timestep: (B, L, H) -> (B, H)
        last_output = lstm_output[:, -1, :]

        last_output = self.lstm_dropout(last_output)

        prediction = self.mlp_head(last_output)

        return prediction