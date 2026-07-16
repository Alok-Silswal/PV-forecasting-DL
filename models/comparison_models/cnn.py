"""
CNN baseline architecture for short-term photovoltaic (PV) power
forecasting.

Architecture
------------
Input
│
Conv1D → BatchNorm → ReLU → Dropout
│
Conv1D → BatchNorm → ReLU → Dropout
│
Global Average Pooling
│
MLP Head
│
Forecast

This model is a spatial-only baseline: it has no attention, no
recurrent layers, no residual connections, and no feature fusion. It
uses standard convolutions only (dilation=1). The proposed
architecture's DCNN branch is the only component in this project that
uses dilated convolutions; keeping this baseline non-dilated ensures
the comparison does not already contain one of the proposed
architecture's own contributions. It exists purely for comparison
against the proposed hybrid architecture.
"""

import torch.nn as nn
from torch import Tensor

from configs import config

from models.mlp_head import MLPHead


class CNN(nn.Module):
    """
    Spatial baseline PV forecasting model.

    Composes two standard (non-dilated) Conv1D blocks for local
    temporal feature extraction, global average pooling to collapse
    the sequence dimension, and ``MLPHead`` for the final forecast.

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
        mlp_hidden_dim: int | None = None,
        mlp_dropout_rate: float | None = None,
    ) -> None:

        super().__init__()

        # Any parameter left as None falls back to config.py, preserving
        # CNN() as fully equivalent to prior behavior.
        num_filters = num_filters if num_filters is not None else config.DCNN_FILTERS
        kernel_size = kernel_size if kernel_size is not None else config.DCNN_KERNEL_SIZE
        cnn_dropout_rate = cnn_dropout_rate if cnn_dropout_rate is not None else config.DCNN_DROPOUT_RATE
        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else config.MLP_HIDDEN_DIM
        mlp_dropout_rate = mlp_dropout_rate if mlp_dropout_rate is not None else config.MLP_DROPOUT_RATE

        if not (0.0 <= cnn_dropout_rate < 1.0):
            raise ValueError(
                "cnn_dropout_rate must satisfy 0.0 <= cnn_dropout_rate < 1.0."
            )

        input_channels = config.NUM_FEATURES
        output_dim = config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON]

        self.num_filters = num_filters

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
        # Shared Layers
        # ------------------------------------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.cnn_dropout = nn.Dropout(p=cnn_dropout_rate)
        self.global_average_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # ------------------------------------------------------------------
        # Prediction Head
        # ------------------------------------------------------------------
        self.mlp_head = MLPHead(
            input_dim=num_filters,
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
        x = self.dropout(x)

        # ---------------- Global Average Pooling ----------------
        pooled = self.global_average_pool(x)

        # (B, C, 1) -> (B, C)
        pooled = pooled.squeeze(-1)

        prediction = self.mlp_head(pooled)

        return prediction