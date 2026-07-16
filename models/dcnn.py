"""
Input Shape:  (batch_size, sequence_length, input_channels)

Output Shape: (batch_size, sequence_length, num_filters)
"""

import torch
import torch.nn as nn
from torch import Tensor


class DCNN(nn.Module):
    """
    Dilated Convolutional Neural Network (DCNN) for local temporal feature extraction.

    Parameters
    ----------
    input_channels : Number of input features at each timestep.

    num_filters : Number of convolutional filters.

    kernel_size : Size of the temporal convolution kernel.

    dilation_rate : Dilation factor for temporal convolution.

    dropout_rate : Dropout probability applied after each convolution block.
    """

    def __init__(
        self,
        input_channels: int,
        num_filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                "dropout_rate must satisfy 0.0 <= dropout_rate < 1.0."
            )

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
            dilation=dilation_rate,
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
            dilation=dilation_rate,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_filters)

        # ------------------------------------------------------------------
        # Shared Layers
        # ------------------------------------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)

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
            Input tensor of shape:
            (batch_size, sequence_length, input_channels)

        Returns
        -------
        Tensor
            Output tensor of shape:
            (batch_size, sequence_length, num_filters)
        """

        # (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        # ------------------------- Block 1 -------------------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # ------------------------- Block 2 -------------------------
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        return x