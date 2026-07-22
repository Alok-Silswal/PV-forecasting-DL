"""
Complete sequential hybrid architecture for short-term photovoltaic (PV) power forecasting.

Architecture
------------
Input
│
▼
DCNN
│
▼
(Optional) Feature Attention
│
▼
Residual BiLSTM
│
▼
(Optional) Temporal Attention
│
▼
Shallow MLP Head
│
▼
PV Power Prediction

Unlike ProposedModel, this architecture contains no parallel branches,
no feature fusion, and no scalar gated fusion. The DCNN and Residual
BiLSTM are composed sequentially: the DCNN's spatial feature maps are
fed directly into the Residual BiLSTM as its input sequence.
"""

import torch.nn as nn

from configs import config

from models.dcnn import DCNN
from models.feature_attention import FeatureAttention
from models.residual_bilstm import ResidualBiLSTM
from models.temporal_attention import TemporalAttention
from models.mlp_head import MLPHead

from torch import Tensor


class DCNNResidualBiLSTM(nn.Module):
    """
    Complete sequential PV forecasting model.
    """

    def __init__(
        self,
        dcnn_filters: int | None = None,
        dcnn_kernel_size: int | None = None,
        dcnn_dilation_rate: int | None = None,
        dcnn_dropout_rate: float | None = None,
        bilstm_hidden_size: int | None = None,
        bilstm_dropout_rate: float | None = None,
        mlp_hidden_dim: int | None = None,
        mlp_dropout_rate: float | None = None,
        use_feature_attention: bool = True,
        use_temporal_attention: bool = True,
    ) -> None:

        super().__init__()

        self.use_feature_attention = use_feature_attention
        self.use_temporal_attention = use_temporal_attention

        # Any parameter left as None falls back to config.py, preserving
        # SequentialProposedModel() as fully equivalent to the default
        # configured behavior.
        dcnn_filters = dcnn_filters if dcnn_filters is not None else config.DCNN_FILTERS
        dcnn_kernel_size = dcnn_kernel_size if dcnn_kernel_size is not None else config.DCNN_KERNEL_SIZE
        dcnn_dilation_rate = dcnn_dilation_rate if dcnn_dilation_rate is not None else config.DCNN_DILATION_RATE
        dcnn_dropout_rate = dcnn_dropout_rate if dcnn_dropout_rate is not None else config.DCNN_DROPOUT_RATE
        bilstm_hidden_size = bilstm_hidden_size if bilstm_hidden_size is not None else config.BILSTM_HIDDEN_SIZE
        bilstm_dropout_rate = bilstm_dropout_rate if bilstm_dropout_rate is not None else config.BILSTM_DROPOUT_RATE
        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else config.MLP_HIDDEN_DIM
        mlp_dropout_rate = mlp_dropout_rate if mlp_dropout_rate is not None else config.MLP_DROPOUT_RATE

        # ------------------------------------------------------------
        # Spatial Stage
        # ------------------------------------------------------------
        self.dcnn = DCNN(
            input_channels=config.NUM_FEATURES,
            num_filters=dcnn_filters,
            kernel_size=dcnn_kernel_size,
            dilation_rate=dcnn_dilation_rate,
            dropout_rate=dcnn_dropout_rate,
        )

        if self.use_feature_attention:
            self.feature_attention = FeatureAttention(
                num_features=dcnn_filters,
                reduction_ratio=config.FEATURE_ATTENTION_REDUCTION,
        )

        # ------------------------------------------------------------
        # Temporal Stage
        # ------------------------------------------------------------
        # Consumes the DCNN's (optionally attention-refined) spatial
        # feature maps as its input sequence, rather than the raw
        # input features used by the parallel ProposedModel.
        self.residual_bilstm = ResidualBiLSTM(
            input_size=dcnn_filters,
            hidden_size=bilstm_hidden_size,
            dropout_rate=bilstm_dropout_rate,
        )

        if self.use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                embedding_dim=bilstm_hidden_size * 2,
        )

        # ------------------------------------------------------------
        # Prediction Head
        # ------------------------------------------------------------
        self.mlp_head = MLPHead(
            input_dim=bilstm_hidden_size * 2,
            hidden_dim=mlp_hidden_dim,
            output_dim=config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON],
            dropout_rate=mlp_dropout_rate,
        )

    def forward(self, x: Tensor) -> Tensor:

        # ---------------- Spatial Stage ----------------

        x = self.dcnn(x)

        if self.use_feature_attention:
            x = self.feature_attention(x)

        # ---------------- Temporal Stage ----------------

        x = self.residual_bilstm(x)

        if self.use_temporal_attention:
            x = self.temporal_attention(x)
        else:
            # Use final BiLSTM timestep
            x = x[:, -1, :]

        # ---------------- Prediction ----------------

        prediction = self.mlp_head(x)

        return prediction