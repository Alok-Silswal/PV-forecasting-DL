"""
Complete hybrid architecture for short-term photovoltaic (PV) power forecasting.

Architecture
------------
Input
│
├── DCNN ───────────────► Feature Attention
│
└── Residual BiLSTM ───► Temporal Attention
             │
             ▼
      Scalar Gated Fusion
             │
             ▼
        Shallow MLP Head
             │
             ▼
      PV Power Prediction
"""

import torch.nn as nn

from configs import config

from models.dcnn import DCNN
from models.feature_attention import FeatureAttention
from models.residual_bilstm import ResidualBiLSTM
from models.temporal_attention import TemporalAttention
from models.scalar_gated_fusion import ScalarGatedFusion
from models.mlp_head import MLPHead

from torch import Tensor


class ProposedModel(nn.Module):
    """
    Complete hybrid PV forecasting model.
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
    ) -> None:

        super().__init__()

        # Any parameter left as None falls back to config.py, preserving
        # ProposedModel() as fully equivalent to prior behavior.
        dcnn_filters = dcnn_filters if dcnn_filters is not None else config.DCNN_FILTERS
        dcnn_kernel_size = dcnn_kernel_size if dcnn_kernel_size is not None else config.DCNN_KERNEL_SIZE
        dcnn_dilation_rate = dcnn_dilation_rate if dcnn_dilation_rate is not None else config.DCNN_DILATION_RATE
        dcnn_dropout_rate = dcnn_dropout_rate if dcnn_dropout_rate is not None else config.DCNN_DROPOUT_RATE
        bilstm_hidden_size = bilstm_hidden_size if bilstm_hidden_size is not None else config.BILSTM_HIDDEN_SIZE
        bilstm_dropout_rate = bilstm_dropout_rate if bilstm_dropout_rate is not None else config.BILSTM_DROPOUT_RATE
        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else config.MLP_HIDDEN_DIM
        mlp_dropout_rate = mlp_dropout_rate if mlp_dropout_rate is not None else config.MLP_DROPOUT_RATE

        # ------------------------------------------------------------
        # Spatial Branch
        # ------------------------------------------------------------
        self.dcnn = DCNN(
            input_channels=config.NUM_FEATURES,
            num_filters=dcnn_filters,
            kernel_size=dcnn_kernel_size,
            dilation_rate=dcnn_dilation_rate,
            dropout_rate=dcnn_dropout_rate,
        )

        self.feature_attention = FeatureAttention(
            num_features=dcnn_filters,
            reduction_ratio=config.FEATURE_ATTENTION_REDUCTION,
        )

        # ------------------------------------------------------------
        # Temporal Branch
        # ------------------------------------------------------------
        self.residual_bilstm = ResidualBiLSTM(
            input_size=config.NUM_FEATURES,
            hidden_size=bilstm_hidden_size,
            dropout_rate=bilstm_dropout_rate,
        )

        self.temporal_attention = TemporalAttention(
            embedding_dim=bilstm_hidden_size * 2,
        )

        # ------------------------------------------------------------
        # Fusion
        # ------------------------------------------------------------
        self.scalar_gated_fusion = ScalarGatedFusion(
            spatial_dim=dcnn_filters,
            temporal_dim=bilstm_hidden_size * 2,
        )

        # ------------------------------------------------------------
        # Prediction Head
        # ------------------------------------------------------------
        self.mlp_head = MLPHead(
            input_dim=bilstm_hidden_size * 2,
            hidden_dim=mlp_hidden_dim,
            dropout_rate=mlp_dropout_rate,
        )

    def forward(self, x: Tensor) -> Tensor:

        # ---------------- Spatial Branch ----------------

        spatial_features = self.dcnn(x)

        spatial_features = self.feature_attention(
            spatial_features
        )

        # ---------------- Temporal Branch ----------------

        temporal_features = self.residual_bilstm(x)

        temporal_features = self.temporal_attention(
            temporal_features
        )

        # ---------------- Fusion ----------------

        fused_features = self.scalar_gated_fusion(
            spatial_features,
            temporal_features,
        )

        # ---------------- Prediction ----------------

        prediction = self.mlp_head(
            fused_features
        )

        return prediction