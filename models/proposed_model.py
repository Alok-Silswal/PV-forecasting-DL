"""
Complete hybrid architecture for short-term photovoltaic (PV) power forecasting.

Classical architecture
-----------------------
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

PHN (Parallel Hybrid Network) extension
----------------------------------------
When constructed with ``use_quantum_branch=True`` (as done by the
``proposed_phn`` factory entry), the SGF output additionally feeds a
compact quantum prediction branch, run in parallel with the unchanged
classical MLPHead:

SGF (B, 24, 128)
        │
        ├──────────────────────────┐
        ▼                          ▼
  existing MLPHead           mean(dim=1) -> (B, 128)
  [UNCHANGED]                      │
        │                    Linear(128, 2) -> VQC (2 qubits, depth 2)
        ▼                          │
     y_c (B, 3)                 y_q (B, 3)
        │                          │
        └──────────┬───────────────┘
                    ▼
     Learned Scalar Output Fusion (softmax-constrained s_c, s_q)
                    ▼
              Final prediction (B, 3)

The ``proposed`` model (``use_quantum_branch=False``, the default) is
completely unaffected: the quantum branch and output fusion module are
never constructed and never executed.
"""

import torch.nn as nn

from configs import config

from models.dcnn import DCNN
from models.feature_attention import FeatureAttention
from models.residual_bilstm import ResidualBiLSTM
from models.temporal_attention import TemporalAttention
from models.scalar_gated_fusion import ScalarGatedFusion
from models.mlp_head import MLPHead
from models.vqc_branch import VQCBranch
from models.learned_scalar_output_fusion import LearnedScalarOutputFusion

from torch import Tensor


class ProposedModel(nn.Module):
    """
    Complete hybrid PV forecasting model.

    Parameters
    ----------
    use_quantum_branch : bool, default False
        If True, additionally constructs and runs the PHN quantum
        extension (``VQCBranch`` + ``LearnedScalarOutputFusion``) in
        parallel with the existing, unchanged classical ``MLPHead``.
        The existing ``proposed`` model (and all existing ablations)
        must always be constructed with this left at its default of
        False, which reproduces the exact pre-PHN behavior: the
        quantum branch and output fusion module are not created and
        ``forward`` returns the classical MLPHead output unchanged.
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
        use_scalar_gated_fusion: bool = True,
        use_quantum_branch: bool = False,
    ) -> None:

        super().__init__()

        self.use_feature_attention = use_feature_attention
        self.use_temporal_attention = use_temporal_attention
        self.use_scalar_gated_fusion = use_scalar_gated_fusion
        self.use_quantum_branch = use_quantum_branch

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
        # Prediction Head (unchanged, always constructed and used
        # identically regardless of use_quantum_branch)
        # ------------------------------------------------------------
        self.mlp_head = MLPHead(
            input_dim=bilstm_hidden_size * 2,
            hidden_dim=mlp_hidden_dim,
            output_dim=config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON],
            dropout_rate=mlp_dropout_rate,
        )

        # ------------------------------------------------------------
        # PHN quantum extension (only constructed when requested)
        # ------------------------------------------------------------
        if self.use_quantum_branch:
            self.vqc_branch = VQCBranch(
                input_dim=bilstm_hidden_size * 2,
                output_dim=config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON],
                num_qubits=config.VQC_NUM_QUBITS,
                depth=config.VQC_DEPTH,
                simulator=config.VQC_SIMULATOR,
                diff_method=config.VQC_DIFF_METHOD,
            )
            self.output_fusion = LearnedScalarOutputFusion()

    def forward(self, x: Tensor) -> Tensor:

        # ---------------- Spatial Branch ----------------

        spatial_features = self.dcnn(x)

        if self.use_feature_attention:
            spatial_features = self.feature_attention(
                spatial_features
            )

        # ---------------- Temporal Branch ----------------

        temporal_features = self.residual_bilstm(x)

        if self.use_temporal_attention:
            temporal_features = self.temporal_attention(
                temporal_features
            )

        # ---------------- Fusion ----------------

        if self.use_scalar_gated_fusion:
            fused_features = self.scalar_gated_fusion(
                spatial_features,
                temporal_features,
            )
        else:
            # Reuses ScalarGatedFusion's own spatial_projection so the
            # spatial branch is still mapped into temporal_dim
            # (bilstm_hidden_size * 2) exactly as under gated fusion —
            # only the learned gate is removed, replaced with a fixed
            # 0.5 / 0.5 average.
            projected_spatial = self.scalar_gated_fusion.spatial_projection(
                spatial_features
            )
            fused_features = (
                0.5 * projected_spatial + 0.5 * temporal_features
            )

        # ---------------- Classical Prediction (unchanged) ----------------

        classical_prediction = self.mlp_head(
            fused_features
        )

        if not self.use_quantum_branch:
            return classical_prediction

        # ---------------- Quantum Prediction (PHN only) ----------------

        # The quantum branch performs its OWN temporal mean pooling,
        # independent of (and not feeding into) MLPHead's internal
        # pooling above.
        pooled_features = fused_features.mean(dim=1)

        quantum_prediction = self.vqc_branch(pooled_features)

        # ---------------- Learned Scalar Output Fusion ----------------

        final_prediction = self.output_fusion(
            classical_prediction,
            quantum_prediction,
        )

        return final_prediction