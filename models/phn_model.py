"""
phn_model.py

Independent, top-level Parallel Hybrid Network (PHN) model for
short-term photovoltaic (PV) power forecasting.

This is a genuinely parallel PHN, distinct from both:

  * the existing ``proposed`` model (classical-only), and
  * the existing ``proposed_phn`` model (models/proposed_model.py with
    use_quantum_branch=True), where the VQC receives pooled,
    classically-refined (SGF-output) features rather than raw input.

Architecture
------------

                          Raw Input
                          (B, 24, 6)
                              │
                 ┌────────────┴────────────┐
                 │                          │
                 ▼                          ▼
        Existing Classical              PHNVQCBranch
        Hybrid Architecture           (raw input, flatten
        (DCNN -> FeatureAttention,     -> existing VQCBranch)
         ResidualBiLSTM ->
         TemporalAttention ->
         ScalarGatedFusion ->
         MLPHead)
                 │                          │
                 ▼                          ▼
               y_c (B,3)                 y_q (B,3)
                 └────────────┬────────────┘
                              ▼
                LearnedScalarOutputFusion
                              ▼
                     Final prediction (B,3)

Both branches independently receive the SAME raw (B, 24, 6) input.
Neither branch's output or intermediate features feed the other:

  * The classical branch never sees the VQC's output.
  * The VQC never sees any classical intermediate tensor (no DCNN
    output, no BiLSTM output, no SGF output, no MLPHead output feeds
    it) -- only the original raw input, flattened.

The only interaction between the two branches is at the very end:
``y_c`` and ``y_q`` are combined by the existing, unmodified
``LearnedScalarOutputFusion``.

File isolation
---------------
This module does NOT modify, subclass, or import behavior-altering
logic from ``models/proposed_model.py``. The classical branch here is
built from the same underlying submodules (``DCNN``,
``FeatureAttention``, ``ResidualBiLSTM``, ``TemporalAttention``,
``ScalarGatedFusion``, ``MLPHead``) with the same configuration
``ProposedModel`` uses, but is composed independently in this file.
``ProposedModel`` itself is untouched, and the existing ``proposed``
and ``proposed_phn`` model_factory paths continue to instantiate it
exactly as before.

Interface verification
------------------------
Fully verified by real execution against the actual uploaded source of
every dependency: DCNN, FeatureAttention, ResidualBiLSTM,
TemporalAttention, ScalarGatedFusion, MLPHead, VQCBranch, and
LearnedScalarOutputFusion (all real files, none stubbed). Confirmed
in a local Python/PyTorch/PennyLane environment:

  * Real input shape is (B, 24, 7) -- config.NUM_FEATURES is 7, not 6
    (FEATURE_COLUMNS includes Active_Power as both a lagged input
    feature and the prediction target). PHNModel and PHNVQCBranch
    both derive dimensions from config at runtime, so this required
    no code change.
  * Full forward pass: PHNModel()(torch.randn(4,24,7)) ->
    torch.Size([4, 3]), confirmed.
  * Full backward pass: gradients confirmed present on
    dcnn.conv1.weight, mlp_head.output_layer.weight (classical
    branch), phn_vqc_branch.vqc_branch.projection.weight and
    phn_vqc_branch.vqc_branch.weights -- the real PennyLane circuit's
    variational parameters (quantum branch), and
    output_fusion.classical_logit / .quantum_logit (fusion layer).
  * VQCBranch's internal CPU-pinning (_apply override) exercised via
    model.to("cpu") -- weights confirmed to remain on CPU device
    correctly. NOTE: no CUDA device is available in this sandbox, so
    the actual GPU->CPU->GPU transfer path for `angles` (as opposed
    to the permanently-CPU-pinned `weights`) was NOT exercised
    end-to-end; only the CPU-only code path was run.
  * models/proposed_model.py and models/vqc_branch.py confirmed
    byte-identical (md5) to the originally uploaded files -- untouched
    by this implementation.
  * Both "proposed" and "proposed_phn" confirmed to still instantiate
    and run correctly (forward+backward) via the real, unmodified
    ProposedModel, and via model_factory.get_model() with the patch
    applied.
  * "phn" confirmed to instantiate and run correctly via
    model_factory.get_model("phn") with the patch applied (tested
    against a copy of the real model_factory.py; models.comparison_models
    was stubbed only because its source was never uploaded and is
    unrelated to phn/proposed/proposed_phn).
"""

import torch.nn as nn
from torch import Tensor

from configs import config

from models.dcnn import DCNN
from models.feature_attention import FeatureAttention
from models.residual_bilstm import ResidualBiLSTM
from models.temporal_attention import TemporalAttention
from models.scalar_gated_fusion import ScalarGatedFusion
from models.mlp_head import MLPHead
from models.learned_scalar_output_fusion import LearnedScalarOutputFusion

from models.phn_vqc_branch import PHNVQCBranch


class PHNModel(nn.Module):
    """
    Independent, top-level Parallel Hybrid Network model.

    Parameters
    ----------
    dcnn_filters, dcnn_kernel_size, dcnn_dilation_rate,
    dcnn_dropout_rate, bilstm_hidden_size, bilstm_dropout_rate,
    mlp_hidden_dim, mlp_dropout_rate : optional
        Classical-branch hyperparameters. Any left as ``None`` fall
        back to ``configs.config`` defaults, exactly mirroring
        ``ProposedModel``'s own fallback behavior, so that the PHN's
        classical branch matches the existing Proposed model's
        architecture and configuration unless explicitly overridden.
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

        # Any parameter left as None falls back to config.py, matching
        # ProposedModel()'s own default-resolution behavior exactly.
        dcnn_filters = dcnn_filters if dcnn_filters is not None else config.DCNN_FILTERS
        dcnn_kernel_size = dcnn_kernel_size if dcnn_kernel_size is not None else config.DCNN_KERNEL_SIZE
        dcnn_dilation_rate = dcnn_dilation_rate if dcnn_dilation_rate is not None else config.DCNN_DILATION_RATE
        dcnn_dropout_rate = dcnn_dropout_rate if dcnn_dropout_rate is not None else config.DCNN_DROPOUT_RATE
        bilstm_hidden_size = bilstm_hidden_size if bilstm_hidden_size is not None else config.BILSTM_HIDDEN_SIZE
        bilstm_dropout_rate = bilstm_dropout_rate if bilstm_dropout_rate is not None else config.BILSTM_DROPOUT_RATE
        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else config.MLP_HIDDEN_DIM
        mlp_dropout_rate = mlp_dropout_rate if mlp_dropout_rate is not None else config.MLP_DROPOUT_RATE

        # ------------------------------------------------------------
        # Classical branch: the existing hybrid deep-learning
        # architecture, composed independently here (NOT via
        # ProposedModel), using the same submodules and configuration.
        # Constructor kwargs and forward() call order verified by
        # real execution -- see module docstring.
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

        self.residual_bilstm = ResidualBiLSTM(
            input_size=config.NUM_FEATURES,
            hidden_size=bilstm_hidden_size,
            dropout_rate=bilstm_dropout_rate,
        )

        self.temporal_attention = TemporalAttention(
            embedding_dim=bilstm_hidden_size * 2,
        )

        self.scalar_gated_fusion = ScalarGatedFusion(
            spatial_dim=dcnn_filters,
            temporal_dim=bilstm_hidden_size * 2,
        )

        self.mlp_head = MLPHead(
            input_dim=bilstm_hidden_size * 2,
            hidden_dim=mlp_hidden_dim,
            output_dim=config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON],
            dropout_rate=mlp_dropout_rate,
        )

        # ------------------------------------------------------------
        # Quantum branch: independently receives the SAME raw input,
        # via the minimal flatten -> existing VQCBranch mapping.
        # ------------------------------------------------------------
        self.phn_vqc_branch = PHNVQCBranch(
            flat_input_dim=config.LOOKBACK * config.NUM_FEATURES,
            num_qubits=config.VQC_NUM_QUBITS,
            depth=config.VQC_DEPTH,
            output_dim=config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON],
            simulator=config.VQC_SIMULATOR,
            diff_method=config.VQC_DIFF_METHOD,
        )

        # ------------------------------------------------------------
        # Output fusion: combines y_c and y_q only. Never used to
        # combine intermediate features -- that role belongs to
        # ScalarGatedFusion above, which stays entirely inside the
        # classical branch.
        # ------------------------------------------------------------
        self.output_fusion = LearnedScalarOutputFusion()

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Raw input, shape (batch_size, 24, 6).

        Returns
        -------
        Tensor
            Final fused prediction, shape (batch_size, 3).
        """

        # ---------------- Classical branch ----------------

        spatial_features = self.dcnn(x)
        spatial_features = self.feature_attention(spatial_features)

        temporal_features = self.residual_bilstm(x)
        temporal_features = self.temporal_attention(temporal_features)

        fused_features = self.scalar_gated_fusion(
            spatial_features,
            temporal_features,
        )

        classical_prediction = self.mlp_head(fused_features)

        # ---------------- Quantum branch ----------------
        # Receives the ORIGINAL raw input x directly -- not
        # spatial_features, temporal_features, fused_features, or
        # classical_prediction.

        quantum_prediction = self.phn_vqc_branch(x)

        # ---------------- Output fusion ----------------

        final_prediction = self.output_fusion(
            classical_prediction,
            quantum_prediction,
        )

        return final_prediction