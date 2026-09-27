"""
Self-contained QUBO lag-selection variant of the hybrid PV power
forecasting architecture.

Experimental context
---------------------
The baseline ``models.proposed_model.ProposedModel`` consumes a
regularly-sampled input of shape ``(B, 24, 7)`` (24 contiguous lookback
timesteps, 7 raw PV/weather features) and is left completely untouched
by this file.

This module instead consumes a QUBO-selected, irregularly-spaced subset
of ``K = 12`` historical lags, each carrying the original 7 raw features
plus one additional ``Lag_Position`` channel (8 features total):

    Selected lags (most-recent-first, as chosen by the QUBO solver):
        [1, 2, 3, 5, 7, 10, 13, 16, 19, 21, 23, 24]

    Chronological order fed to this model (oldest -> most recent):
        [24, 23, 21, 19, 16, 13, 10, 7, 5, 3, 2, 1]

    Input  shape: (batch_size, 12, 8)
    Output shape: (batch_size, 3)          # unchanged 15-minute horizon

``Lag_Position`` (= original_lag / LOOKBACK, computed upstream during
preprocessing) is treated purely as the 8th raw input feature. It gives
the model each retained timestep's absolute position in the original
24-step lookback window, but it does NOT explicitly encode the elapsed
gap between consecutive retained (irregularly-spaced) observations.
That is a preprocessing-level design decision this file preserves
as-is; no additional temporal encoding is introduced here to compensate
for it.

Architectural isolation
------------------------
Every component below (DCNN, FeatureAttention, ResidualBiLSTM,
TemporalAttention, ScalarGatedFusion, MLPHead) is a faithful,
self-contained reproduction of the corresponding class in
``models/dcnn.py``, ``models/feature_attention.py``,
``models/residual_bilstm.py``, ``models/temporal_attention.py``,
``models/scalar_gated_fusion.py``, and ``models/mlp_head.py``
respectively -- copied rather than imported so that this experiment
remains fully isolated from, and cannot accidentally alter, the
baseline model files. No hidden dimensions, layer counts, activation
functions, attention mechanisms, fusion mechanisms, or MLP structure
have been changed. The only adaptation relative to the baseline is the
raw input channel count (8 instead of 7), which is the sole
architectural consequence of adding the ``Lag_Position`` channel and
reducing the sequence length from 24 to 12; every downstream shape
(DCNN output channels, BiLSTM hidden size, fused embedding dimension,
MLP output dimension) is identical to the baseline because none of
those components depend on the raw channel count or on a sequence
length of exactly 24.

Architecture (identical topology to ProposedModel)
----------------------------------------------------
Input (B, 12, 8)
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
      PV Power Prediction (B, 3)

This file intentionally excludes the PHN quantum extension (VQCBranch /
LearnedScalarOutputFusion), FMQA, and any hyperparameter-optimization
scaffolding (random search, Bayesian optimization, etc.). It implements
exactly one thing: the classical architecture re-targeted at the
QUBO-selected 12-lag, 8-channel input contract.
"""

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# QUBO Input Contract (explicit, self-contained — does not read from or
# mutate configs.config, so the baseline's config.NUM_FEATURES == 7 and
# config.LOOKBACK == 24 are left completely unaffected)
# =============================================================================

QUBO_SEQUENCE_LENGTH = 12          # K: number of QUBO-selected lags
QUBO_INPUT_CHANNELS = 8            # 7 raw PV/weather features + Lag_Position
QUBO_SELECTED_LAGS = [1, 2, 3, 5, 7, 10, 13, 16, 19, 21, 23, 24]
QUBO_CHRONOLOGICAL_LAG_ORDER = [24, 23, 21, 19, 16, 13, 10, 7, 5, 3, 2, 1]


# =============================================================================
# DCNN  (reproduced from models/dcnn.py)
#
# Input Shape:  (batch_size, sequence_length, input_channels)
# Output Shape: (batch_size, sequence_length, num_filters)
# =============================================================================

class DCNN(nn.Module):
    """
    Dilated Convolutional Neural Network (DCNN) for local temporal
    feature extraction.

    Identical to ``models.dcnn.DCNN``. ``padding="same"`` preserves the
    sequence length regardless of its value, so this is verified
    compatible with ``sequence_length = 12`` without any change to
    ``kernel_size``, ``dilation_rate``, or the two-block structure.
    Effective receptive field: ``1 + (kernel_size - 1) * dilation_rate``.

    Parameters
    ----------
    input_channels : Number of input features at each timestep.
        Set to 8 for the QUBO variant (7 raw features + Lag_Position),
        vs. 7 in the baseline.

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


# =============================================================================
# Feature Attention  (reproduced from models/feature_attention.py)
#
# Input Shape:  (batch_size, sequence_length, num_features)
# Output Shape: (batch_size, sequence_length, num_features)
# =============================================================================

class FeatureAttention(nn.Module):
    """
    Channel-wise Feature Attention using the Squeeze-and-Excitation (SE)
    mechanism.

    Identical to ``models.feature_attention.FeatureAttention``. Operates
    on the DCNN's learned ``num_filters``-channel representation, not on
    the raw 7/8-channel input, so it is entirely unaffected by the
    QUBO variant's extra raw input channel.

    Parameters
    ----------
    num_features : Number of feature channels produced by the DCNN.

    reduction_ratio : Reduction ratio used in the excitation network.
    """

    def __init__(
        self,
        num_features: int,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()

        if reduction_ratio <= 0:
            raise ValueError(
                "reduction_ratio must be greater than 0."
            )

        hidden_features = max(1, num_features // reduction_ratio)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(
            in_features=num_features,
            out_features=hidden_features,
        )

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=num_features,
        )

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize Linear layer weights using Kaiming Normal initialization.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )

                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape:
            (batch_size, sequence_length, num_features)

        Returns
        -------
        Tensor
            Attention-refined tensor of shape:
            (batch_size, sequence_length, num_features)
        """

        identity = x

        # (B, L, F) -> (B, F, L)
        attention_weights = x.transpose(1, 2)

        # (B, F, L) -> (B, F, 1)
        attention_weights = self.global_avg_pool(attention_weights)

        # (B, F, 1) -> (B, F)
        attention_weights = attention_weights.squeeze(-1)

        # Excitation Network
        attention_weights = self.fc1(attention_weights)
        attention_weights = self.relu(attention_weights)

        attention_weights = self.fc2(attention_weights)
        attention_weights = self.sigmoid(attention_weights)

        # (B, F) -> (B, 1, F)
        attention_weights = attention_weights.unsqueeze(1)

        output = identity * attention_weights

        return output


# =============================================================================
# Residual BiLSTM  (reproduced from models/residual_bilstm.py)
#
# Input Shape:  (batch_size, sequence_length, input_size)
# Output Shape: (batch_size, sequence_length, hidden_size * 2)
# =============================================================================

class ResidualBiLSTM(nn.Module):
    """
    Single-layer Bidirectional LSTM with a residual projection
    connection.

    Identical to ``models.residual_bilstm.ResidualBiLSTM``. Sequence
    length is dynamic (the LSTM and the residual ``Linear``/``Identity``
    projection both operate per-timestep), so ``sequence_length = 12``
    requires no change. The only QUBO-specific adaptation is
    ``input_size = 8`` (vs. 7 in the baseline), which simply causes the
    residual projection to be built as ``nn.Linear(8, hidden_size * 2)``
    instead of ``nn.Linear(7, hidden_size * 2)`` -- the same mechanism,
    a different ``in_features``.

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


# =============================================================================
# Temporal Attention  (reproduced from models/temporal_attention.py)
#
# Input Shape:  (batch_size, sequence_length, embedding_dim)
# Output Shape: (batch_size, sequence_length, embedding_dim)
# =============================================================================

class TemporalAttention(nn.Module):
    """
    Lightweight Learnable Temporal Attention.

    Identical to ``models.temporal_attention.TemporalAttention``. The
    softmax is taken over ``dim=1`` (the sequence dimension), which is
    fully dynamic, so ``sequence_length = 12`` requires no change. This
    module does not encode elapsed time or lag position itself -- it
    only learns a per-timestep importance score over whatever sequence
    it is given; the ``Lag_Position`` feature (if the caller included
    it in the raw input) is handled upstream, not here.

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


# =============================================================================
# Scalar Gated Fusion  (reproduced from models/scalar_gated_fusion.py)
#
# Inputs:
#     Spatial Features  : (batch_size, sequence_length, embedding_dim)
#     Temporal Features : (batch_size, sequence_length, embedding_dim)
# Output:
#     Fused Features    : (batch_size, sequence_length, embedding_dim)
# =============================================================================

class ScalarGatedFusion(nn.Module):
    """
    Scalar Gated Fusion.

    Identical to ``models.scalar_gated_fusion.ScalarGatedFusion``. Both
    branch summaries are computed via ``mean(dim=1)``, which is dynamic
    over sequence length, and the module never inspects the raw feature
    count -- only ``spatial_dim`` / ``temporal_dim`` (the DCNN filter
    count and ``hidden_size * 2``, both unchanged from the baseline).

    Parameters
    ----------
    spatial_dim : Feature dimension of the spatial (DCNN) branch.

    temporal_dim : Feature dimension of the temporal (BiLSTM) branch.
    """

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
    ) -> None:
        super().__init__()

        self.spatial_projection = (
            nn.Identity()
            if spatial_dim == temporal_dim
            else nn.Linear(
                in_features=spatial_dim,
                out_features=temporal_dim,
            )
        )

        self.gate_generator = nn.Linear(
            in_features=2 * temporal_dim,
            out_features=1,
        )

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize Linear layer weights.

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
        spatial_features: Tensor,
        temporal_features: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        spatial_features : Tensor
            Spatial representations from the Feature Attention branch.

            Shape:
            (batch_size, sequence_length, embedding_dim)

        temporal_features : Tensor
            Temporal representations from the Temporal Attention branch.

            Shape:
            (batch_size, sequence_length, embedding_dim)

        Returns
        -------
        Tensor
            Fused representation.

            Shape:
            (batch_size, sequence_length, embedding_dim)
        """

        spatial_features = self.spatial_projection(spatial_features)

        # Global representation of each branch
        spatial_summary = spatial_features.mean(dim=1)

        temporal_summary = temporal_features.mean(dim=1)

        # Concatenate branch summaries
        fusion_summary = torch.cat(
            [spatial_summary, temporal_summary],
            dim=1,
        )

        # Learn scalar gate
        gate = self.gate_generator(fusion_summary)

        gate = self.sigmoid(gate)

        # (B, 1) -> (B, 1, 1)
        gate = gate.unsqueeze(-1)

        # Adaptive fusion
        fused_features = (
            gate * temporal_features
            + (1.0 - gate) * spatial_features
        )

        return fused_features


# =============================================================================
# MLP Head  (reproduced from models/mlp_head.py)
#
# Input Shape:  (batch_size, embedding_dim) or
#               (batch_size, sequence_length, embedding_dim)
# Output Shape: (batch_size, output_dim)
# =============================================================================

class MLPHead(nn.Module):
    """
    Shallow MLP prediction head.

    Identical to ``models.mlp_head.MLPHead``. Pools over the sequence
    dimension internally when given a 3D tensor, so it is unaffected by
    the change from 24 to 12 timesteps. ``output_dim`` stays at 3
    (the unchanged 15-minute horizon).

    Parameters
    ----------
    input_dim : Dimension of the fused feature representation.

    hidden_dim : Number of neurons in the hidden layer.

    dropout_rate : Dropout probability applied after the hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                "dropout_rate must satisfy 0.0 <= dropout_rate < 1.0."
            )

        self.hidden_layer = nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
        )

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(
            p=dropout_rate,
        )

        self.output_layer = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize network weights.

        Hidden Layer
        ------------
        Kaiming Normal Initialization

        Output Layer
        ------------
        Xavier Uniform Initialization

        Bias
        ----
        Initialized to 0.
        """

        nn.init.kaiming_normal_(
            self.hidden_layer.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.constant_(self.hidden_layer.bias, 0.0)

        nn.init.xavier_uniform_(
            self.output_layer.weight
        )
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(
        self,
        fused_features: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        fused_features : Tensor
            Fused spatio-temporal representation, either already
            pooled to (batch_size, input_dim) or unpooled with shape
            (batch_size, sequence_length, input_dim).

        Returns
        -------
        Tensor
            Predicted PV power.

            Shape:
            (batch_size, output_dim)
        """

        if fused_features.dim() == 3:
            # Global Average Pooling over the temporal dimension
            pooled_features = fused_features.mean(dim=1)
        elif fused_features.dim() == 2:
            # Already pooled by the caller (e.g. CNN's global average
            # pool, or the LSTM branches' last-timestep extraction)
            pooled_features = fused_features
        else:
            raise ValueError(
                f"MLPHead expects a 2D (batch_size, embedding_dim) or "
                f"3D (batch_size, sequence_length, embedding_dim) "
                f"tensor; got a {fused_features.dim()}D tensor of "
                f"shape {tuple(fused_features.shape)}."
            )

        hidden_features = self.hidden_layer(
            pooled_features
        )

        hidden_features = self.relu(
            hidden_features
        )

        hidden_features = self.dropout(
            hidden_features
        )

        prediction = self.output_layer(
            hidden_features
        )

        return prediction


# =============================================================================
# ProposedModelQUBO
# =============================================================================

class ProposedModelQUBO(nn.Module):
    """
    QUBO-lag-selection variant of the hybrid PV forecasting model.

    Reproduces ``models.proposed_model.ProposedModel``'s topology and
    default hyperparameters exactly (DCNN -> Feature Attention in
    parallel with Residual BiLSTM -> Temporal Attention, fused via
    Scalar Gated Fusion, predicted via a shallow MLP head), re-targeted
    at the QUBO-selected input contract:

        Input  : (batch_size, 12, 8)
        Output : (batch_size, 3)

    All hyperparameter defaults below match ``configs.config``'s
    baseline defaults (``DCNN_FILTERS=64``, ``DCNN_KERNEL_SIZE=3``,
    ``DCNN_DILATION_RATE=2``, ``DCNN_DROPOUT_RATE=0.20``,
    ``BILSTM_HIDDEN_SIZE=64``, ``BILSTM_DROPOUT_RATE=0.20``,
    ``MLP_HIDDEN_DIM=64``, ``MLP_DROPOUT_RATE=0.20``,
    ``FEATURE_ATTENTION_REDUCTION=8``), but are declared as explicit
    constructor defaults here rather than imported from
    ``configs.config``, so this model has its own self-contained input
    contract and does not depend on (or risk being perturbed by) any
    future change to ``config.NUM_FEATURES`` or ``config.LOOKBACK``,
    which remain reserved for the baseline pipeline.

    Parameters
    ----------
    input_channels : int, default 8
        Raw feature channels per timestep (7 PV/weather features +
        Lag_Position). Exposed explicitly (rather than hardcoded) so
        the input contract is visible at the call site, but should not
        be changed from 8 without revisiting the QUBO preprocessing
        contract described in the module docstring.

    sequence_length : int, default 12
        Number of QUBO-selected lags (K). Not enforced at construction
        time (the underlying layers are sequence-length-agnostic), but
        documented here as the intended value; passed through only for
        validation in ``forward``.

    dcnn_filters, dcnn_kernel_size, dcnn_dilation_rate,
    dcnn_dropout_rate, bilstm_hidden_size, bilstm_dropout_rate,
    mlp_hidden_dim, mlp_dropout_rate, feature_attention_reduction :
        Architecture hyperparameters, identical in meaning and default
        value to their counterparts in ``ProposedModel`` /
        ``configs.config``.

    output_dim : int, default 3
        Forecast horizon output dimension. Left at 3 to match the
        existing 15-minute horizon (``config.HORIZON_TO_OUTPUT_DIM["15"]``).

    use_feature_attention, use_temporal_attention,
    use_scalar_gated_fusion : bool, default True
        Ablation switches, mirroring ``ProposedModel``'s constructor.
        Provided for experimental parity; the QUBO experiment itself
        is expected to run with all three left at their default of
        True.
    """

    def __init__(
        self,
        input_channels: int = QUBO_INPUT_CHANNELS,
        sequence_length: int = QUBO_SEQUENCE_LENGTH,
        dcnn_filters: int = 64,
        dcnn_kernel_size: int = 3,
        dcnn_dilation_rate: int = 2,
        dcnn_dropout_rate: float = 0.20,
        bilstm_hidden_size: int = 64,
        bilstm_dropout_rate: float = 0.20,
        mlp_hidden_dim: int = 64,
        mlp_dropout_rate: float = 0.20,
        feature_attention_reduction: int = 8,
        output_dim: int = 3,
        use_feature_attention: bool = True,
        use_temporal_attention: bool = True,
        use_scalar_gated_fusion: bool = True,
    ) -> None:

        super().__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length

        self.use_feature_attention = use_feature_attention
        self.use_temporal_attention = use_temporal_attention
        self.use_scalar_gated_fusion = use_scalar_gated_fusion

        # ------------------------------------------------------------
        # Spatial Branch
        # ------------------------------------------------------------
        self.dcnn = DCNN(
            input_channels=input_channels,
            num_filters=dcnn_filters,
            kernel_size=dcnn_kernel_size,
            dilation_rate=dcnn_dilation_rate,
            dropout_rate=dcnn_dropout_rate,
        )

        self.feature_attention = FeatureAttention(
            num_features=dcnn_filters,
            reduction_ratio=feature_attention_reduction,
        )

        # ------------------------------------------------------------
        # Temporal Branch
        # ------------------------------------------------------------
        self.residual_bilstm = ResidualBiLSTM(
            input_size=input_channels,
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
            output_dim=output_dim,
            dropout_rate=mlp_dropout_rate,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            QUBO-selected input tensor of shape
            (batch_size, sequence_length, input_channels), i.e.
            (batch_size, 12, 8) under the established contract.

        Returns
        -------
        Tensor
            Predicted PV power, shape (batch_size, output_dim), i.e.
            (batch_size, 3) for the 15-minute horizon.
        """

        if x.dim() != 3 or x.size(-1) != self.input_channels:
            raise ValueError(
                f"ProposedModelQUBO expects input of shape "
                f"(batch_size, sequence_length, {self.input_channels}); "
                f"got {tuple(x.shape)}."
            )

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
            # 0.5 / 0.5 average. Mirrors ProposedModel's ablation
            # behaviour exactly.
            projected_spatial = self.scalar_gated_fusion.spatial_projection(
                spatial_features
            )
            fused_features = (
                0.5 * projected_spatial + 0.5 * temporal_features
            )

        # ---------------- Prediction ----------------

        prediction = self.mlp_head(fused_features)

        return prediction