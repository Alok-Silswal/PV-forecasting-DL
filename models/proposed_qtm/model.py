"""Final hybrid model: quantum temporal weighting and vector gated fusion."""

import torch
from torch import Tensor, nn

from configs import config
from models.dcnn import DCNN
from models.feature_attention import FeatureAttention
from models.residual_bilstm import ResidualBiLSTM
from models.mlp_head import MLPHead
from .quantum_temporal import QuantumTemporalWeighting
from .vector_gated_fusion import VectorGatedFusion


class ProposedQTM(nn.Module):
    def __init__(self, quantum_backend: str | None = None, seed: int | None = None) -> None:
        super().__init__()
        if (config.LOOKBACK, config.NUM_FEATURES, config.BILSTM_HIDDEN_SIZE,
                config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON], config.MLP_HIDDEN_DIM) != (24, 7, 64, 3, 64):
            raise ValueError("ProposedQTM requires the locked 24-step, 128-wide, 15-minute architecture.")
        seed = config.RANDOM_SEED + config.RUN_NUMBER - 1 if seed is None else seed
        backend = config.QTM_QUANTUM_BACKEND if quantum_backend is None else quantum_backend
        self.dcnn = DCNN(config.NUM_FEATURES, config.DCNN_FILTERS, config.DCNN_KERNEL_SIZE,
                         config.DCNN_DILATION_RATE, config.DCNN_DROPOUT_RATE)
        self.feature_attention = FeatureAttention(config.DCNN_FILTERS, config.FEATURE_ATTENTION_REDUCTION)
        self.spatial_projection = (nn.Identity() if config.DCNN_FILTERS == 128
                                   else nn.Linear(config.DCNN_FILTERS, 128))
        if isinstance(self.spatial_projection, nn.Linear):
            nn.init.xavier_uniform_(self.spatial_projection.weight)
            nn.init.zeros_(self.spatial_projection.bias)
        self.residual_bilstm = ResidualBiLSTM(config.NUM_FEATURES, config.BILSTM_HIDDEN_SIZE,
                                            config.BILSTM_DROPOUT_RATE)
        self.quantum_temporal = QuantumTemporalWeighting(seed, backend)
        self.vector_gated_fusion = VectorGatedFusion(seed)
        self.mlp_head = MLPHead(128, config.MLP_HIDDEN_DIM, config.MLP_DROPOUT_RATE, output_dim=3)

    def optimizer_parameter_groups(self, learning_rate: float, weight_decay: float,
                                   quantum_lr_multiplier: float) -> list[dict]:
        if quantum_lr_multiplier <= 0:
            raise ValueError("Quantum LR multiplier must be positive.")
        quantum = [self.quantum_temporal.quantum.angles, self.quantum_temporal.gamma]
        quantum_ids = {id(parameter) for parameter in quantum}
        classical = [parameter for parameter in self.parameters() if id(parameter) not in quantum_ids]
        return [{"params": classical, "lr": learning_rate, "weight_decay": weight_decay},
                {"params": quantum, "lr": learning_rate * quantum_lr_multiplier, "weight_decay": 0.0}]

    def _forward(self, inputs: Tensor, diagnostics: bool = False,
                 uniform_temporal: bool = False) -> tuple[Tensor, dict]:
        if inputs.ndim != 3 or inputs.shape[1:] != (24, 7):
            raise ValueError("Expected historical inputs [B,24,7].")
        if uniform_temporal and self.training:
            raise ValueError("Uniform temporal weighting is an inference-only diagnostic.")
        spatial = self.spatial_projection(self.feature_attention(self.dcnn(inputs)))
        hidden = self.residual_bilstm(inputs)
        if uniform_temporal:
            temporal, report = hidden, {}
        elif diagnostics:
            temporal, report = self.quantum_temporal.forward_with_diagnostics(hidden)
        else:
            temporal, report = self.quantum_temporal(hidden), {}
        fused = self.vector_gated_fusion(spatial, temporal)
        prediction = self.mlp_head(fused.mean(dim=1))
        return prediction, report

    def forward(self, inputs: Tensor) -> Tensor:
        return self._forward(inputs)[0]

    def forward_with_diagnostics(self, inputs: Tensor) -> tuple[Tensor, dict]:
        """Aggregate one batch without storing samples or rerunning the circuit."""
        return self._forward(inputs, diagnostics=True)

    @torch.no_grad()
    def predict_uniform(self, inputs: Tensor) -> Tensor:
        """Bypass temporal weighting in eval mode, without retraining."""
        return self._forward(inputs, uniform_temporal=True)[0]
