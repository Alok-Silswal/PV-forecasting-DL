"""Matched end-to-end control with the historical zero-angle reupload map."""

from torch import nn

from models.proposed_model import ProposedModel
from models.proposed_rvqc import ProposedRVQC
from models.residual_learning.quantum_fixed_reupload_feature_map import QuantumFixedReuploadFeatureMap


class ProposedReupload(ProposedRVQC):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.backbone = ProposedModel(use_quantum_branch=False)
        self.rvqc = QuantumFixedReuploadFeatureMap()
