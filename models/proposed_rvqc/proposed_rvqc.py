"""Jointly trained ProposedModel plus the existing analytic re-uploading RVQC."""

from typing import Any

import torch
from torch import nn

from models.proposed_model import ProposedModel
from models.residual_learning.quantum_residual_reupload_vqc import QuantumResidualReuploadVQC


class ProposedRVQC(nn.Module):
    """Keep the backbone on its chosen device and the complete RVQC on CPU.

    Both transfers in forward preserve autograd. The correction is in normalized
    forecast units; no pilot bias or residual scaler belongs to joint training.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ProposedModel(use_quantum_branch=False)
        self.rvqc = QuantumResidualReuploadVQC()

    def to(self, *args: Any, **kwargs: Any) -> "ProposedRVQC":
        """Move the backbone normally while preserving the CPU simulator contract."""
        self.backbone.to(*args, **kwargs)
        self.rvqc.to(device="cpu", dtype=next(self.backbone.parameters()).dtype)
        return self

    def forward_components(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Call the original forward once and capture its live fusion output."""
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            captured.append(output)
            return None

        hook = self.backbone.scalar_gated_fusion.register_forward_hook(capture)
        try:
            baseline = self.backbone(inputs)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape != (len(inputs), 24, 128):
            raise RuntimeError("Expected exactly one live fusion tensor [B,24,128].")
        fusion = captured[0]
        pooled = fusion.mean(dim=1)
        cpu_latent = pooled.to("cpu")
        correction_cpu = self.rvqc(cpu_latent)
        correction = correction_cpu.to(baseline.device)
        return {"fusion": fusion, "pooled": pooled, "cpu_latent": cpu_latent,
                "baseline": baseline, "correction_cpu": correction_cpu,
                "correction": correction, "final": baseline + correction}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_components(inputs)["final"]
