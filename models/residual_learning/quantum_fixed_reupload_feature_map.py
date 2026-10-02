"""Fixed, zero-angle quantum re-uploading map with trainable classical mappings."""

import torch

from models.residual_learning.quantum_residual_reupload_vqc import QuantumResidualReuploadVQC


class QuantumFixedReuploadFeatureMap(QuantumResidualReuploadVQC):
    """Keep both encodings and rings; replace variational angles with a zero buffer.

    Parent initialization preserves exactly the comparator's projection/readout
    random draws. Its temporary quantum draws are discarded, never frozen at random.
    """

    def __init__(self) -> None:
        super().__init__()
        del self.weights
        self.register_buffer("weights", torch.zeros(2, 6, 2))

    def parameter_counts(self) -> dict[str, int]:
        """Count trainable parameters; fixed angles are persisted buffers."""
        if self.weights.requires_grad or torch.count_nonzero(self.weights).item():
            raise ValueError("Fixed quantum rotation angles must remain zero and nontrainable.")
        counts = {"projection": sum(p.numel() for p in self.projection.parameters() if p.requires_grad),
                  "quantum": 0,
                  "readout": sum(p.numel() for p in self.readout.parameters() if p.requires_grad)}
        counts["total"] = sum(counts.values())
        return counts
