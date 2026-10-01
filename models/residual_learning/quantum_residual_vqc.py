"""Isolated, analytic six-qubit centered-residual predictor."""

import math

import torch
from torch import nn


class QuantumResidualVQC(nn.Module):
    """128 -> bounded six angles -> two-layer VQC -> three centered residuals.

    ``pi*tanh`` keeps encoding angles in (-pi, pi). Inputs are encoded once.
    Native parameter broadcasting executes one QNode per batch, not per sample.
    This pilot runs entirely on CPU, including the default.qubit simulator.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("Quantum pilot requires PennyLane: pip install pennylane") from exc
        self.projection = nn.Linear(128, 6)
        self.weights = nn.Parameter(torch.empty(2, 6, 2).uniform_(-0.1, 0.1))
        self.readout = nn.Linear(6, 3)
        device = qml.device("default.qubit", wires=6, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(angles: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            if angles.device.type != "cpu" or weights.device.type != "cpu":
                raise ValueError("This pilot's default.qubit QNode requires CPU inputs and quantum weights.")
            for wire in range(6):
                qml.RY(angles[:, wire], wires=wire)
            for layer in range(2):
                for wire in range(6):
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                    qml.RY(weights[layer, wire, 0], wires=wire)
                for wire in range(6):
                    qml.CNOT(wires=[wire, (wire + 1) % 6])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(6)]

        self.circuit = circuit

    def quantum_features(self, latent: torch.Tensor) -> torch.Tensor:
        """Return six local expectations, preserving batch axis and autograd."""
        if latent.ndim != 2 or latent.shape[1] != 128 or not len(latent):
            raise ValueError("Quantum residual input must be nonempty [B,128].")
        if latent.device.type != "cpu" or any(p.device.type != "cpu" for p in self.parameters()):
            raise ValueError("This default.qubit pilot is CPU-only; keep the entire residual model and inputs on CPU.")
        angles = math.pi * torch.tanh(self.projection(latent))
        # PennyLane may promote simulator precision; match the classical readout.
        return torch.stack(self.circuit(angles, self.weights), dim=-1).to(latent.dtype)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.readout(self.quantum_features(latent))

    def parameter_counts(self) -> dict[str, int]:
        """Count registered trainable parameters by branch component."""
        counts = {"projection": sum(p.numel() for p in self.projection.parameters()),
                  "quantum": self.weights.numel(),
                  "readout": sum(p.numel() for p in self.readout.parameters())}
        counts["total"] = sum(counts.values())
        return counts
