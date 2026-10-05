"""Single-encoding six-qubit feature map; imported only by arms D/E."""

import torch
from torch import nn


class QuantumFeatures(nn.Module):
    def __init__(self, seed, trainable=True):
        super().__init__()
        import pennylane as qml

        generator = torch.Generator().manual_seed(seed)
        initial = torch.empty(2, 6, 2).uniform_(-0.1, 0.1, generator=generator)
        self.angles = nn.Parameter(initial, requires_grad=trainable)
        self.register_buffer("initial_angles", initial.clone())
        device = qml.device("default.qubit", wires=6, shots=None, seed=seed)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(theta, weights):
            for wire in range(6):
                qml.RY(theta[:, wire], wires=wire)
            for block in range(2):
                for wire in range(6):
                    qml.CZ(wires=[wire, (wire + 1) % 6])
                for wire in range(6):
                    qml.RZ(weights[block, wire, 0], wires=wire)
                    qml.RY(weights[block, wire, 1], wires=wire)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(6)]

        self.circuit = circuit

    def forward(self, theta):
        if theta.ndim != 2 or theta.shape[1] != 6:
            raise ValueError("Expected quantum angles [B,6].")
        return torch.stack(self.circuit(theta, self.angles), dim=1).to(dtype=theta.dtype)

    @torch.no_grad()
    def parameter_movement(self):
        return (self.angles - self.initial_angles).norm().item()
