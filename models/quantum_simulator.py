"""Batched historical six-qubit, two-block RVQC; wire 0 is the most significant bit."""

import torch
from torch import Tensor, nn


def pennylane_reference():
    import pennylane as qml
    device = qml.device("default.qubit", wires=6, shots=None, seed=0)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(data, weights):
        for block in range(2):
            for wire in range(6):
                qml.RY(data[:, wire], wires=wire)
                qml.RZ(weights[block, wire, 0], wires=wire)
                qml.RY(weights[block, wire, 1], wires=wire)
            for wire in range(6):
                qml.CNOT(wires=[wire, (wire + 1) % 6])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(6)]
    return circuit


class RVQCSimulator(nn.Module):
    def __init__(self):
        super().__init__()
        indices = torch.arange(64)
        bits = (indices[:, None] >> torch.arange(5, -1, -1)) & 1
        self.register_buffer("z_signs", (1 - 2 * bits).float())
        permutations = []
        for control in range(6):
            target = (control + 1) % 6
            permutations.append(indices ^ (bits[:, control] << (5 - target)))
        self.register_buffer("cnot_indices", torch.stack(permutations))

    @staticmethod
    def ry(state: Tensor, angle: Tensor, wire: int) -> Tensor:
        paired = state.reshape(len(state), 2 ** wire, 2, 2 ** (5 - wire))
        low, high = paired[:, :, 0], paired[:, :, 1]
        cosine = torch.cos(angle / 2).reshape(-1, 1, 1)
        sine = torch.sin(angle / 2).reshape(-1, 1, 1)
        return torch.stack((cosine * low - sine * high,
                            sine * low + cosine * high), dim=2).reshape(-1, 64)

    @staticmethod
    def rz(state: Tensor, angle: Tensor, wire: int) -> Tensor:
        paired = state.reshape(len(state), 2 ** wire, 2, 2 ** (5 - wire))
        phase = torch.complex(torch.cos(angle / 2), torch.sin(angle / 2))
        phase = phase.reshape(-1, 1, 1)
        return torch.stack((paired[:, :, 0] * phase.conj(),
                            paired[:, :, 1] * phase), dim=2).reshape(-1, 64)

    def forward(self, theta: Tensor, angles: Tensor) -> Tensor:
        if theta.ndim != 2 or theta.shape[1] != 6 or angles.shape != (2, 6, 2) or not len(theta):
            raise ValueError("Expected nonempty theta [B,6] and angles [2,6,2].")
        if theta.device != angles.device or theta.device != self.z_signs.device:
            raise ValueError("Inputs, angles and simulator must share a device.")
        with torch.autocast(device_type=theta.device.type, enabled=False):
            theta, angles = theta.float(), angles.float()
            state = torch.zeros(len(theta), 64, dtype=torch.complex64, device=theta.device)
            state[:, 0] = 1
            for block in range(2):
                for wire in range(6):
                    state = self.ry(state, theta[:, wire], wire)
                    state = self.rz(state, angles[block, wire, 0], wire)
                    state = self.ry(state, angles[block, wire, 1], wire)
                # Ordered CNOTs do not commute: preserve the directed ring order.
                for wire in range(6):
                    state = state.index_select(1, self.cnot_indices[wire])
            return (state.real.square() + state.imag.square()) @ self.z_signs.float()
