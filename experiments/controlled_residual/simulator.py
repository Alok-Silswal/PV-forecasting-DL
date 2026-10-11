"""Exact RVQC circuit using batched QTM rotation primitives."""

import torch
from torch import Tensor, nn
from models.proposed_qtm.quantum_simulator import SixQubitSimulator


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
                    state = SixQubitSimulator.ry(state, theta[:, wire], wire)
                    state = SixQubitSimulator.rz(state, angles[block, wire, 0], wire)
                    state = SixQubitSimulator.ry(state, angles[block, wire, 1], wire)
                # Ordered CNOTs do not commute: preserve the directed ring order.
                for wire in range(6):
                    state = state.index_select(1, self.cnot_indices[wire])
            return (state.real.square() + state.imag.square()) @ self.z_signs.float()
