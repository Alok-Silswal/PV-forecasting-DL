"""Batched six-qubit Torch statevector and matching PennyLane reference."""

import torch
from torch import Tensor, nn


def pennylane_reference():
    import pennylane as qml

    device = qml.device("default.qubit", wires=6, shots=None, seed=0)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(theta, angles):
        for wire in range(6):
            qml.RY(theta[:, wire], wires=wire)
        for wire in range(6):
            qml.RZ(angles[0, wire, 0], wires=wire)
            qml.RY(angles[0, wire, 1], wires=wire)
        for wire in range(5):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(6):
            qml.RZ(angles[1, wire, 0], wires=wire)
            qml.RY(angles[1, wire, 1], wires=wire)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(6)]

    return circuit


class SixQubitSimulator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        indices = torch.arange(64)
        # Wire 0 is the most significant bit, matching PennyLane ordering.
        bits = (indices[:, None] >> torch.arange(5, -1, -1)) & 1
        parity = (bits[:, :-1] * bits[:, 1:]).sum(1) % 2
        self.register_buffer("cz_signs", (1 - 2 * parity).float())
        self.register_buffer("z_signs", (1 - 2 * bits).float())

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

    def statevector(self, theta: Tensor, angles: Tensor) -> Tensor:
        if theta.ndim != 2 or theta.shape[1] != 6 or angles.shape != (2, 6, 2):
            raise ValueError("Expected theta [B,6] and quantum angles [2,6,2].")
        if theta.device != angles.device or theta.device != self.cz_signs.device:
            raise ValueError("Quantum inputs, angles, and simulator must share a device.")
        if len(theta) == 0:
            raise ValueError("Quantum batch must be nonempty.")
        with torch.autocast(device_type=theta.device.type, enabled=False):
            theta, angles = theta.float(), angles.float()
            state = torch.zeros(len(theta), 64, dtype=torch.complex64, device=theta.device)
            state[:, 0] = 1
            for wire in range(6):
                state = self.ry(state, theta[:, wire], wire)
            for block in range(2):
                for wire in range(6):
                    state = self.rz(state, angles[block, wire, 0], wire)
                    state = self.ry(state, angles[block, wire, 1], wire)
                if block == 0:
                    # All five adjacent CZ gates are diagonal and commute.
                    state = state * self.cz_signs.float()
            return state

    def forward(self, theta: Tensor, angles: Tensor) -> Tensor:
        with torch.autocast(device_type=theta.device.type, enabled=False):
            state = self.statevector(theta, angles)
            probabilities = state.real.square() + state.imag.square()
            return probabilities @ self.z_signs.float()


class QuantumCircuit(nn.Module):
    def __init__(self, seed: int = 42, backend: str = "torch") -> None:
        super().__init__()
        if backend not in ("torch", "pennylane"):
            raise ValueError("backend must be torch or pennylane.")
        self.backend = backend
        generator = torch.Generator().manual_seed(seed + 10000)
        self.angles = nn.Parameter(torch.empty(2, 6, 2).normal_(0, 0.1, generator=generator))
        self.simulator = SixQubitSimulator()
        self.reference = pennylane_reference() if backend == "pennylane" else None

    def forward(self, theta: Tensor) -> Tensor:
        with torch.autocast(device_type=theta.device.type, enabled=False):
            if self.backend == "torch":
                return self.simulator(theta.float(), self.angles.float())
            if theta.ndim != 2 or theta.shape[1] != 6:
                raise ValueError("Expected theta [B,6].")
            return torch.stack(self.reference(theta.float(), self.angles.float()), dim=1).float()
