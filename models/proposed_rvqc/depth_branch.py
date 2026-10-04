"""Depth-only extension; the official two-block implementation stays unchanged."""

import pennylane as qml
import torch

from .residual_branch import ResidualBranch, canonical_state


def depth_state(run_seed, num_reupload_blocks=2):
    if num_reupload_blocks not in (1, 2, 3, 4):
        raise ValueError("Reupload blocks must be 1, 2, 3 or 4.")
    state = canonical_state(run_seed)
    if num_reupload_blocks != 2:
        generator = torch.Generator(device="cpu").manual_seed(run_seed + 10000)
        state["quantum_angles"] = torch.empty(num_reupload_blocks, 6, 2, dtype=torch.float32).uniform_(
            -0.1, 0.1, generator=generator)
    return state


class DepthResidualBranch(ResidualBranch):
    def __init__(self, family, state, num_reupload_blocks=2):
        if num_reupload_blocks not in (1, 2, 3, 4):
            raise ValueError("Reupload blocks must be 1, 2, 3 or 4.")
        if state["quantum_angles"].shape != (num_reupload_blocks, 6, 2):
            raise ValueError("Quantum angles do not match the requested depth.")
        super().__init__(family, state)
        self.num_reupload_blocks = num_reupload_blocks
        if num_reupload_blocks == 2:
            return
        device = qml.device("default.qubit", wires=6, shots=None, seed=0)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(data, weights):
            for block in range(num_reupload_blocks):
                for wire in range(6):
                    qml.RY(data[:, wire], wires=wire)
                    qml.RZ(weights[block, wire, 0], wires=wire)
                    qml.RY(weights[block, wire, 1], wires=wire)
                for wire in range(6):
                    qml.CNOT(wires=[wire, (wire + 1) % 6])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(6)]

        self.circuit = circuit

    def forward_cached(self, pooled, baseline_prediction):
        return baseline_prediction + self(pooled)
