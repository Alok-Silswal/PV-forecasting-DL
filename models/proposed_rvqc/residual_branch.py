"""Identically initialized six-qubit trainable and frozen controls."""

import math

import pennylane as qml
import torch
from torch import nn

from .frozen_baseline import FrozenBaseline

FAMILIES = ("proposed_rvqc", "proposed_rvqc_frozen")


def canonical_state(run_seed):
    common = torch.Generator(device="cpu").manual_seed(run_seed)
    quantum = torch.Generator(device="cpu").manual_seed(run_seed + 10000)
    state = {}
    for name, inputs, outputs in [("projection", 128, 6), ("readout", 6, 3)]:
        bound = 1 / math.sqrt(inputs)
        state[name + ".weight"] = torch.empty(outputs, inputs, dtype=torch.float32).uniform_(-bound, bound, generator=common)
        state[name + ".bias"] = torch.empty(outputs, dtype=torch.float32).uniform_(-bound, bound, generator=common)
    state["quantum_angles"] = torch.empty(2, 6, 2, dtype=torch.float32).uniform_(-0.1, 0.1, generator=quantum)
    return state


class ResidualBranch(nn.Module):
    def __init__(self, family, state):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"Unknown residual family: {family}")
        self.family = family
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(128, 6, dtype=torch.float32)
            self.readout = nn.Linear(6, 3, dtype=torch.float32)
        self.quantum_angles = nn.Parameter(
            state["quantum_angles"].clone(), requires_grad=family == "proposed_rvqc"
        )
        self.load_state_dict(state, strict=True)
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

        self.circuit = circuit

    def forward(self, pooled):
        if pooled.ndim != 2 or pooled.shape[1] != 128 or pooled.device.type != "cpu":
            raise ValueError("Residual simulator expects CPU pooled features [B,128].")
        angles = math.pi * torch.tanh(self.projection(pooled))
        measured = torch.stack(self.circuit(angles, self.quantum_angles), dim=1)
        return self.readout(measured.to(dtype=pooled.dtype))


class FrozenResidualModel(nn.Module):
    def __init__(self, run, family, state=None):
        super().__init__()
        self.frozen = FrozenBaseline(run)
        self.residual = ResidualBranch(family, canonical_state(41 + run) if state is None else state)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward_cached(self, pooled, baseline_prediction):
        return baseline_prediction + self.residual(pooled)

    def forward(self, inputs):
        pooled, prediction = self.frozen(inputs)
        return self.forward_cached(pooled, prediction)
