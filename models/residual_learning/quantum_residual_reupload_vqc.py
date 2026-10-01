"""Six-qubit residual pilot with two encodings of the same projected angles."""

import torch

from models.residual_learning.quantum_residual_vqc import QuantumResidualVQC


class QuantumResidualReuploadVQC(QuantumResidualVQC):
    """Preserve the first pilot's initialization/readout; change encoding frequency."""

    def __init__(self) -> None:
        super().__init__()
        import pennylane as qml

        device = qml.device("default.qubit", wires=6, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(angles: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            if angles.device.type != "cpu" or weights.device.type != "cpu":
                raise ValueError("This re-uploading pilot requires CPU inputs and quantum weights.")
            for layer in range(2):
                for wire in range(6):
                    qml.RY(angles[:, wire], wires=wire)
                for wire in range(6):
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                    qml.RY(weights[layer, wire, 0], wires=wire)
                for wire in range(6):
                    qml.CNOT(wires=[wire, (wire + 1) % 6])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(6)]

        self.circuit = circuit
