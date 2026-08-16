"""
Variational Quantum Circuit (VQC) branch for the PHN (Parallel Hybrid
Network) extension of the PV power forecasting model.

This module implements a *compact, complementary* quantum prediction
branch that consumes the pooled Scalar Gated Fusion (SGF) representation
and produces a small quantum prediction. It is the ONLY quantum
component in the project.

Data flow (see models/proposed_model.py for how this is wired in):

    SGF output (B, 24, 128)
            │
            ▼
      mean(dim=1)              <- performed by this module's caller
            │
            ▼
         (B, 128)
            │
            ▼
      Linear(128, 2)           <- trainable classical->quantum interface
            │
            ▼
      bounded angle encoding   <- tanh * pi, for numerical stability
            │
            ▼
      2-qubit VQC, depth = 2   <- RY + RZ variational layers, CNOT entangler
            │
            ▼
   expectation values (<=3)    <- <Z0>, <Z1>, <Z0 Z1>
            │
            ▼
         y_q (B, output_dim)

Configuration
-------------
* Simulator (``simulator``, default ``"default.qubit"`` via
  ``configs/config.py``'s ``VQC_SIMULATOR``): a noiseless, analytic
  (``shots=None``) PennyLane statevector simulator. ``"lightning.qubit"``
  is also supported for comparison/fallback.
* Differentiation (``diff_method``, default ``"backprop"`` via
  ``VQC_DIFF_METHOD``): reverse-mode automatic differentiation through
  the simulated statevector evolution. ``"adjoint"`` is also supported.
  ``"backprop"`` requires ``simulator="default.qubit"``.
* Both are noiseless simulators with no hardware assumptions; there is
  no quantum hardware involved anywhere in this project.

Output dimension
-----------------
The 15-minute horizon requires 3 expectation values (<Z0>, <Z1>,
<Z0 Z1>) with NO classical layer after the VQC — the circuit itself
must produce the forecast. A 2-qubit circuit restricted to Pauli-Z-based
observables can supply at most 3 independent expectation values, so the
60-minute horizon (12 outputs) is not currently supported by this
design; see the ``NotImplementedError`` raised in ``VQCBranch.__init__``
for the specific conflict and why it is not silently resolved here.
"""

from typing import List

import pennylane as qml
import torch
import torch.nn as nn
from torch import Tensor

# Pauli-Z-based observables available from a 2-qubit circuit, in the
# order the spec prefers them. Only the first `output_dim` are used.
_MAX_SUPPORTED_OUTPUT_DIM = 3

# Simulators this module has been validated against. Both are
# noiseless, analytic statevector simulators; no hardware-specific
# backend is supported.
_SUPPORTED_SIMULATORS = ("default.qubit", "lightning.qubit")


def _build_qnode(
    num_qubits: int,
    depth: int,
    output_dim: int,
    diff_method: str,
    simulator: str,
):
    """
    Construct the PennyLane QNode implementing the VQC.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (fixed at 2 for the initial implementation).
    depth : int
        Number of variational (RY + RZ + CNOT) layers (fixed at 2).
    output_dim : int
        Number of Pauli-Z-based expectation values to return. Must be
        <= 3 for a 2-qubit circuit under this design (see module
        docstring).
    diff_method : str
        PennyLane differentiation method, e.g. "backprop" or "adjoint".
    simulator : str
        PennyLane device name, e.g. "default.qubit". Passed straight
        through from ``VQCBranch`` (which already validates it) so the
        device name exists in exactly one place at call time, rather
        than being independently re-hardcoded here.

    Returns
    -------
    qml.QNode
        A QNode bound to the requested simulator device, taking
        ``(inputs, weights)`` and returning ``output_dim`` expectation
        values. ``inputs`` has shape ``(B, num_qubits)`` and ``weights``
        has shape ``(depth, num_qubits, 2)`` (RY, RZ angles per qubit
        per layer).
    """

    device = qml.device(simulator, wires=num_qubits)

    observables = [
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
    ][:output_dim]

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs: Tensor, weights: Tensor) -> List[Tensor]:
        # ---- Angle encoding (one rotation per qubit) ----
        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")

        # ---- Variational layers: RY + RZ per qubit, CNOT entanglement ----
        #
        # For exactly 2 qubits, a "ring of CNOTs" (CNOT(q, (q+1) % n) for
        # every qubit q) degenerates into TWO distinct gates -- CNOT(0,1)
        # and CNOT(1,0) -- which are not equivalent and not redundant.
        # That would silently double the entangling-gate count per layer
        # versus this design's single-CNOT-per-layer depth-2 circuit.
        # Entanglement here is therefore a single CNOT(0, 1) per layer.
        for layer in range(depth):
            for qubit in range(num_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)

            qml.CNOT(wires=[0, 1])

        return [qml.expval(observable) for observable in observables]

    return circuit


class VQCBranch(nn.Module):
    """
    Compact 2-qubit, depth-2 variational quantum prediction branch.

    Consumes the pooled (temporal-mean) SGF representation and produces
    a small quantum forecast, entirely independent of and parallel to
    the existing classical ``MLPHead`` branch.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the pooled SGF representation fed into this
        branch (``bilstm_hidden_size * 2`` in the existing architecture,
        i.e. 128 by default).
    output_dim : int
        Number of forecast steps this branch must produce. Must be
        <= 3 for the current 2-qubit design (see module docstring).
    num_qubits : int, default 2
        Number of qubits in the VQC. Fixed at 2 per the research spec;
        exposed as a constructor argument only for configurability, not
        as an invitation to scale it up without cause.
    depth : int, default 2
        Number of variational layers. Fixed at 2 per the research spec.
    simulator : str, default "default.qubit"
        PennyLane device name. See module docstring for supported
        values. Must remain a noiseless, analytic statevector simulator;
        no hardware-specific backend is supported.
    diff_method : str, default "backprop"
        PennyLane differentiation method used for the QNode. See module
        docstring. "backprop" requires simulator="default.qubit".

    Raises
    ------
    ValueError
        If ``output_dim`` exceeds what a 2-qubit, Pauli-Z-only
        measurement scheme can support (3), if ``num_qubits`` is not 2
        (the only qubit count validated by this design; see module
        docstring regarding the 60-minute horizon conflict), if
        ``simulator`` is not one of the supported simulators, or if
        ``diff_method="backprop"`` is requested with a simulator other
        than ``"default.qubit"``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_qubits: int = 2,
        depth: int = 2,
        simulator: str = "default.qubit",
        diff_method: str = "backprop",
    ) -> None:
        super().__init__()

        if num_qubits != 2:
            raise ValueError(
                "VQCBranch is validated only for num_qubits=2, per the "
                "research spec's explicit qubit-count restriction. "
                f"Got num_qubits={num_qubits}."
            )

        if output_dim > _MAX_SUPPORTED_OUTPUT_DIM:
            raise NotImplementedError(
                f"VQCBranch was asked for output_dim={output_dim}, but a "
                f"{num_qubits}-qubit circuit restricted to Pauli-Z-based "
                f"observables ({{<Z0>, <Z1>, <Z0 Z1>}}) can supply at most "
                f"{_MAX_SUPPORTED_OUTPUT_DIM} expectation values without "
                "increasing qubit count or circuit depth. This is a "
                "genuine architectural conflict for the 60-minute "
                "(12-output) horizon under the current spec, which "
                "forbids both (a) a classical Linear layer after the VQC "
                "and (b) scaling up the circuit without an explicit "
                "experimental reason. Resolving this requires an "
                "explicit research decision (e.g. additional qubits/"
                "observables, or a different measurement scheme) and is "
                "intentionally NOT silently resolved here."
            )

        if simulator not in _SUPPORTED_SIMULATORS:
            raise ValueError(
                f"Unsupported simulator '{simulator}'. Supported: "
                f"{_SUPPORTED_SIMULATORS} (both noiseless, analytic "
                "statevector simulators; no hardware-specific backend "
                "is supported)."
            )

        if diff_method == "backprop" and simulator != "default.qubit":
            raise ValueError(
                "diff_method='backprop' requires simulator='default.qubit' "
                f"(backprop is not available on '{simulator}'). Use "
                "diff_method='adjoint' if 'lightning.qubit' is required, "
                "or switch simulator to 'default.qubit'."
            )

        self.num_qubits = num_qubits
        self.depth = depth
        self.output_dim = output_dim
        self.simulator = simulator
        self.diff_method = diff_method

        # ------------------------------------------------------------
        # Classical -> quantum interface: Linear(input_dim, num_qubits).
        # This is a plain trainable linear projection, not a hidden
        # network — exactly as specified.
        # ------------------------------------------------------------
        self.projection = nn.Linear(
            in_features=input_dim,
            out_features=num_qubits,
        )
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0.0)

        # ------------------------------------------------------------
        # Variational parameters: (depth, num_qubits, 2) for RY, RZ.
        # Small random initialization is standard for VQCs to avoid
        # barren-plateau-like flat starting regions.
        # ------------------------------------------------------------
        self.weights = nn.Parameter(
            0.01 * torch.randn(depth, num_qubits, 2)
        )

        self._circuit = _build_qnode(
            num_qubits=num_qubits,
            depth=depth,
            output_dim=output_dim,
            diff_method=diff_method,
            simulator=simulator,
        )

    def forward(self, pooled_features: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        pooled_features : Tensor
            Temporal-mean-pooled SGF representation, shape
            (batch_size, input_dim). Pooling itself is performed by the
            caller (``ProposedModel``), NOT by this module — this
            branch owns only the projection and the VQC.

        Returns
        -------
        Tensor
            Quantum prediction, shape (batch_size, output_dim).
        """

        if pooled_features.dim() != 2:
            raise ValueError(
                "VQCBranch expects a pooled 2D (batch_size, input_dim) "
                f"tensor; got a {pooled_features.dim()}D tensor of shape "
                f"{tuple(pooled_features.shape)}. Temporal pooling must "
                "be performed by the caller before calling VQCBranch."
            )

        # Bounded transformation before angle encoding, for numerical
        # stability (rotation angles are periodic in 2*pi, but an
        # unbounded projection could otherwise wrap many times over
        # and destabilize gradients early in training). This is a
        # simple, documented interface choice — not quantum
        # preprocessing.
        raw_angles = self.projection(pooled_features)
        angles = torch.tanh(raw_angles) * torch.pi

        expectation_values = self._circuit(angles, self.weights)

        # QNode returns a list of `output_dim` tensors, each of shape
        # (batch_size,); stack into (batch_size, output_dim).
        quantum_prediction = torch.stack(expectation_values, dim=-1)

        return quantum_prediction