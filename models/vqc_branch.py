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

Design notes
------------
* Simulator: ``lightning.qubit`` (noiseless, no hardware assumptions).
* Differentiation: ``adjoint`` (exact, efficient for statevector sims;
  compatible with PyTorch autograd via PennyLane's torch interface).
* Batching (verified against the ``pennylane-lightning`` device source,
  not assumed): ``lightning.qubit`` does **not** intrinsically support
  parameter broadcasting. Its own ``preprocess_transforms`` documents
  this explicitly ("Currently does not intrinsically support parameter
  broadcasting"), and its device transform pipeline includes
  ``qml.transforms.broadcast_expand``, which splits a broadcasted tape
  into ``B`` separate non-broadcasted tapes *before* execution. The
  device then executes those ``B`` tapes in an internal loop (one
  ``simulate()`` call, and one adjoint-Jacobian pass, per tape).

  Passing ``(B, 2)`` inputs directly to this QNode is still the right
  choice — it is accepted at the QNode interface with no shape errors,
  gradients are correct, and it avoids a *Python*-level per-sample loop
  with its interpreter overhead — but it is NOT a single vectorized
  circuit evaluation the way a batched matmul would be. Under the hood,
  PennyLane dispatches ``B`` sequential statevector simulations via its
  C++/Python glue rather than one broadcasted circuit call. No
  unverified performance claim is made beyond that; actual per-batch
  runtime should be measured empirically rather than assumed (see the
  accompanying pilot instructions).

  ``default.qubit`` does support native parameter broadcasting, but the
  research spec requires ``lightning.qubit``, so that swap is not made
  here. If a future PennyLane/Lightning release adds native broadcasting
  to ``lightning.qubit``, no code change is needed in this module — only
  the runtime characteristics of the existing call would improve.
  ``diff_method`` remains a constructor argument so a fallback to
  ``"parameter-shift"`` requires no other code changes.

* Output dimension: item 8 of the spec requires 3 expectation values
  for the 15-minute horizon (<Z0>, <Z1>, <Z0 Z1>) with NO classical
  Linear(2, 3) layer after the VQC — the quantum circuit itself must
  produce the forecast. For the 60-minute horizon (12 outputs), a
  2-qubit circuit has at most 3 independent single/two-qubit Pauli-Z
  expectation values available from {Z0, Z1, Z0Z1} (a 2-qubit Hilbert
  space admits more observables in principle, e.g. X/Y bases or higher
  moments, but the spec explicitly restricts this branch to "Pauli-Z
  based expectation measurements" and forbids increasing qubit count/
  depth "without an explicit experimental reason"). This is a genuine
  architectural conflict for the 60-minute horizon and is NOT silently
  resolved here — see the ``NotImplementedError`` raised below and the
  discussion in the change summary / PR description.
"""

from typing import List

import pennylane as qml
import torch
import torch.nn as nn
from torch import Tensor

# Pauli-Z-based observables available from a 2-qubit circuit, in the
# order the spec prefers them. Only the first `output_dim` are used.
_MAX_SUPPORTED_OUTPUT_DIM = 3


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
        PennyLane differentiation method, e.g. "adjoint" or
        "parameter-shift".
    simulator : str
        PennyLane device name, e.g. "lightning.qubit". Passed straight
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
        # NOTE (corrected after review): a "ring of CNOTs" implemented as
        # CNOT(q, (q+1) % num_qubits) for every qubit q is standard for
        # num_qubits > 2, but degenerates on exactly 2 qubits into TWO
        # distinct gates — CNOT(0, 1) and CNOT(1, 0) — which are NOT
        # equivalent (control and target are swapped) and NOT redundant.
        # That would silently double the entangling-gate count per layer
        # versus the depth-2, single-CNOT-per-layer design this module
        # documents and the spec's benchmark circuit implies. For the
        # fixed 2-qubit case used here, entanglement is therefore a
        # single CNOT(0, 1) per layer.
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
    simulator : str, default "lightning.qubit"
        PennyLane device name. Must remain a noiseless statevector
        simulator; no hardware-specific backend is supported.
    diff_method : str, default "adjoint"
        PennyLane differentiation method used for the QNode.

    Raises
    ------
    ValueError
        If ``output_dim`` exceeds what a 2-qubit, Pauli-Z-only
        measurement scheme can support (3), or if ``num_qubits`` is not
        2 (the only qubit count validated by this design; see module
        docstring regarding the 60-minute horizon conflict).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_qubits: int = 2,
        depth: int = 2,
        simulator: str = "lightning.qubit",
        diff_method: str = "adjoint",
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

        if simulator != "lightning.qubit":
            raise ValueError(
                "Only the 'lightning.qubit' simulator is supported in "
                f"this initial implementation. Got '{simulator}'."
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