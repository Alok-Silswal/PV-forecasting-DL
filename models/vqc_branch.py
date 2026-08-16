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
* Simulator: configurable via ``simulator`` (default ``"default.qubit"``,
  see ``configs/config.py``'s ``VQC_SIMULATOR``). Both ``"default.qubit"``
  and ``"lightning.qubit"`` are supported -- both are noiseless,
  analytic (``shots=None``) statevector simulators with no hardware
  assumptions; only their internal execution strategy for a batched
  input differs (see below).
* Differentiation: configurable via ``diff_method`` (default
  ``"backprop"``, see ``VQC_DIFF_METHOD``). ``"adjoint"`` remains
  supported as a constructor argument for both simulators.

* Batching (verified empirically, not assumed -- see the accompanying
  simulator-comparison and gradient-equivalence validation scripts):
  ``lightning.qubit`` does **not** intrinsically support parameter
  broadcasting. Its own ``preprocess_transforms`` documents this
  explicitly ("Currently does not intrinsically support parameter
  broadcasting"), and its device transform pipeline includes
  ``qml.transforms.broadcast_expand``, which splits a broadcasted tape
  into ``B`` separate non-broadcasted tapes *before* execution -- i.e.
  ``B`` sequential statevector simulations per forward call, each with
  its own adjoint-Jacobian pass under ``diff_method="adjoint"``.

  ``default.qubit`` genuinely executes a broadcasted tape as a SINGLE
  call for our exact observables (plain ``qml.expval(PauliZ(...))`` /
  ``qml.expval(PauliZ(...) @ PauliZ(...))`` measurements -- not shadow
  measurements, which are the only case ``default.qubit`` falls back to
  per-sample tape splitting for). Under ``diff_method="backprop"``,
  gradients are computed via ordinary reverse-mode automatic
  differentiation through the simulated statevector evolution, which
  PyTorch can trace and differentiate as a single batched computation
  graph rather than as ``B`` independent graphs.

  This was measured directly (device-execution instrumentation
  confirming exactly 1 tape dispatched per forward call for
  ``default.qubit``, versus ``B`` for ``lightning.qubit``) and produced
  a ~44x forward+backward speedup for batch_size=256 on the target
  Kaggle CPU environment, with outputs and gradients (both w.r.t. VQC
  weights and w.r.t. the VQC input tensor) agreeing with the prior
  ``lightning.qubit + adjoint`` configuration to within 1e-5 absolute
  tolerance (observed max differences: output 2.57e-07, weight
  gradient 7.45e-09, input gradient 1.16e-09) -- i.e. numerically
  equivalent to floating-point precision, not merely "close enough to
  train." This is a simulator/differentiation-method substitution, not
  a change to the VQC's mathematical definition: the circuit itself
  (qubits, depth, gates, angle encoding, observables, output dimension)
  is identical regardless of which simulator/diff_method executes it.

  ``lightning.qubit`` (with ``diff_method="adjoint"`` or
  ``"parameter-shift"``) remains fully supported by this module via the
  ``simulator``/``diff_method`` constructor arguments and
  ``configs/config.py``'s ``VQC_SIMULATOR``/``VQC_DIFF_METHOD``, for
  cases where reverting to it is useful (e.g. comparison experiments).

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
    batch_obs: bool = False,
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
    batch_obs : bool, default False
        Forwarded to ``qml.device(..., batch_obs=batch_obs)``. Only
        meaningful for ``lightning.qubit``/``lightning.gpu``, where it
        parallelizes the per-observable adjoint-differentiation passes
        of a SINGLE tape across OpenMP threads (see
        ``OMP_NUM_THREADS``). It does NOT change the circuit, the
        differentiation method, the observables computed, or the
        numerical result — only how the existing 3-observable adjoint
        backward pass is scheduled across CPU threads. Left at its
        default of False unless explicitly enabled, so existing
        behavior is unaffected unless opted into. Has no effect on
        ``default.qubit`` (which does not accept or use this keyword
        the same way); already tested and rejected as an optimization
        for ``lightning.qubit`` and retained here only for
        configurability/backward compatibility, not as an active
        recommendation.

    Returns
    -------
    qml.QNode
        A QNode bound to the requested simulator device, taking
        ``(inputs, weights)`` and returning ``output_dim`` expectation
        values. ``inputs`` has shape ``(B, num_qubits)`` and ``weights``
        has shape ``(depth, num_qubits, 2)`` (RY, RZ angles per qubit
        per layer).
    """

    # batch_obs is a lightning.qubit/lightning.gpu-specific device
    # keyword (OpenMP-thread parallelism across observables within one
    # tape's adjoint backward pass). default.qubit does not use this
    # keyword the same way, so it is only forwarded for lightning-family
    # devices, keeping default.qubit's device construction exactly as
    # plain as lightning.qubit's was before batch_obs was introduced.
    if simulator.startswith("lightning."):
        device = qml.device(simulator, wires=num_qubits, batch_obs=batch_obs)
    else:
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
    simulator : str, default "default.qubit"
        PennyLane device name. Supported: "default.qubit" (default,
        genuinely batches our exact observable set as a single tape
        execution -- see module docstring) and "lightning.qubit"
        (executes a batch as B sequential single-sample tapes; retained
        for comparison/fallback). Must remain a noiseless, analytic
        statevector simulator; no hardware-specific backend is
        supported.
    diff_method : str, default "backprop"
        PennyLane differentiation method used for the QNode.
        "backprop" (default, requires simulator="default.qubit") and
        "adjoint" (available on both supported simulators) are both
        mathematically exact for this noiseless circuit and were
        verified to agree numerically to within 1e-5 absolute
        tolerance for both outputs and gradients (see module
        docstring).
    batch_obs : bool, default False
        Whether to parallelize the per-observable adjoint-Jacobian
        passes across OpenMP threads (see ``_build_qnode`` docstring
        for exactly what this does and does not affect). Purely a
        CPU-scheduling optimization for ``lightning.qubit``'s existing
        3-observable adjoint backward pass; does not alter the circuit,
        qubit count, depth, gates, observables, output values, or
        differentiation method. Effective thread count is governed by
        the ``OMP_NUM_THREADS`` environment variable, which must be set
        before PennyLane is imported (see the accompanying profiling
        script for a runtime-detected, non-arbitrary value).

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
        simulator: str = "default.qubit",
        diff_method: str = "backprop",
        batch_obs: bool = False,
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

        _SUPPORTED_SIMULATORS = ("default.qubit", "lightning.qubit")
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
        self.batch_obs = batch_obs

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
            batch_obs=batch_obs,
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