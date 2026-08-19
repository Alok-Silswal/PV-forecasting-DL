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
         (B, 128)              <- may be on CUDA (classical backbone device)
            │
            ▼
      Linear(128, num_qubits)  <- trainable classical->quantum interface
            │
            ▼
      bounded angle encoding   <- tanh * pi, for numerical stability
            │
            ▼
      [ GPU -> CPU transfer ]  <- see "Device handling" below
            │
            ▼
    num_qubits-qubit VQC,      <- RY + RZ variational layers,
    depth = 2                     nearest-neighbour CNOT entangler
            │  (executes entirely on CPU; default.qubit has no CUDA path)
            ▼
   expectation values (=3)    <- <Z0>, <Z1>, <Z0 Z1> (always wires 0,1;
            │                     see "Qubit-count generalization" below)
            ▼
      [ CPU -> GPU transfer ]  <- back to the caller's original device
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

Qubit-count generalization (2 <= num_qubits <= 10)
----------------------------------------------------
This branch was originally validated only for a fixed 2-qubit circuit.
It has been generalized to support ``2 <= num_qubits <= 10`` as a
controlled experimental variable, while leaving every other design
choice (depth, gate family, encoding, simulator, differentiation
method, output semantics) untouched. Two things change with
``num_qubits``:

1. **Entanglement topology.** The original circuit applied a single
   ``CNOT(0, 1)`` per variational layer. For ``num_qubits == 2`` this
   is preserved EXACTLY, byte-for-byte, as before. For
   ``num_qubits > 2``, this generalizes to the smallest natural
   extension of "one entangling pass across the register per layer":
   a nearest-neighbour chain, ``CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,
   n-1)``, applied once per layer after the RY/RZ rotations, in qubit
   order. This keeps circuit depth unchanged (still one entangling
   sub-layer per variational layer) and reduces exactly to the
   original single-CNOT circuit at ``num_qubits=2``.

2. **Nothing else.** The observables (below), the RY/RZ rotation
   structure, the angle encoding, the projection layer, and the
   device-pinning logic are all unchanged and already were
   parameterized by ``num_qubits`` (projection width, weight tensor
   shape, device wire count) or independent of it (observables).

Device handling
----------------
``default.qubit`` (via PennyLane's Torch interface) constructs and
propagates its statevector on CPU; it has no CUDA execution path. The
rest of this project's classical backbone (DCNN/BiLSTM/SGF/MLPHead)
is expected to run on GPU during real training, so ``pooled_features``
arriving at ``VQCBranch.forward()`` may be a CUDA tensor.

Two categories of device handling exist here, for two different
reasons:

1. ``angles`` (the QNode's per-batch *input*, produced fresh every
   forward call from ``pooled_features``) is moved to CPU
   (``angles.to("cpu")``) right before the circuit call, and the
   QNode's output is moved back to ``pooled_features``'s original
   device (``.to(original_device)``) right after. This transfer is
   unavoidable and happens every forward call, since ``angles``
   genuinely depends on GPU-resident classical activations that
   change every batch.

2. ``self.weights`` (the QNode's *variational parameters* -- a
   trainable ``nn.Parameter``, persistent across calls and owned by
   the optimizer) is instead kept permanently pinned to CPU via a
   ``_apply`` override (see ``VQCBranch._apply`` below), rather than
   being moved inside ``forward()`` on every call. ``nn.Module.to()``,
   ``.cuda()``, and ``.cpu()`` all route through ``_apply`` internally,
   so overriding it lets every other submodule (e.g.
   ``self.projection``) move normally with the rest of the model while
   ``self.weights`` alone stays exempt. This means ``forward()`` never
   needs to call ``.to()`` on ``self.weights`` at all -- it is simply
   already CPU-resident by construction, avoiding a redundant
   GPU->CPU copy of the same tensor on every single forward call, and
   avoiding the (correct, but easy-to-regress) invariant that "the
   parameters happen to be on CPU because .to('cpu') was remembered
   in forward()." The optimizer's reference to ``self.weights`` is
   captured once, by object identity, when the optimizer is
   constructed (typically ``torch.optim.Adam(model.parameters())``);
   ``_apply`` updates ``self.weights.data`` in place rather than
   replacing the ``nn.Parameter`` object, so that reference -- and
   therefore correct optimization -- remains valid regardless of how
   many times the parent model is moved between devices.

``.to(device)`` is a differentiable, autograd-tracked operation (not
``.detach()`` and not a NumPy round-trip), so gradients flow correctly
backward through the ``angles``/output transfer: quantum prediction ->
VQC -> VQC input (angles, on CPU) -> across the CPU/GPU boundary ->
projection layer (on GPU) -> pooled SGF features -> classical backbone,
all on their originally intended devices. ``self.weights`` itself never
needs to cross a device boundary during forward/backward at all, since
it is CPU-resident throughout: the QNode reads it directly, and
gradients accumulate directly into ``self.weights.grad`` on CPU, right
where the optimizer expects to find them.

Output dimension
-----------------
The 15-minute horizon requires 3 expectation values (<Z0>, <Z1>,
<Z0 Z1>) with NO classical layer after the VQC — the circuit itself
must produce the forecast. These three observables are always measured
on wires 0 and 1 only, regardless of ``num_qubits``, so that the
first-three-output semantics are IDENTICAL across every supported
qubit count and reproduce the original 2-qubit circuit's output
exactly when ``num_qubits=2`` (see "Qubit-count generalization"
above). Additional qubits beyond wire 1 participate in the variational
rotations and entanglement, expanding the circuit's representational
capacity, but are not directly measured; this keeps ``output_dim``
decoupled from ``num_qubits``, as required (the 15-minute horizon
always needs exactly 3 outputs, independent of quantum capacity).
A 2-qubit circuit restricted to Pauli-Z-based observables on wires 0/1
can supply at most 3 independent expectation values from those two
wires, so the 60-minute horizon (12 outputs) is not currently
supported by this design; see the ``NotImplementedError`` raised in
``VQCBranch.__init__`` for the specific conflict and why it is not
silently resolved here.
"""

from typing import List

import pennylane as qml
import torch
import torch.nn as nn
from torch import Tensor

# Pauli-Z-based observables available from this circuit, in the order
# the spec prefers them. Only the first `output_dim` are used. These
# are ALWAYS defined on wires 0 and 1 only, regardless of `num_qubits`,
# so that output semantics stay identical across every supported qubit
# count (see "Output dimension" in the module docstring).
_MAX_SUPPORTED_OUTPUT_DIM = 3

# Simulators this module has been validated against. Both are
# noiseless, analytic statevector simulators; no hardware-specific
# backend is supported.
_SUPPORTED_SIMULATORS = ("default.qubit", "lightning.qubit")

# Supported qubit-count range for the PHN qubit-scaling experiment.
# num_qubits=2 remains the validated, default configuration; 3-10 are
# the experimental range being benchmarked (see smoke_test_qubits.py).
_MIN_SUPPORTED_QUBITS = 2
_MAX_SUPPORTED_QUBITS = 10


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
        Number of qubits, ``2 <= num_qubits <= 10``.
    depth : int
        Number of variational (RY + RZ + entangler) layers (fixed at 2).
    output_dim : int
        Number of Pauli-Z-based expectation values to return. Must be
        <= 3 (see module docstring); these are always measured on
        wires 0 and 1, independent of ``num_qubits``.
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
        per layer). Both ``inputs`` and ``weights`` must be CPU tensors
        at call time — this simulator has no CUDA execution path; see
        the "Device handling" section of the module docstring for where
        that is enforced.
    """

    device = qml.device(simulator, wires=num_qubits)

    # Observables are always defined on wires 0/1 only, regardless of
    # num_qubits, so that output semantics are identical across every
    # supported qubit count and exactly reproduce the original 2-qubit
    # circuit's outputs when num_qubits=2. See module docstring,
    # "Output dimension".
    observables = [
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
    ][:output_dim]

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs: Tensor, weights: Tensor) -> List[Tensor]:
        # ---- Angle encoding (one rotation per qubit) ----
        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")

        # ---- Variational layers: RY + RZ per qubit, entangler ----
        #
        # Entanglement topology:
        #
        # For num_qubits == 2, this is a single CNOT(0, 1) per layer,
        # IDENTICAL to the original fixed 2-qubit circuit (a ring of
        # CNOTs over 2 qubits would degenerate into the two distinct,
        # non-redundant gates CNOT(0,1) and CNOT(1,0), which is why the
        # original circuit used a single CNOT rather than a ring; that
        # reasoning is preserved here by using a nearest-neighbour
        # CHAIN, not a ring, for num_qubits > 2 as well).
        #
        # For num_qubits > 2, this extends to the smallest natural
        # generalization of "one entangling pass across the register
        # per layer": a nearest-neighbour chain,
        #     CNOT(0,1), CNOT(1,2), ..., CNOT(n-2, n-1),
        # applied once per layer, in qubit order, after the RY/RZ
        # rotations. Circuit depth (number of variational layers) is
        # unchanged; only the number of entangling gates within each
        # layer's entangling sub-step grows with num_qubits, which is
        # unavoidable for any topology that connects a larger register.
        for layer in range(depth):
            for qubit in range(num_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)

            for qubit in range(num_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return [qml.expval(observable) for observable in observables]

    return circuit


class VQCBranch(nn.Module):
    """
    Compact, depth-2 variational quantum prediction branch.

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
        <= 3 (see module docstring), independent of ``num_qubits``.
    num_qubits : int, default 2
        Number of qubits in the VQC, ``2 <= num_qubits <= 10``. The
        validated default remains 2; 3-10 are supported as an explicit
        experimental range for studying quantum-capacity scaling (see
        module docstring, "Qubit-count generalization"). Exposed as a
        constructor argument for exactly this controlled experiment,
        not as a general invitation to scale it further.
    depth : int, default 2
        Number of variational layers. Fixed at 2 per the research spec.
    simulator : str, default "default.qubit"
        PennyLane device name. See module docstring for supported
        values. Must remain a noiseless, analytic statevector simulator;
        no hardware-specific backend is supported. Note: this simulator
        executes on CPU regardless of what device the rest of the model
        is on — see the module-level "Device handling" section.
    diff_method : str, default "backprop"
        PennyLane differentiation method used for the QNode. See module
        docstring. "backprop" requires simulator="default.qubit".

    Raises
    ------
    ValueError
        If ``output_dim`` exceeds what the Pauli-Z-only measurement
        scheme on wires 0/1 can support (3), if ``num_qubits`` is
        outside ``[2, 10]``, if ``simulator`` is not one of the
        supported simulators, or if ``diff_method="backprop"`` is
        requested with a simulator other than ``"default.qubit"``.
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

        if not (_MIN_SUPPORTED_QUBITS <= num_qubits <= _MAX_SUPPORTED_QUBITS):
            raise ValueError(
                f"VQCBranch supports {_MIN_SUPPORTED_QUBITS} <= "
                f"num_qubits <= {_MAX_SUPPORTED_QUBITS} (the validated "
                f"default is num_qubits=2; 3-10 are the explicit "
                f"experimental range for the quantum-capacity-scaling "
                f"study — see module docstring). Got "
                f"num_qubits={num_qubits}."
            )

        if output_dim > _MAX_SUPPORTED_OUTPUT_DIM:
            raise NotImplementedError(
                f"VQCBranch was asked for output_dim={output_dim}, but "
                f"the Pauli-Z-based observable scheme used here "
                f"({{<Z0>, <Z1>, <Z0 Z1>}}, always measured on wires 0/1 "
                f"regardless of num_qubits) can supply at most "
                f"{_MAX_SUPPORTED_OUTPUT_DIM} expectation values without "
                "changing the measurement scheme or circuit depth. This "
                "is a genuine architectural conflict for the 60-minute "
                "(12-output) horizon under the current spec, which "
                "forbids both (a) a classical Linear layer after the VQC "
                "and (b) increasing output count merely because "
                "num_qubits increased. Resolving this requires an "
                "explicit research decision (e.g. a different "
                "measurement scheme) and is intentionally NOT silently "
                "resolved here."
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
        # network — exactly as specified. This layer is allowed to
        # move to CUDA along with the rest of the classical model; the
        # CPU boundary is enforced later, only around the QNode call.
        # Already parameterized by num_qubits; unchanged.
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
        # barren-plateau-like flat starting regions. Already
        # parameterized by num_qubits; unchanged.
        #
        # NOTE on device: this nn.Parameter is registered on the module
        # in the normal way (so it appears in .parameters()/state_dict
        # as usual and is optimized as usual), but it is PERMANENTLY
        # pinned to CPU via the _apply() override below, regardless of
        # what device the rest of this module or its parent model is
        # moved to. default.qubit has no CUDA execution path, so the
        # QNode must always read this tensor directly off CPU; forward()
        # therefore never needs to call `.to()` on self.weights at all.
        # See _apply() and the module docstring's "Device handling"
        # section for the full reasoning.
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

    def _apply(self, fn, recurse: bool = True):
        """
        Override of ``nn.Module._apply``, the internal hook that
        ``nn.Module.to()``, ``.cuda()``, ``.cpu()``, ``.half()``, etc.
        all call under the hood to transform every parameter/buffer in
        a module (and its submodules, recursively).

        Purpose: when the parent model (``ProposedModel``) is moved to
        CUDA via ``model.to("cuda")``, every submodule -- including
        this one -- has that move applied to it by default. That is
        exactly what should happen to ``self.projection`` (a normal
        classical layer). It is NOT what should happen to
        ``self.weights``: the VQC's variational parameters must stay
        on CPU permanently, because ``default.qubit`` has no CUDA
        execution path (see module + class docstrings).

        This override lets the normal ``_apply`` machinery run first
        (so ``self.projection`` and anything else in this module moves
        exactly as it would without this override), then immediately
        restores ``self.weights`` to CPU afterward.

        Critically, this updates ``self.weights.data`` (and
        ``self.weights.grad``, if present) IN PLACE rather than
        rebinding ``self.weights`` to a new ``nn.Parameter`` object.
        This matters because ``torch.optim`` optimizers capture a
        reference to the actual ``nn.Parameter`` object at
        construction time (typically via ``model.parameters()``); if
        this method replaced ``self.weights`` with a new object, the
        optimizer would keep updating the old (orphaned) tensor while
        ``forward()`` read from the new one, silently breaking
        training. In-place ``.data`` mutation preserves the object
        identity the optimizer is holding, so ``self.weights`` remains
        the single, correctly-optimized source of truth regardless of
        how many device moves the parent model goes through.

        Parameters
        ----------
        fn : Callable
            The per-tensor transform being applied (e.g. a function
            that calls ``.to("cuda")`` on a tensor). Supplied
            internally by PyTorch; this module never calls ``fn``
            directly on ``self.weights``.
        recurse : bool, default True
            Forwarded to ``nn.Module._apply`` unchanged; standard
            PyTorch parameter for this hook.

        Returns
        -------
        VQCBranch
            ``self``, per the standard ``nn.Module._apply`` /
            ``nn.Module.to()`` chaining convention.
        """
        super()._apply(fn, recurse=recurse)

        with torch.no_grad():
            self.weights.data = self.weights.data.to("cpu")
            if self.weights.grad is not None:
                self.weights.grad = self.weights.grad.to("cpu")

        return self

    def forward(self, pooled_features: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        pooled_features : Tensor
            Temporal-mean-pooled SGF representation, shape
            (batch_size, input_dim). Pooling itself is performed by the
            caller (``ProposedModel``), NOT by this module — this
            branch owns only the projection and the VQC. May be on any
            device (e.g. CUDA, if the classical backbone is on GPU);
            this method transfers to/from CPU internally as needed for
            the QNode call and returns a tensor on the SAME device
            ``pooled_features`` was on.

        Returns
        -------
        Tensor
            Quantum prediction, shape (batch_size, output_dim), on the
            same device as the input ``pooled_features``.
        """

        if pooled_features.dim() != 2:
            raise ValueError(
                "VQCBranch expects a pooled 2D (batch_size, input_dim) "
                f"tensor; got a {pooled_features.dim()}D tensor of shape "
                f"{tuple(pooled_features.shape)}. Temporal pooling must "
                "be performed by the caller before calling VQCBranch."
            )

        original_device = pooled_features.device

        # Bounded transformation before angle encoding, for numerical
        # stability (rotation angles are periodic in 2*pi, but an
        # unbounded projection could otherwise wrap many times over
        # and destabilize gradients early in training). This is a
        # simple, documented interface choice — not quantum
        # preprocessing. Runs on original_device (e.g. CUDA), same as
        # before.
        raw_angles = self.projection(pooled_features)
        angles = torch.tanh(raw_angles) * torch.pi

        # ------------------------------------------------------------
        # GPU -> CPU boundary (angles only): default.qubit has no CUDA
        # execution path, so the per-batch QNode input must be a CPU
        # tensor at call time. `.to("cpu")` is autograd-tracked (not
        # `.detach()`, not a NumPy round-trip), so the gradient path
        # back through this transfer to `angles` (and from there to
        # `self.projection` and `pooled_features`) remains intact.
        #
        # self.weights is NOT transferred here: it is already
        # permanently CPU-resident (see _apply() above), so it is
        # passed straight into the QNode as-is.
        # ------------------------------------------------------------
        angles_cpu = angles.to("cpu")

        expectation_values = self._circuit(angles_cpu, self.weights)

        # QNode returns a list of `output_dim` tensors, each of shape
        # (batch_size,); stack into (batch_size, output_dim). This
        # stacked tensor is CPU-resident, matching the QNode's inputs.
        quantum_prediction_cpu = torch.stack(expectation_values, dim=-1)

        # ------------------------------------------------------------
        # CPU -> GPU boundary: return the prediction on whatever
        # device the caller's pooled_features originally lived on, so
        # LearnedScalarOutputFusion (and the rest of the model) sees a
        # single consistent device, matching classical_prediction.
        # `.to(original_device)` is likewise autograd-tracked.
        # ------------------------------------------------------------
        quantum_prediction = quantum_prediction_cpu.to(original_device)

        return quantum_prediction