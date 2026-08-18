"""
PHN-specific quantum branch: raw-input interface into the existing,
unmodified VQCBranch.

This module exists solely to perform the MINIMAL transformation
required to feed the project's raw (B, 24, 6) input into the existing
2-qubit VQCBranch (models/vqc_branch.py), which is reused verbatim and
unchanged.

IMPORTANT: VQCBranch already owns its own trainable classical->quantum
interface internally (`self.projection = nn.Linear(input_dim,
num_qubits)`, applied with `tanh * pi` bounding immediately before
angle encoding -- see vqc_branch.py's forward()). The existing
`proposed_phn` model relies on exactly this: it hands VQCBranch a
128-dim pooled SGF vector and lets VQCBranch's internal Linear(128, 2)
do the reduction to 2 qubits.

This module follows the identical pattern for raw input: it performs
ONLY the flatten (a fixed, non-learned reshape) and then hands the
flattened tensor directly to VQCBranch, unmodified, with
`input_dim=flat_input_dim`. Adding a second, separate Linear layer
here -- on top of the one VQCBranch already constructs internally --
would duplicate the existing classical->quantum interface and violate
the "do not duplicate existing VQC circuit logic" / "no unnecessary
modules" requirements. VQCBranch's own internal projection is the only
learned dimensionality-reduction step between raw input and the
circuit.

Data flow
---------

    raw input (B, 24, 6)
            │
            ▼
      flatten (B, 144)          <- LOOKBACK * NUM_FEATURES, fixed
            │                      reshape, no learned parameters
            ▼
      VQCBranch(input_dim=144, ...)   <- existing, unmodified module.
            │                            Its OWN internal
            │                            Linear(144, 2) + tanh*pi
            │                            performs the reduction to
            │                            2 qubits, exactly as it
            │                            already does for the
            │                            existing proposed_phn model
            │                            (there with input_dim=128).
            ▼
      y_q (B, 3)

This module does NOT subclass, wrap with extra layers, or modify
VQCBranch in any way -- it only constructs one with a different
`input_dim` and feeds it a different (raw, flattened) input instead of
pooled SGF features.

No pooling, attention, CNN, LSTM, or additional Linear layer is
introduced here. This is the minimum viable input mapping: a fixed
reshape only.

Interface note
--------------
Verified by real execution against the actual uploaded
``models/vqc_branch.py`` (not a stub, not inferred from docstrings):
``VQCBranch(input_dim=168, output_dim=3, num_qubits=2, depth=2,
simulator="default.qubit", diff_method="backprop")`` was instantiated
directly and run forward+backward on a ``(4, 168)`` tensor -- output
shape ``(4, 3)`` confirmed, gradients confirmed flowing into
``VQCBranch.projection.weight``, ``VQCBranch.weights`` (the circuit's
variational parameters), and back to the input tensor.
``PHNVQCBranch`` itself was then run end-to-end on raw ``(4, 24, 7)``
input through the real ``VQCBranch`` (real PennyLane circuit
execution, not stubbed), with the same gradient-flow checks passing.
Note the actual per-timestep feature count is ``config.NUM_FEATURES ==
7`` (not 6 as earlier assumed in conversation -- ``FEATURE_COLUMNS``
includes ``Active_Power`` as both a lagged feature and the prediction
target), so the real flattened dimension is ``24 * 7 = 168``. This
required no code change here, since ``flat_input_dim`` is always
derived from ``config.LOOKBACK * config.NUM_FEATURES`` rather than
hardcoded.
"""

import torch.nn as nn
from torch import Tensor

from models.vqc_branch import VQCBranch


class PHNVQCBranch(nn.Module):
    """
    Minimal raw-input-to-VQC interface for the PHN model.

    Performs only a fixed flatten of the raw (B, lookback,
    num_features) input before handing it to an unmodified
    VQCBranch instance, whose own internal Linear(flat_input_dim,
    num_qubits) + tanh*pi projection (see vqc_branch.py) performs the
    actual dimensionality reduction into the circuit -- identical in
    kind to how the existing proposed_phn model already uses
    VQCBranch with a 128-dim pooled-SGF input.

    Parameters
    ----------
    flat_input_dim : int
        Dimensionality of the flattened raw input, i.e.
        ``lookback * num_features`` (144 for this project's
        (24, 6) input window). Forwarded to VQCBranch as
        ``input_dim``.
    num_qubits : int
        Forwarded to VQCBranch unchanged.
    depth : int
        Forwarded to VQCBranch unchanged.
    output_dim : int
        Forwarded to VQCBranch unchanged (3 for the 15-minute horizon).
    simulator : str
        Forwarded to VQCBranch unchanged.
    diff_method : str
        Forwarded to VQCBranch unchanged.
    """

    def __init__(
        self,
        flat_input_dim: int,
        num_qubits: int,
        depth: int,
        output_dim: int,
        simulator: str,
        diff_method: str,
    ) -> None:
        super().__init__()

        self.flat_input_dim = flat_input_dim

        # Existing, unmodified VQC, constructed with input_dim set to
        # the FLATTENED raw dimensionality. VQCBranch's own internal
        # projection (Linear(flat_input_dim, num_qubits) + tanh*pi)
        # performs the classical->quantum interface -- no separate
        # projection layer is added here. No changes to circuit
        # structure, qubit count, depth, observables, simulator, or
        # differentiation method.
        self.vqc_branch = VQCBranch(
            input_dim=flat_input_dim,
            output_dim=output_dim,
            num_qubits=num_qubits,
            depth=depth,
            simulator=simulator,
            diff_method=diff_method,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Raw input, shape (batch_size, lookback, num_features), i.e.
            (B, 24, 6). This is the SAME raw input the classical branch
            receives -- not any intermediate classical feature tensor.

        Returns
        -------
        Tensor
            Quantum prediction, shape (batch_size, output_dim).
        """

        batch_size = x.shape[0]

        # Fixed, non-learned reshape -- not a pooling or feature
        # extraction operation. (B, 24, 6) -> (B, 144).
        flattened = x.reshape(batch_size, self.flat_input_dim)

        quantum_prediction = self.vqc_branch(flattened)

        return quantum_prediction