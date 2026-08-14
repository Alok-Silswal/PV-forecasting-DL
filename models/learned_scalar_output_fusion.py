"""
Learned Scalar Output Fusion for the PHN (Parallel Hybrid Network)
extension of the PV power forecasting model.

Combines the classical prediction ``y_c`` (from the existing, unchanged
``MLPHead``) and the quantum prediction ``y_q`` (from ``VQCBranch``)
using two trainable, softmax-constrained scalar weights:

    s_c = exp(a_c) / (exp(a_c) + exp(a_q))
    s_q = exp(a_q) / (exp(a_c) + exp(a_q))

    y_hat = s_c * y_c + s_q * y_q

with ``s_c + s_q == 1`` and both non-negative by construction. The two
logits ``a_c`` and ``a_q`` are initialized to zero, so training starts
at ``s_c = s_q = 0.5``.

This is deliberately NOT a gating network (no dependence on the input
sample), NOT an unrestricted pair of coefficients, and NOT followed by
any additional MLP or concatenation — exactly two learned scalars
governing a convex combination of the two branch outputs, exactly as
specified.
"""

import torch
import torch.nn as nn
from torch import Tensor


class LearnedScalarOutputFusion(nn.Module):
    """
    Softmax-constrained learned scalar fusion of two prediction
    branches.

    Attributes
    ----------
    classical_logit : nn.Parameter
        Scalar logit ``a_c``, initialized to 0.
    quantum_logit : nn.Parameter
        Scalar logit ``a_q``, initialized to 0.
    """

    def __init__(self) -> None:
        super().__init__()

        self.classical_logit = nn.Parameter(torch.zeros(1))
        self.quantum_logit = nn.Parameter(torch.zeros(1))

    def get_branch_weights(self) -> tuple[Tensor, Tensor]:
        """
        Compute the current softmax-constrained branch weights.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(s_c, s_q)``, each a scalar tensor, satisfying
            ``s_c + s_q == 1`` and both non-negative. Left attached to
            the autograd graph — callers that only want to log/inspect
            the values (e.g. after training) should call ``.detach()``
            or ``.item()`` themselves; this method does not detach, per
            the requirement that these remain trainable and
            graph-connected during training.
        """

        logits = torch.cat([self.classical_logit, self.quantum_logit])
        weights = torch.softmax(logits, dim=0)

        classical_weight = weights[0]
        quantum_weight = weights[1]

        return classical_weight, quantum_weight

    def forward(self, classical_prediction: Tensor, quantum_prediction: Tensor) -> Tensor:
        """
        Fuse the classical and quantum predictions.

        Parameters
        ----------
        classical_prediction : Tensor
            ``y_c``, shape (batch_size, output_dim).
        quantum_prediction : Tensor
            ``y_q``, shape (batch_size, output_dim).

        Returns
        -------
        Tensor
            Final fused prediction, shape (batch_size, output_dim).
        """

        classical_weight, quantum_weight = self.get_branch_weights()

        fused_prediction = (
            classical_weight * classical_prediction
            + quantum_weight * quantum_prediction
        )

        return fused_prediction