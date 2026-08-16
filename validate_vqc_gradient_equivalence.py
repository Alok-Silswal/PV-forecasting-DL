"""
validate_vqc_gradient_equivalence.py

Standalone, temporary, read-only validation script. Does NOT modify
any project file and does NOT change the VQC architecture (qubits,
depth, gates, encoding, observables, output dimension are all
identical to models/vqc_branch.py). This script exists solely to
answer one question with direct numerical evidence: do

    A) lightning.qubit + diff_method="adjoint"   (current baseline)
    C) default.qubit    + diff_method="backprop"  (candidate)

produce equivalent GRADIENTS -- not just equivalent forward outputs --
for both (1) the trainable VQC weights and (2) the VQC input tensor,
for a batch of 256 samples.

Why this check is necessary even though forward outputs already agree
------------------------------------------------------------------------
Adjoint differentiation and backpropagation are structurally different
algorithms:
  - "adjoint" performs a single forward pass to build the final state,
    then reverses through the circuit by applying the INVERSE of each
    gate and accumulating <bra| d(gate)/d(theta) |ket> overlaps -- it
    never constructs or differentiates through an explicit
    computational graph of the full statevector evolution.
  - "backprop" builds a full autodiff graph of the entire simulated
    statevector evolution (every gate application is an ordinary
    differentiable tensor op) and differentiates it via reverse-mode
    automatic differentiation, exactly like a standard PyTorch/NumPy
    computation graph.
Both are mathematically EXACT for a noiseless circuit with analytic
(shots=None) expectation values -- there is no reason a priori for
them to disagree beyond floating-point rounding. But floating-point
rounding differs meaningfully between "accumulate via explicit gate
inversion in double precision C++" (adjoint, lightning.qubit) and
"accumulate via a PyTorch/NumPy autodiff graph" (backprop,
default.qubit), and outputs agreeing to 1e-7 does not by itself prove
gradients agree -- gradients are a separate computation with their own
error accumulation, so this must be checked directly.

Usage (Kaggle)
--------------
    python validate_vqc_gradient_equivalence.py

Fully self-contained; does not import project modules, so it can run
standalone.
"""

import pennylane as qml
import torch


# =============================================================================
# Configuration -- identical circuit/shapes to the project's VQCBranch
# =============================================================================

BATCH_SIZE = 256
NUM_QUBITS = 2
DEPTH = 2
OUTPUT_DIM = 3          # <Z0>, <Z1>, <Z0 Z1>
INPUT_DIM = 128
RANDOM_SEED = 42

# Tolerances, justified per-quantity below rather than a single
# blanket number.
WEIGHT_GRAD_ATOL = 1e-5   # matches the forward-output tolerance already
                          # used and approved in the prior benchmark;
                          # weight gradients are the direct optimization
                          # signal for the VQC's 12 trainable parameters
                          # (depth=2 * num_qubits=2 * 2 angles), so this
                          # is held to the same standard as the output
                          # check already accepted.
INPUT_GRAD_ATOL = 1e-5   # same standard applied to the gradient that
                          # flows back into the classical projection
                          # layer (Linear(128, 2)) and, from there, into
                          # the rest of the classical backbone -- this
                          # is the gradient path that actually matters
                          # for whether the classical network trains
                          # correctly under either configuration.
RELATIVE_TOL = 1e-3      # applied only where the reference gradient
                          # magnitude is large enough that a relative
                          # comparison is meaningful (see below); a
                          # looser bound than the absolute tolerances
                          # because relative error near zero-crossings
                          # is not informative.
MIN_MAGNITUDE_FOR_RELATIVE = 1e-4


# =============================================================================
# Circuit definition -- copy-identical to models/vqc_branch.py's
# _build_qnode, duplicated here so this script is fully standalone.
# =============================================================================

def build_qnode(device_name: str, diff_method: str):
    device = qml.device(device_name, wires=NUM_QUBITS)

    observables = [
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
    ][:OUTPUT_DIM]

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")

        for layer in range(DEPTH):
            for qubit in range(NUM_QUBITS):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
            qml.CNOT(wires=[0, 1])

        return [qml.expval(observable) for observable in observables]

    return circuit


def _compute_output_and_grads(device_name: str, diff_method: str, inputs: torch.Tensor, weights: torch.Tensor):
    """
    Run one forward + backward pass and return (output, grad_weights,
    grad_inputs), with a FRESH leaf tensor for both inputs and weights
    each call so gradients from a prior configuration cannot leak in.
    """

    circuit = build_qnode(device_name, diff_method)

    weights_leaf = weights.clone().detach().requires_grad_(True)
    inputs_leaf = inputs.clone().detach().requires_grad_(True)

    output = torch.stack(circuit(inputs_leaf, weights_leaf), dim=-1)  # (B, OUTPUT_DIM)

    # Use a fixed, deterministic reduction identical across both
    # configurations so the backward pass being compared is the same
    # scalar function of the same outputs in both cases.
    loss = output.pow(2).mean()
    loss.backward()

    return output.detach(), weights_leaf.grad.detach(), inputs_leaf.grad.detach()


def _report_tensor_comparison(name: str, tensor_a: torch.Tensor, tensor_c: torch.Tensor, atol: float):
    diff = (tensor_a - tensor_c).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    ref_magnitude = tensor_a.abs()
    # Relative difference only computed where the reference is large
    # enough to make the ratio meaningful; entries near zero are
    # excluded to avoid reporting a misleadingly huge relative error
    # from dividing by a near-zero reference value.
    meaningful_mask = ref_magnitude > MIN_MAGNITUDE_FOR_RELATIVE
    if meaningful_mask.any():
        relative_diff = diff[meaningful_mask] / ref_magnitude[meaningful_mask]
        max_relative_diff = relative_diff.max().item()
        mean_relative_diff = relative_diff.mean().item()
        num_meaningful = int(meaningful_mask.sum().item())
    else:
        max_relative_diff = float("nan")
        mean_relative_diff = float("nan")
        num_meaningful = 0

    finite_a = torch.isfinite(tensor_a).all().item()
    finite_c = torch.isfinite(tensor_c).all().item()
    nonzero_a = (tensor_a.abs().sum().item() > 0.0)
    nonzero_c = (tensor_c.abs().sum().item() > 0.0)

    agree_absolute = max_abs_diff < atol
    agree_relative = (
        (max_relative_diff < RELATIVE_TOL) if num_meaningful > 0 else True
    )
    agree = agree_absolute and agree_relative

    print(f"--- {name} ---")
    print(f"  shape                         : {tuple(tensor_a.shape)}")
    print(f"  max absolute difference       : {max_abs_diff:.6e}   (tolerance {atol:.0e})")
    print(f"  mean absolute difference      : {mean_abs_diff:.6e}")
    if num_meaningful > 0:
        print(
            f"  max relative difference       : {max_relative_diff:.6e}   "
            f"(tolerance {RELATIVE_TOL:.0e}, computed over {num_meaningful}/"
            f"{tensor_a.numel()} entries with |A| > {MIN_MAGNITUDE_FOR_RELATIVE:.0e})"
        )
        print(f"  mean relative difference       : {mean_relative_diff:.6e}")
    else:
        print(
            f"  relative difference           : not computed "
            f"(all reference values below {MIN_MAGNITUDE_FOR_RELATIVE:.0e}; "
            f"absolute tolerance is the meaningful criterion here)"
        )
    print(f"  A finite / nonzero             : {finite_a} / {nonzero_a}")
    print(f"  C finite / nonzero             : {finite_c} / {nonzero_c}")
    print(f"  --> Agree within tolerance      : {agree}")
    print()

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_relative_diff": max_relative_diff,
        "finite_a": finite_a,
        "finite_c": finite_c,
        "nonzero_a": nonzero_a,
        "nonzero_c": nonzero_c,
        "agree": agree,
    }


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    # Identical inputs and weights for BOTH configurations.
    pooled_features = torch.randn(BATCH_SIZE, INPUT_DIM)
    projection = torch.nn.Linear(INPUT_DIM, NUM_QUBITS)
    torch.nn.init.xavier_uniform_(projection.weight)
    torch.nn.init.constant_(projection.bias, 0.0)
    with torch.no_grad():
        raw_angles = projection(pooled_features)
        shared_inputs = torch.tanh(raw_angles) * torch.pi  # (BATCH_SIZE, NUM_QUBITS)

    shared_weights = 0.01 * torch.randn(DEPTH, NUM_QUBITS, 2)

    print("=" * 88)
    print("GRADIENT EQUIVALENCE VALIDATION: lightning.qubit+adjoint vs default.qubit+backprop")
    print("=" * 88)
    print(f"Batch size: {BATCH_SIZE}   Qubits: {NUM_QUBITS}   Depth: {DEPTH}   Output dim: {OUTPUT_DIM}")
    print(f"Input tensor shape: {tuple(shared_inputs.shape)}   Weights shape: {tuple(shared_weights.shape)}")
    print(
        "Reduction used for backward pass in both configurations: "
        "loss = output.pow(2).mean()  (identical scalar function of "
        "identical outputs, applied identically in both cases)"
    )
    print("=" * 88)
    print()

    print("Running Configuration A: lightning.qubit + adjoint ...")
    output_a, grad_weights_a, grad_inputs_a = _compute_output_and_grads(
        "lightning.qubit", "adjoint", shared_inputs, shared_weights
    )
    print("Running Configuration C: default.qubit + backprop ...")
    output_c, grad_weights_c, grad_inputs_c = _compute_output_and_grads(
        "default.qubit", "backprop", shared_inputs, shared_weights
    )
    print()

    # =========================================================================
    # 1. Output comparison (re-verified here, independent of the prior
    #    benchmark run, with the exact tensors used for this gradient check)
    # =========================================================================
    print("=" * 88)
    print("1) OUTPUT COMPARISON")
    print("=" * 88)
    output_result = _report_tensor_comparison(
        "VQC output (batch_size, output_dim)", output_a, output_c, atol=1e-5
    )

    # =========================================================================
    # 2. Gradient w.r.t. trainable VQC weights
    # =========================================================================
    print("=" * 88)
    print("2) GRADIENT COMPARISON: VQC trainable weights (shape: depth, num_qubits, 2)")
    print("=" * 88)
    weight_grad_result = _report_tensor_comparison(
        "grad(loss)/d(weights)", grad_weights_a, grad_weights_c, atol=WEIGHT_GRAD_ATOL
    )

    # =========================================================================
    # 3. Gradient w.r.t. VQC input tensor (what flows back into the
    #    classical projection layer / classical backbone)
    # =========================================================================
    print("=" * 88)
    print("3) GRADIENT COMPARISON: VQC input tensor (shape: batch_size, num_qubits)")
    print("=" * 88)
    input_grad_result = _report_tensor_comparison(
        "grad(loss)/d(inputs)", grad_inputs_a, grad_inputs_c, atol=INPUT_GRAD_ATOL
    )

    # =========================================================================
    # Verdict
    # =========================================================================
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)

    all_finite = (
        weight_grad_result["finite_a"] and weight_grad_result["finite_c"]
        and input_grad_result["finite_a"] and input_grad_result["finite_c"]
    )
    all_nonzero = (
        weight_grad_result["nonzero_a"] and weight_grad_result["nonzero_c"]
        and input_grad_result["nonzero_a"] and input_grad_result["nonzero_c"]
    )
    all_agree = (
        output_result["agree"] and weight_grad_result["agree"] and input_grad_result["agree"]
    )

    print(f"All gradients finite            : {all_finite}")
    print(f"All gradients nonzero            : {all_nonzero}")
    print(f"Output agrees within tolerance   : {output_result['agree']}")
    print(f"Weight gradients agree           : {weight_grad_result['agree']}")
    print(f"Input gradients agree            : {input_grad_result['agree']}")
    print()

    if all_finite and all_nonzero and all_agree:
        print(
            "SAFE (pending your review of the numbers above): outputs and "
            "gradients (both weights and inputs) agree within the stated, "
            "justified tolerances, and all gradients are finite and "
            "nonzero. This supports treating default.qubit+backprop as a "
            "numerically equivalent SIMULATOR/DIFFERENTIATION-METHOD "
            "substitution for lightning.qubit+adjoint on this exact "
            "circuit -- not a change to the VQC's mathematical "
            "definition, since the circuit (qubits, depth, gates, "
            "encoding, observables, output dimension) is byte-for-byte "
            "identical between configurations; only the numerical "
            "backend computing the same mathematical expectation values "
            "and their exact gradients has changed."
        )
    else:
        print(
            "NOT SAFE: at least one of finiteness, nonzero-ness, or "
            "tolerance agreement failed above. Do not adopt "
            "default.qubit+backprop without investigating the specific "
            "failing quantity reported above first."
        )
    print("=" * 88)


if __name__ == "__main__":
    main()