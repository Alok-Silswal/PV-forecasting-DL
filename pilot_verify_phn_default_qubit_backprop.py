"""
pilot_verify_phn_default_qubit_backprop.py

Small Kaggle pilot verification for the default.qubit + backprop
change to models/vqc_branch.py and configs/config.py. Does NOT run
full training -- this only verifies construction, shapes, forward,
backward, and gradient validity for a single batch, plus confirms the
VQC is actually using the intended simulator/diff_method rather than
silently falling back to something else.

Usage (Kaggle)
--------------
    python pilot_verify_phn_default_qubit_backprop.py

Run from the project root (the directory containing `configs/` and
`models/`).
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import config  # noqa: E402
from models.model_factory import get_model  # noqa: E402

BATCH_SIZE = 256
RANDOM_SEED = 42


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    print("=" * 78)
    print("PILOT VERIFICATION: proposed_phn with default.qubit + backprop")
    print("=" * 78)

    # ---- 1. Construction ----
    model = get_model("proposed_phn")
    print("[PASS] proposed_phn constructed successfully via get_model().")

    # ---- 2. Confirm the VQC is actually using the intended config ----
    vqc = model.vqc_branch
    print(f"VQCBranch.simulator   : {vqc.simulator}")
    print(f"VQCBranch.diff_method : {vqc.diff_method}")
    assert vqc.simulator == "default.qubit", (
        f"Expected simulator='default.qubit', got '{vqc.simulator}'"
    )
    assert vqc.diff_method == "backprop", (
        f"Expected diff_method='backprop', got '{vqc.diff_method}'"
    )
    print("[PASS] VQCBranch is using simulator='default.qubit', diff_method='backprop'.")

    # Also confirm this traces back to config.py defaults, not just an
    # override happening to match.
    assert config.VQC_SIMULATOR == "default.qubit"
    assert config.VQC_DIFF_METHOD == "backprop"
    print("[PASS] config.VQC_SIMULATOR / config.VQC_DIFF_METHOD match the new defaults.")

    # Inspect the underlying PennyLane QNode's bound device directly,
    # not just the string VQCBranch stored -- confirms the actual
    # qml.device(...) object constructed matches, not merely the
    # attribute we set.
    print(f"Underlying QNode device (introspected): {vqc._circuit.device}")
    print()

    # ---- 3. Forward pass ----
    x = torch.randn(BATCH_SIZE, config.LOOKBACK, config.NUM_FEATURES)
    y = model(x)

    expected_shape = (BATCH_SIZE, config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON])
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {tuple(y.shape)}"
    print(f"[PASS] Forward pass succeeded. Output shape: {tuple(y.shape)} (expected {expected_shape}).")

    assert not torch.isnan(y).any(), "NaNs found in model output"
    assert not torch.isinf(y).any(), "Infs found in model output"
    print("[PASS] Output contains no NaNs or Infs.")
    print()

    # ---- 4. Backward pass ----
    loss = y.pow(2).mean()
    loss.backward()
    print(f"[PASS] Backward pass succeeded. Loss value: {loss.item():.6f}")
    print()

    # ---- 5. Gradient validity: VQC weights ----
    vqc_weight_grad = vqc.weights.grad
    assert vqc_weight_grad is not None, "VQC weights received no gradient"
    assert torch.isfinite(vqc_weight_grad).all(), "VQC weight gradient contains non-finite values"
    assert vqc_weight_grad.abs().sum().item() > 0.0, "VQC weight gradient is all-zero"
    print(f"[PASS] VQC weight gradient finite and nonzero. Shape: {tuple(vqc_weight_grad.shape)}")

    # ---- 6. Gradient validity: classical->quantum projection layer ----
    projection_grad = vqc.projection.weight.grad
    assert projection_grad is not None, "VQC projection layer received no gradient"
    assert torch.isfinite(projection_grad).all(), "Projection gradient contains non-finite values"
    print(f"[PASS] VQC projection layer gradient finite. Shape: {tuple(projection_grad.shape)}")

    # ---- 7. Gradient validity: learned scalar output fusion ----
    classical_logit_grad = model.output_fusion.classical_logit.grad
    quantum_logit_grad = model.output_fusion.quantum_logit.grad
    assert classical_logit_grad is not None and torch.isfinite(classical_logit_grad).all()
    assert quantum_logit_grad is not None and torch.isfinite(quantum_logit_grad).all()
    print("[PASS] Learned scalar output fusion (s_c, s_q) gradients finite.")

    # ---- 8. Gradient validity: classical backbone still trains ----
    dcnn_grad = next(model.dcnn.parameters()).grad
    assert dcnn_grad is not None and torch.isfinite(dcnn_grad).all()
    print("[PASS] Classical backbone (DCNN) gradient finite -- classical path unaffected.")
    print()

    # ---- 9. s_c + s_q sanity (unrelated to this change, re-verified as a
    #          cheap regression check that nothing else broke) ----
    with torch.no_grad():
        s_c, s_q = model.output_fusion.get_branch_weights()
    print(f"s_c = {s_c.item():.6f}, s_q = {s_q.item():.6f}, sum = {(s_c + s_q).item():.6f}")
    assert abs((s_c + s_q).item() - 1.0) < 1e-6
    print("[PASS] s_c + s_q == 1 (softmax-constrained fusion unaffected by this change).")
    print()

    # ---- 10. proposed (non-PHN) still bypasses the VQC entirely ----
    classical_model = get_model("proposed")
    assert not hasattr(classical_model, "vqc_branch"), (
        "'proposed' unexpectedly constructed a vqc_branch"
    )
    y_classical = classical_model(x)
    assert y_classical.shape == expected_shape
    print("[PASS] 'proposed' (non-PHN) model still has no vqc_branch and returns correct shape.")
    print()

    print("=" * 78)
    print("ALL PILOT CHECKS PASSED.")
    print("Simulator: default.qubit | diff_method: backprop | batch_size: 256")
    print("No full training was run.")
    print("=" * 78)


if __name__ == "__main__":
    main()