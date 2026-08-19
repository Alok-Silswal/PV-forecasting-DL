"""
verify_5qubit_proposed_phn.py

Direct, single-configuration sanity check for `proposed_phn` at
VQC_NUM_QUBITS=5, per the exact checklist in the request:

  - model constructs
  - output shape = (4, 3)
  - output is finite
  - VQC executes
  - VQC gradients are finite
  - output-fusion gradients are finite
  - CUDA execution works (if a CUDA device is available in this
    environment; otherwise this is reported honestly as CPU-only and
    is NOT claimed as a CUDA/T4 verification)

This is a sanity check only. It does not train and does not report
predictive performance.
"""

import sys
import torch

sys.path.insert(0, ".")

from configs import config
from models.model_factory import get_model

assert config.VQC_NUM_QUBITS == 5, (
    f"Expected config.VQC_NUM_QUBITS=5, got {config.VQC_NUM_QUBITS}. "
    "Config change did not take effect."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")
if not torch.cuda.is_available():
    print(
        "NOTE: no CUDA device in this environment -- this run verifies "
        "CPU correctness only. The CUDA/Tesla T4 path (classical "
        "backbone on GPU, VQC on CPU, cross-device autograd) is NOT "
        "exercised here and must be verified on your actual T4 machine."
    )

torch.manual_seed(config.RANDOM_SEED)

model = get_model("proposed_phn")
assert hasattr(model, "vqc_branch"), "proposed_phn did not build a vqc_branch"
assert model.vqc_branch.num_qubits == 5, (
    f"vqc_branch.num_qubits={model.vqc_branch.num_qubits}, expected 5"
)
assert tuple(model.vqc_branch.projection.weight.shape) == (5, model.vqc_branch.projection.in_features), (
    f"projection shape {tuple(model.vqc_branch.projection.weight.shape)} "
    "does not match Linear(in_features, 5)"
)
print(f"model constructs: OK  (vqc_branch.num_qubits={model.vqc_branch.num_qubits}, "
      f"projection={tuple(model.vqc_branch.projection.weight.shape)})")

model = model.to(device)
model.train()

x = torch.randn(4, config.LOOKBACK, config.NUM_FEATURES, device=device)
y_true = torch.randn(4, config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON], device=device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)
loss_fn = torch.nn.MSELoss()

optimizer.zero_grad(set_to_none=True)
out = model(x)

assert tuple(out.shape) == (4, 3), f"output shape {tuple(out.shape)} != (4, 3)"
print(f"output shape: OK  {tuple(out.shape)}")

assert torch.isfinite(out).all(), "output contains non-finite values"
print(f"output finite: OK")

loss = loss_fn(out, y_true)
loss.backward()

vqc_grad_finite = all(
    torch.isfinite(p.grad).all()
    for p in model.vqc_branch.parameters()
    if p.grad is not None
)
vqc_has_grad = any(p.grad is not None for p in model.vqc_branch.parameters())
assert vqc_has_grad, "VQC branch has no gradients -- it did not execute in the backward pass"
assert vqc_grad_finite, "VQC branch gradients are non-finite"
print(f"VQC executes + VQC gradients finite: OK  "
      f"(vqc_branch.weights.grad finite: {torch.isfinite(model.vqc_branch.weights.grad).all().item()}, "
      f"vqc_branch.projection.weight.grad finite: {torch.isfinite(model.vqc_branch.projection.weight.grad).all().item()})")

fusion_grad_finite = all(
    torch.isfinite(p.grad).all()
    for p in model.output_fusion.parameters()
    if p.grad is not None
)
fusion_has_grad = any(p.grad is not None for p in model.output_fusion.parameters())
assert fusion_has_grad, "output_fusion has no gradients"
assert fusion_grad_finite, "output_fusion gradients are non-finite"
print(f"output-fusion gradients finite: OK")

optimizer.step()
print("optimizer.step(): OK (no exception raised)")

print()
print("ALL CHECKS PASSED for proposed_phn @ num_qubits=5")