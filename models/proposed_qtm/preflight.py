"""Bounded synthetic overfit/profile checks; never reads forecasting datasets."""

import argparse
import json
import time

import torch

from configs import config
from models.model_factory import get_model
from utils.seed import set_seed
from .quantum_simulator import SixQubitSimulator, pennylane_reference


def reference_agreement() -> dict:
    generator = torch.Generator().manual_seed(42)
    theta = (torch.rand(4, 6, generator=generator) * torch.pi).requires_grad_()
    angles = (torch.randn(2, 6, 2, generator=generator) * 0.3).requires_grad_()
    actual = SixQubitSimulator()(theta, angles)
    expected = torch.stack(pennylane_reference()(theta, angles), dim=1).float()
    actual_grads = torch.autograd.grad(actual.square().sum(), (theta, angles))
    expected_grads = torch.autograd.grad(expected.square().sum(), (theta, angles))
    pairs = [(actual, expected), *zip(actual_grads, expected_grads)]
    for left, right in pairs:
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
    return dict(zip(("output_max_abs_error", "theta_gradient_max_abs_error", "angle_gradient_max_abs_error"),
                    [(left - right).abs().max().item() for left, right in pairs]))


def profile(device: torch.device) -> list[dict]:
    simulator = SixQubitSimulator().to(device)
    reference = pennylane_reference()
    results = []
    for batch_size in (16, 64):
        theta = (torch.rand(batch_size, 6, device=device) * torch.pi).requires_grad_()
        angles = (torch.randn(2, 6, 2, device=device) * 0.1).requires_grad_()
        for backend in ("torch", "pennylane"):
            def step():
                q = (simulator(theta, angles) if backend == "torch"
                     else torch.stack(reference(theta, angles), dim=1).float())
                torch.autograd.grad(q.square().sum(), (theta, angles))
            step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            results.append({"backend": backend, "device": str(device), "batch_size": batch_size,
                            "forward_backward_batch_ms": elapsed * 1000 / 3,
                            "samples_per_second": 3 * batch_size / elapsed})
    return results


def small_overfit(device: torch.device, steps: int = 40) -> dict:
    if steps not in range(1, 101):
        raise ValueError("Engineering overfit is limited to 1..100 steps.")
    set_seed(42)
    model = get_model("proposed_qtm", seed=42, quantum_backend="torch").to(device)
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(16, 24, 7, generator=generator).to(device)
    targets = inputs[:, -1, :3].clone()
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(
        config.LEARNING_RATE, config.WEIGHT_DECAY, config.QTM_QUANTUM_LR_MULTIPLIER))
    model.eval()
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(inputs), targets).item()
    start = time.perf_counter()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite synthetic overfit loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(inputs), targets).item()
    return {"synthetic_windows": 16, "steps": steps, "initial_eval_mse": initial,
            "final_eval_mse": final, "reduction_percent": 100 * (initial - final) / initial,
            "seconds": time.perf_counter() - start, "scientific_evidence": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overfit-steps", type=int, default=40, choices=range(1, 101))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable.")
    torch.set_num_threads(2)
    # Verify the training backend before its synthetic optimization check.
    agreement = reference_agreement()
    print(json.dumps({"reference_agreement": agreement,
                      "overfit": small_overfit(device, args.overfit_steps),
                      "quantum_micro_profile": profile(device)}, indent=2))


if __name__ == "__main__":
    main()
