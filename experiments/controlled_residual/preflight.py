"""Bounded engineering verification; optional full validation gate, never training."""

import argparse
import json
import tempfile
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Subset
from models.proposed_rvqc.cache import cache_identity, write_cache
from models.proposed_rvqc.data import reconstruct
from .cache import check_live, prepare
from .model import ARMS, Correction
from .simulator import RVQCSimulator, pennylane_reference
from .run_experiment import select_device


def reference_agreement(device=torch.device("cpu")):
    generator = torch.Generator().manual_seed(142)
    reference = pennylane_reference()
    simulator = RVQCSimulator().to(device)
    maximum = [0.0, 0.0, 0.0]
    for batch in (1, 3, 7):
        data = torch.rand(batch, 6, generator=generator) * (2 * torch.pi) - torch.pi
        weights = torch.randn(2, 6, 2, generator=generator) * 0.9
        coefficient = torch.randn(batch, 6, generator=generator)
        theta, angles = data.to(device).requires_grad_(), weights.to(device).requires_grad_()
        reference_theta, reference_angles = data.clone().requires_grad_(), weights.clone().requires_grad_()
        actual = simulator(theta, angles)
        expected = torch.stack(reference(reference_theta, reference_angles), dim=1).float()
        actual_gradients = torch.autograd.grad((actual * coefficient.to(device)).sum(), (theta, angles))
        expected_gradients = torch.autograd.grad((expected * coefficient).sum(), (reference_theta, reference_angles))
        for i, (left, right) in enumerate([(actual.cpu(), expected), *[(a.cpu(), b) for a, b in zip(actual_gradients, expected_gradients)]]):
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
            maximum[i] = max(maximum[i], (left - right).abs().max().item())
    return {"batches": [1, 3, 7], "rtol": 2e-5, "atol": 2e-5,
            "output_max_abs_error": maximum[0], "input_gradient_max_abs_error": maximum[1],
            "angle_gradient_max_abs_error": maximum[2], "device": str(device)}


def smoke(device):
    generator = torch.Generator().manual_seed(42)
    pooled = torch.randn(16, 128, generator=generator).to(device)
    base = torch.randn(16, 3, generator=generator).to(device)
    target = base + 0.2 * torch.tanh(pooled[:, :3])
    results = {}
    for arm in ARMS:
        model = Correction(arm).to(device)
        if not torch.equal(base + model(pooled), base):
            raise AssertionError("Initial combined forecast changed.")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        initial = nn.functional.mse_loss(base, target).item()
        meaningful = False
        for step in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(base + model(pooled), target)
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise AssertionError("Missing or nonfinite correction gradient.")
            if step == 1:
                upstream = [p for name, p in model.named_parameters() if not name.startswith("readout.")]
                meaningful = all(p.grad.abs().sum() > 0 for p in upstream)
                if not meaningful:
                    raise AssertionError("Upstream gradients did not appear after readout update.")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        final = nn.functional.mse_loss(base + model(pooled), target).item()
        if not final < initial:
            raise AssertionError("Tiny synthetic correction did not reduce MSE.")
        results[arm] = {"windows": 16, "steps": 8, "initial_mse": initial, "final_mse": final,
                        "upstream_gradients_after_readout_update": meaningful,
                        "trainable_parameters": sum(p.numel() for p in model.parameters())}
    return {"scientific_evidence": False, "arms": results}


def profile(device):
    generator = torch.Generator().manual_seed(42)
    data = torch.rand(32, 6, generator=generator) * torch.pi
    weights = torch.randn(2, 6, 2, generator=generator) * 0.1
    results = []
    reference, simulator = pennylane_reference(), RVQCSimulator().to(device)
    for backend in ("torch", "pennylane"):
        execution_device = device if backend == "torch" else torch.device("cpu")
        theta = data.to(execution_device).clone().requires_grad_()
        angles = weights.to(execution_device).clone().requires_grad_()
        def step():
            output = simulator(theta, angles) if backend == "torch" else torch.stack(reference(theta, angles), dim=1).float()
            torch.autograd.grad(output.square().sum(), (theta, angles))
        step()
        if execution_device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            step()
        if execution_device.type == "cuda":
            torch.cuda.synchronize()
        results.append({"backend": backend, "device": str(execution_device), "batch_size": 32,
                        "threads": torch.get_num_threads(),
                        "forward_backward_batch_ms": (time.perf_counter() - start) * 1000 / 3})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path)
    parser.add_argument("--run", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    report = {"reference_agreement": reference_agreement(device), "engineering_smoke": smoke(device),
              "micro_profile": profile(device), "torch_version": str(torch.__version__),
              "cuda_available": torch.cuda.is_available()}
    if args.processed_csv:
        directory, identity, baseline, status, checks = prepare(args.processed_csv, args.run, device, build=False)
        report["validation_gate"] = identity["validation_gate"]
        report["cache_status"] = status
        report["cache_live_checks"] = checks
        datasets, feature_scaler, target_scaler, provenance = reconstruct(args.processed_csv)
        samples = {split: Subset(dataset, list(range(8))) for split, dataset in datasets.items()}
        smoke_identity = cache_identity(provenance, baseline, args.run, identity["validation_gate"], "smoke")
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            baseline.to("cpu")
            write_cache(cache_dir, samples, baseline, smoke_identity, feature_scaler, target_scaler)
            report["tiny_cache_live_checks"] = check_live(cache_dir, samples, baseline, torch.device("cpu"), "smoke")
        report["frozen_backbone_parameters"] = sum(p.numel() for p in baseline.parameters())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
