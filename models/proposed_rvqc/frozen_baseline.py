"""Load and observe the existing ProposedModel without changing its forward."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.proposed_model import ProposedModel
from training.metrics import compute_metrics
from .data import ROOT, file_hash


def checkpoint_path(run):
    if run not in range(1, 6):
        raise ValueError("Run must be 1..5.")
    return ROOT / f"experiments/proposed/horizon_15/run_{run}/checkpoints/best_checkpoint.pt"


class FrozenBaseline(nn.Module):
    def __init__(self, run, device="cpu"):
        super().__init__()
        # Baseline construction must not advance any residual RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.baseline = ProposedModel()
        self.path = checkpoint_path(run)
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        self.baseline.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if self.baseline.mlp_head.output_layer.out_features != 3:
            raise ValueError("This experiment requires the unchanged 15-minute configuration.")
        self.checkpoint = {k: checkpoint[k] for k in ["epoch", "best_val_loss"]}
        self.sha256 = file_hash(self.path)
        self.baseline.requires_grad_(False).eval().to(device)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, inputs):
        self.baseline.eval()
        captured = []
        hook = self.baseline.mlp_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        try:
            prediction = self.baseline(inputs)
        finally:
            hook.remove()
        fused = captured[0]
        if fused.shape != (len(inputs), 24, 128) or prediction.shape != (len(inputs), 3):
            raise ValueError("Unexpected frozen ProposedModel tensor shapes.")
        return fused.mean(dim=1), prediction


def validation_gate(baseline, dataset, run, device="cpu"):
    """Fail closed on disagreement with checkpoint-associated normalized metrics."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    predictions, targets = [], []
    total_loss = 0.0
    for inputs, target in loader:
        _, prediction = baseline(inputs.to(device))
        total_loss += nn.functional.mse_loss(prediction.cpu(), target).item() * len(inputs)
        predictions.append(prediction.cpu())
        targets.append(target)
    actual_loss = total_loss / len(dataset)
    actual_metrics = compute_metrics(torch.cat(predictions).numpy(), torch.cat(targets).numpy())
    history_path = ROOT / f"experiments/proposed/horizon_15/run_{run}/history.json"
    history = json.loads(history_path.read_text())
    epoch = baseline.checkpoint["epoch"]
    expected_loss = float(baseline.checkpoint["best_val_loss"])
    if not np.isclose(history["val_loss"][epoch], expected_loss, rtol=1e-8, atol=1e-10):
        raise ValueError("Baseline history does not match its checkpoint.")
    # Float32 CPU/GPU kernels and reduction order can differ, not preprocessing.
    checks = {"val_loss": bool(np.isclose(actual_loss, expected_loss, rtol=1e-5, atol=1e-7))}
    for key, value in actual_metrics.items():
        checks[key] = bool(np.isclose(value, history[key][epoch], rtol=1e-4, atol=1e-6))
    return {
        "passed": all(checks.values()), "checks": checks,
        "actual_val_loss": actual_loss, "expected_val_loss": expected_loss,
        "actual_metrics": actual_metrics,
        "expected_metrics": {key: history[key][epoch] for key in actual_metrics},
        "checkpoint_epoch": epoch, "checkpoint_sha256": baseline.sha256,
        "history_sha256": file_hash(history_path),
        "loss_tolerance": {"rtol": 1e-5, "atol": 1e-7},
        "metric_tolerance": {"rtol": 1e-4, "atol": 1e-6},
    }
