"""Explicit residual-only training; never loads or evaluates test rows."""

import argparse
import json
from pathlib import Path

import torch
import pennylane as qml
from torch import nn

from experiments.baseline_experiment import _plot_learning_curve, _save_baseline_metrics
from training.metrics import compute_metrics
from utils.logger import get_logger

from .batch_order import batches
from .cache import CACHE_ROOT, CachedSplit, cache_identity, verify_cache, write_cache
from .data import ROOT, file_hash, reconstruct
from .frozen_baseline import FrozenBaseline, validation_gate
from .preflight import residual_checks
from .residual_branch import FAMILIES, FrozenResidualModel, canonical_state

HISTORY_KEYS = ("train_loss", "val_loss", "rmse", "mae", "mape", "r2", "nrmse")


def protocol():
    return {
        "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-5,
        "batch_size": 256, "max_epochs": 100, "patience": 15,
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-6},
        "quantum_shape": [2, 6, 2], "quantum_initialization": "Uniform(-0.1,0.1)",
        "common_seed_offset": 0, "quantum_seed_offset": 10000,
        "permutation_seed_offset": 20000, "epoch_indexing": "zero_based",
        "criterion": "lowest_validation_MSE", "baseline_fallback": False,
        "torch_version": str(torch.__version__), "pennylane_version": qml.__version__,
        "requirements_sha256": file_hash(ROOT / "requirements.txt"),
        "module_hashes": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


@torch.no_grad()
def validate(model, cache):
    model.eval()
    total = 0.0
    predictions, targets = [], []
    for start in range(0, len(cache), 256):
        pooled, baseline, target = cache[slice(start, start + 256)]
        prediction = model.forward_cached(pooled, baseline)
        total += nn.functional.mse_loss(prediction, target).item() * len(target)
        predictions.append(prediction)
        targets.append(target)
    return total / len(cache), compute_metrics(torch.cat(predictions).numpy(), torch.cat(targets).numpy())


def train_family(run, family, state, cache_dir, identity, preflight):
    directory = ROOT / f"experiments/{family}/horizon_15/run_{run}"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite or implicitly resume {directory}")
    directory.mkdir(parents=True)
    model = FrozenResidualModel(run, family, state)
    if model.frozen.sha256 != identity["checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint differs from accepted cache.")
    baseline_before = {key: value.clone() for key, value in model.frozen.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.residual.parameters() if p.requires_grad],
                                 lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    train = CachedSplit(cache_dir, "train")
    validation = CachedSplit(cache_dir, "validation")
    settings = {"family": family, "run": run, "seed": 41 + run, "protocol": protocol(),
                "cache_identity": identity, "training_complete": False}
    (directory / "settings.json").write_text(json.dumps(settings, indent=2))
    (directory / "preflight.json").write_text(json.dumps(preflight, indent=2))
    logger = get_logger(directory)
    history = {key: [] for key in HISTORY_KEYS}
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    best, stale = float("inf"), 0
    for epoch in range(100):
        model.train()
        total = 0.0
        epoch_batches = batches(len(train), 41 + run, epoch)
        for indices in epoch_batches:
            pooled, baseline, target = train[indices]
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_cached(pooled, baseline)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite residual training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        val_loss, metrics = validate(model, validation)
        if not torch.isfinite(torch.tensor(val_loss)):
            raise ValueError("Nonfinite validation loss.")
        history["train_loss"].append(total / len(train))
        history["val_loss"].append(val_loss)
        for key, value in metrics.items():
            history[key].append(value)
        logger.info("Epoch %d/100 | train_loss: %.6f | val_loss: %.6f | rmse: %.6f | mae: %.6f | mape: %.4f%% | r2: %.6f | nrmse: %.6f | lr: %.8f",
                    epoch + 1, history["train_loss"][-1], val_loss, metrics["rmse"], metrics["mae"],
                    metrics["mape"], metrics["r2"], metrics["nrmse"], optimizer.param_groups[0]["lr"])
        scheduler.step()
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best}, checkpoint_dir / "best_checkpoint.pt")
        else:
            stale += 1
        if not all(torch.equal(value, model.frozen.state_dict()[key]) for key, value in baseline_before.items()):
            raise AssertionError("Frozen baseline state changed.")
        (directory / "history.json").write_text(json.dumps(history, indent=2))
        if stale >= 15:
            logger.info("Early stopping at epoch %d.", epoch + 1)
            break
    _plot_learning_curve(history, directory / "plots/loss_curve.png")
    _save_baseline_metrics(history, directory / "results/baseline_metrics.json")
    settings["training_complete"] = True
    settings["best_checkpoint_sha256"] = file_hash(checkpoint_dir / "best_checkpoint.pt")
    (directory / "settings.json").write_text(json.dumps(settings, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--family", choices=(*FAMILIES, "both"), default="both")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    families = FAMILIES if args.family == "both" else (args.family,)
    for family in families:
        if (ROOT / f"experiments/{family}/horizon_15/run_{args.run}").exists():
            raise SystemExit("STOP: experiment directory already exists; no implicit overwrites.")
    datasets, feature_scaler, target_scaler, provenance = reconstruct(args.processed_csv)
    baseline = FrozenBaseline(args.run)
    gate = validation_gate(baseline, datasets["validation"], args.run)
    if not gate["passed"]:
        raise SystemExit("STOP: reconstructed validation does not reproduce the Proposed checkpoint.\n" + json.dumps(gate, indent=2))
    checks = residual_checks(datasets, feature_scaler, target_scaler, provenance, baseline, gate, args.run)
    identity = cache_identity(provenance, baseline, args.run, gate, "full")
    cache_dir = CACHE_ROOT / f"horizon_15/run_{args.run}"
    if not cache_dir.exists():
        write_cache(cache_dir, datasets, baseline, identity, feature_scaler, target_scaler)
    verify_cache(cache_dir, identity)
    state = canonical_state(41 + args.run)
    for family in families:
        train_family(args.run, family, state, cache_dir, identity, checks)


if __name__ == "__main__":
    main()
