"""Isolated validation-only depth pilot; no test interface or depth selection."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experiments.baseline_experiment import _plot_learning_curve
from utils.logger import get_logger

from .batch_order import batches
from .cache import CACHE_ROOT, CachedSplit, cache_identity, verify_cache, write_cache
from .data import ROOT, VAL_END, WINDOW_COUNTS, file_hash, reconstruct
from .depth_branch import DepthResidualBranch, depth_state
from .frozen_baseline import FrozenBaseline, validation_gate
from .residual_branch import FAMILIES
from .run_experiment import HISTORY_KEYS, protocol, validate

PILOT_ROOT = ROOT / "experiments/proposed_rvqc_depth_pilot"


def output_path(run, blocks, family):
    root = PILOT_ROOT if family == "proposed_rvqc" else PILOT_ROOT / "frozen"
    return root / f"blocks_{blocks}/run_{run}"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def angle_analysis(initial, selected):
    delta = selected.detach().double() - initial.detach().double()

    def statistics(values):
        return {"l1_change": values.abs().sum().item(),
                "l2_change": values.norm().item(),
                "max_abs_change": values.abs().max().item(),
                "mean_abs_change": values.abs().mean().item(),
                "rms_change": values.square().mean().sqrt().item()}

    return {"source": "initial_vs_best_validation_checkpoint", **statistics(delta),
            "per_block": [{"block": i + 1, **statistics(values)}
                          for i, values in enumerate(delta)],
            "initial_angles": initial.tolist(), "selected_angles": selected.tolist(),
            "delta_theta": delta.tolist()}


def train_depth(run, blocks, family, cache_dir, identity, baseline):
    directory = output_path(run, blocks, family)
    if directory.exists():
        raise FileExistsError(f"Pilot output already exists; no implicit resume: {directory}")
    if (identity["provenance"]["test_rows_read"] != 0 or
            identity["provenance"]["source_rows_read"] != VAL_END or
            identity["provenance"]["window_counts"] != WINDOW_COUNTS or
            identity["checkpoint_sha256"] != baseline.sha256):
        raise ValueError("Pilot requires verified train/validation-only baseline caches.")
    if baseline.training or any(p.requires_grad for p in baseline.parameters()):
        raise ValueError("Proposed baseline must be frozen and in eval mode.")
    state = depth_state(41 + run, blocks)
    model = DepthResidualBranch(family, state, blocks)
    before = {k: v.clone() for k, v in baseline.state_dict().items()}
    initial = state["quantum_angles"].clone()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    train, validation = CachedSplit(cache_dir, "train"), CachedSplit(cache_dir, "validation")
    settings = {"family": family, "run": run, "seed": 41 + run, "blocks": blocks,
                "quantum_angles": 12 * blocks, "trainable_quantum_angles":
                model.quantum_angles.numel() if model.quantum_angles.requires_grad else 0,
                "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "protocol": {**protocol(), "quantum_shape": [blocks, 6, 2]},
                "cache_identity": identity, "test_rows_read": 0,
                "metric_units": "train_standardized_target", "training_complete": False,
                "pilot_source_sha256": file_hash(__file__),
                "depth_branch_sha256": file_hash(Path(__file__).with_name("depth_branch.py"))}
    save_json(directory / "settings.json", settings)
    logger = get_logger(directory)
    history = {key: [] for key in (*HISTORY_KEYS, "learning_rate")}
    best, stale = float("inf"), 0
    checkpoint_path = directory / "checkpoints/best_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    for epoch in range(100):
        model.train()
        total = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for indices in batches(len(train), 41 + run, epoch):
            pooled, prediction, target = train[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model.forward_cached(pooled, prediction), target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite pilot training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        val_loss, metrics = validate(model, validation)
        if not all(torch.isfinite(torch.tensor(v)) for v in (val_loss, *metrics.values())):
            raise ValueError("Nonfinite pilot validation statistics.")
        history["train_loss"].append(total / len(train))
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(learning_rate)
        for key, value in metrics.items():
            history[key].append(value)
        logger.info("Epoch %d | train_loss: %.6f | val_loss: %.6f | lr: %.8f",
                    epoch + 1, history["train_loss"][-1], val_loss, learning_rate)
        scheduler.step()
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "initial_quantum_angles": initial, "best_val_loss": best,
                        "validation_metrics": metrics, "learning_rate": learning_rate,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "run": run, "seed": 41 + run, "blocks": blocks, "family": family,
                        "baseline_checkpoint_sha256": baseline.sha256}, checkpoint_path)
        else:
            stale += 1
        if not all(torch.equal(v, baseline.state_dict()[k]) for k, v in before.items()):
            raise AssertionError("Frozen Proposed baseline changed.")
        if family == "proposed_rvqc_frozen" and not torch.equal(model.quantum_angles, initial):
            raise AssertionError("Frozen quantum angles changed.")
        save_json(directory / "history.json", history)
        if stale >= 15:
            break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    analysis = angle_analysis(checkpoint["initial_quantum_angles"], model.quantum_angles)
    results = {"run": run, "seed": 41 + run, "family": family, "blocks": blocks,
               "quantum_angles": settings["quantum_angles"],
               "trainable_quantum_angles": settings["trainable_quantum_angles"],
               "trainable_parameter_count": settings["trainable_parameter_count"],
               "best_epoch": checkpoint["epoch"] + 1, "epoch_indexing": "one_based",
               "best_val_mse": checkpoint["best_val_loss"],
               **{f"best_val_{k}": v for k, v in checkpoint["validation_metrics"].items()},
               "selected_learning_rate": checkpoint["learning_rate"],
               "final_learning_rate": history["learning_rate"][-1],
               "epochs_executed": len(history["val_loss"]), "early_stopped": stale >= 15,
               "stale_epochs": stale, "metric_units": settings["metric_units"], "test_rows_read": 0}
    save_json(directory / "results/validation_metrics.json", results)
    save_json(directory / "results/quantum_angle_analysis.json", analysis)
    _plot_learning_curve(history, directory / "plots/loss_curve.png")
    settings.update(training_complete=True, best_checkpoint_sha256=file_hash(checkpoint_path))
    save_json(directory / "settings.json", settings)
    print(f"RVQC depth pilot complete\nRun: {run}\nSeed: {41 + run}\nReupload blocks: {blocks}\n"
          f"Trainable quantum angles: {settings['trainable_quantum_angles']}\n"
          f"Best epoch: {results['best_epoch']}\nBest validation MSE: {results['best_val_mse']:.6f}\n"
          f"Best validation RMSE: {results['best_val_rmse']:.6f}\n"
          f"Best validation MAE: {results['best_val_mae']:.6f}\nBest validation R2: {results['best_val_r2']:.6f}\n"
          f"Quantum angle L2 change: {analysis['l2_change']:.6f}\n"
          f"Quantum angle mean absolute change: {analysis['mean_abs_change']:.6f}\nOutput path: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--blocks", type=int, choices=(1, 2, 3, 4), nargs="+", required=True)
    parser.add_argument("--family", choices=FAMILIES, default="proposed_rvqc")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if len(set(args.blocks)) != len(args.blocks):
        raise ValueError("Duplicate depths are not allowed.")
    for blocks in args.blocks:
        if output_path(args.run, blocks, args.family).exists():
            raise FileExistsError("Selected pilot already exists; official results are never reused as pilots.")
    datasets, feature_scaler, target_scaler, provenance = reconstruct(args.processed_csv)
    if set(datasets) != {"train", "validation"} or provenance["test_rows_read"] != 0:
        raise ValueError("Only train and validation are permitted.")
    baseline = FrozenBaseline(args.run)
    gate = validation_gate(baseline, datasets["validation"], args.run)
    if not gate["passed"]:
        raise SystemExit("STOP: reconstructed validation failed the Proposed checkpoint gate.")
    identity = cache_identity(provenance, baseline, args.run, gate, "full")
    cache_dir = CACHE_ROOT / f"depth_pilot/run_{args.run}"
    if not cache_dir.exists():
        write_cache(cache_dir, datasets, baseline, identity, feature_scaler, target_scaler)
    verify_cache(cache_dir, identity)
    del datasets
    for blocks in args.blocks:
        train_depth(args.run, blocks, args.family, cache_dir, identity, baseline)


if __name__ == "__main__":
    main()
