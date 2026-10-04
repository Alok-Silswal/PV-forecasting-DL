"""Aggregate completed validation-only pilots without ranking depths."""

import csv
import json

from .data import file_hash
from .depth_pilot import PILOT_ROOT

FIELDS = ("family", "run", "seed", "blocks", "quantum_angles", "trainable_parameter_count",
          "best_epoch", "epochs_executed", "early_stopped", "selected_lr",
          "baseline_val_mse", "baseline_val_rmse", "best_val_mse", "best_val_rmse",
          "best_val_mae", "best_val_r2", "delta_val_mse_vs_baseline",
          "delta_val_rmse_vs_baseline", "theta_l1_change", "theta_l2_change",
          "theta_mean_abs_change", "theta_max_abs_change", "theta_rms_change", "output_path")


def summarize(root=PILOT_ROOT):
    rows = []
    paths = sorted(root.glob("blocks_*/run_*/settings.json"))
    paths += sorted(root.glob("frozen/blocks_*/run_*/settings.json"))
    for path in paths:
        settings = json.loads(path.read_text())
        if not settings["training_complete"]:
            continue
        directory = path.parent
        if file_hash(directory / "checkpoints/best_checkpoint.pt") != settings["best_checkpoint_sha256"]:
            raise ValueError(f"Pilot checkpoint changed: {directory}")
        metrics = json.loads((directory / "results/validation_metrics.json").read_text())
        angles = json.loads((directory / "results/quantum_angle_analysis.json").read_text())
        if metrics["test_rows_read"] != 0 or settings["test_rows_read"] != 0:
            raise ValueError("Summary accepts validation-only pilots.")
        if any(metrics[k] != settings[k] for k in ("run", "seed", "family", "blocks")):
            raise ValueError(f"Inconsistent pilot metadata: {directory}")
        identity = settings["cache_identity"]
        gate = identity["validation_gate"]
        if not gate["passed"] or gate["checkpoint_sha256"] != identity["checkpoint_sha256"]:
            raise ValueError(f"Baseline validation provenance mismatch: {directory}")
        if metrics["metric_units"] != "train_standardized_target":
            raise ValueError("Pilot and baseline metrics must use the same standardized units.")
        baseline_mse = gate["actual_val_loss"]
        baseline_rmse = gate["actual_metrics"]["rmse"]
        rows.append({**{k: metrics[k] for k in
                        ("family", "run", "seed", "blocks", "quantum_angles", "trainable_parameter_count",
                         "best_epoch", "epochs_executed", "early_stopped", "best_val_mse",
                         "best_val_rmse", "best_val_mae", "best_val_r2")},
                     "selected_lr": metrics["selected_learning_rate"],
                     "baseline_val_mse": baseline_mse, "baseline_val_rmse": baseline_rmse,
                     "delta_val_mse_vs_baseline": metrics["best_val_mse"] - baseline_mse,
                     "delta_val_rmse_vs_baseline": metrics["best_val_rmse"] - baseline_rmse,
                     **{"theta_" + k: angles[k] for k in
                        ("l1_change", "l2_change", "mean_abs_change", "max_abs_change", "rms_change")},
                     "output_path": str(directory.resolve())})
    if not rows:
        raise ValueError("No completed validation-only depth pilots found.")
    destination = root / "depth_summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return destination, len(rows)


if __name__ == "__main__":
    destination, count = summarize()
    print(f"Validation-only summary: {count} runs\nResults saved to: {destination}")
