"""Test evaluation for one completed residual experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import _save_metrics, _save_predictions
from evaluation.plots import EvaluationPlotter
from training.metrics import compute_metrics

from .data import FEATURES, ROOT, TOTAL_ROWS, VAL_END, Windows, file_hash, reconstruct
from .residual_branch import FAMILIES, FrozenResidualModel
from .run_experiment import protocol


def load_settings(run, family):
    directory = ROOT / f"experiments/{family}/horizon_15/run_{run}"
    record = json.loads((directory / "settings.json").read_text())
    identity = record["cache_identity"]
    if (not record["training_complete"] or record["family"] != family or
            record["run"] != run or record["seed"] != 41 + run or
            identity["run"] != run or identity["seed"] != 41 + run or
            identity["scope"] != "full"):
        raise ValueError("Selected run is incomplete or has mismatched run/seed metadata.")
    current_protocol = protocol()
    for key, value in current_protocol.items():
        saved = record["protocol"][key]
        if key == "module_hashes":
            # Evaluation changes do not change the fitted model or training protocol.
            matches = all(saved[name] == digest for name, digest in value.items()
                          if name != "evaluate.py")
        elif key == "torch_version":
            matches = saved.split("+")[0] == value.split("+")[0]
        else:
            matches = saved == value
        if not matches:
            raise ValueError(f"Selected run protocol changed: {key}")
    if file_hash(directory / "checkpoints/best_checkpoint.pt") != record["best_checkpoint_sha256"]:
        raise ValueError("Selected residual checkpoint changed after training.")
    checks = json.loads((directory / "preflight.json").read_text())
    gate = identity["validation_gate"]
    if (checks["test_rows_read"] != 0 or checks["quantum_trainable_counts"] != [24, 0] or
            not gate["passed"] or gate["checkpoint_sha256"] != identity["checkpoint_sha256"] or
            identity["frozen_extractor_sha256"] != current_protocol["module_hashes"]["frozen_baseline.py"]):
        raise ValueError("Missing or mismatched preflight/validation evidence.")
    return record


def load_test(processed_csv, identity):
    datasets, feature_scaler, target_scaler, provenance = reconstruct(processed_csv)
    del datasets
    saved = identity["provenance"]
    for key, value in provenance.items():
        # The notebook is a reference; data.py is the executable reconstruction.
        if key in ("source_path", "notebook_sha256", "scalers") or key.endswith("_version"):
            continue
        if saved[key] != value:
            raise ValueError(f"Source preprocessing provenance changed: {key}")
    for name, scaler in (("feature", feature_scaler), ("target", target_scaler)):
        expected = saved["scalers"][name]
        actual = provenance["scalers"][name]
        for key in ("class", "fitted_rows", "n_features"):
            if expected[key] != actual[key]:
                raise ValueError(f"Train-fitted {name} scaler metadata changed: {key}")
        for key, attribute in (("mean", "mean_"), ("scale", "scale_"), ("variance", "var_")):
            if not np.allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Train-fitted {name} scaler changed: {key}")
            # Restore exact saved statistics after independently verifying the train fit.
            setattr(scaler, attribute, np.asarray(expected[key], dtype=np.float64))
    frame = pd.read_csv(processed_csv, parse_dates=["timestamp"])
    if len(frame) != TOTAL_ROWS:
        raise ValueError("Unexpected finalized source row count.")
    test = frame.iloc[VAL_END:TOTAL_ROWS]
    if (test["timestamp"].isna().any() or not test["timestamp"].is_monotonic_increasing or
            test["timestamp"].duplicated().any()):
        raise ValueError("Invalid test timestamp order.")
    features = feature_scaler.transform(test.loc[:, list(FEATURES)])
    target = target_scaler.transform(test["Active_Power"].to_numpy().reshape(-1, 1))
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite finalized test data.")
    dataset = Windows(features, target)
    if len(dataset) != 159768:
        raise ValueError("Unexpected test window count.")
    return dataset, target_scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    record = load_settings(args.run, args.family)
    identity = record["cache_identity"]
    directory = ROOT / f"evaluation/{args.family}/horizon_15/run_{args.run}"
    if directory.exists():
        raise FileExistsError("Refusing to overwrite existing evaluation results.")
    model = FrozenResidualModel(args.run, args.family)
    if model.frozen.sha256 != identity["checkpoint_sha256"]:
        raise ValueError("Original Proposed checkpoint changed.")
    checkpoint_path = ROOT / f"experiments/{args.family}/horizon_15/run_{args.run}/checkpoints/best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    original_baseline = {key: value.clone() for key, value in model.frozen.state_dict().items()}
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.equal(value, model.frozen.state_dict()[key]) for key, value in original_baseline.items()):
        raise ValueError("Residual checkpoint contains a changed baseline.")
    dataset, target_scaler = load_test(args.processed_csv, identity)
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, target in DataLoader(dataset, batch_size=256, shuffle=False,
                                        generator=torch.Generator().manual_seed(0)):
            predictions.append(model(inputs))
            targets.append(target)
    prediction = target_scaler.inverse_transform(torch.cat(predictions).numpy().reshape(-1, 1)).reshape(-1)
    target = target_scaler.inverse_transform(torch.cat(targets).numpy().reshape(-1, 1)).reshape(-1)
    _save_predictions(directory / "results/predictions.csv", prediction, target)
    _save_metrics(directory / "results/evaluation_metrics.json", compute_metrics(prediction, target))
    plotter = EvaluationPlotter(directory / "plots", max_plot_samples=1000)
    plotter.plot_predictions(prediction, target)
    plotter.plot_residuals(prediction, target)
    plotter.plot_prediction_scatter(prediction, target)


if __name__ == "__main__":
    main()
