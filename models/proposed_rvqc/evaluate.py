"""Final test evaluation, gated on completion of all ten paired experiments."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import _save_metrics, _save_predictions
from evaluation.plots import EvaluationPlotter
from training.metrics import compute_metrics

from .cache import CACHE_ROOT, verify_cache
from .data import FEATURES, ROOT, TOTAL_ROWS, VAL_END, Windows, file_hash
from .residual_branch import FAMILIES, FrozenResidualModel
from .run_experiment import protocol


def release_settings():
    settings = {}
    current_protocol = protocol()
    source_prefix = None
    for family in FAMILIES:
        for run in range(1, 6):
            directory = ROOT / f"experiments/{family}/horizon_15/run_{run}"
            record = json.loads((directory / "settings.json").read_text())
            if (not record["training_complete"] or record["protocol"] != current_protocol or
                    record["family"] != family or record["run"] != run):
                raise ValueError("All ten experiments must be complete under this exact protocol.")
            checkpoint = directory / "checkpoints/best_checkpoint.pt"
            if file_hash(checkpoint) != record["best_checkpoint_sha256"]:
                raise ValueError("Selected residual checkpoint changed after training.")
            identity = record["cache_identity"]
            prefix = identity["provenance"]["source_prefix_sha256"]
            if source_prefix is not None and prefix != source_prefix:
                raise ValueError("Paired experiments did not use the same source prefix.")
            source_prefix = prefix
            checks = json.loads((directory / "preflight.json").read_text())
            if checks["test_rows_read"] != 0 or checks["quantum_trainable_counts"] != [24, 0]:
                raise ValueError("Missing required preflight evidence.")
            settings[family, run] = record
    return settings


def load_released_test(processed_csv, identity, cache_dir):
    """Called only after the explicit release and all training-completion checks."""
    digest = hashlib.sha256()
    with Path(processed_csv).open("rb") as stream:
        for _ in range(VAL_END + 1):
            line = stream.readline()
            if not line:
                raise ValueError("Incomplete source data.")
            digest.update(line)
    if digest.hexdigest() != identity["provenance"]["source_prefix_sha256"]:
        raise ValueError("Source train/validation prefix changed.")
    frame = pd.read_csv(processed_csv, parse_dates=["timestamp"])
    if len(frame) != TOTAL_ROWS:
        raise ValueError("Unexpected finalized source row count.")
    test = frame.iloc[VAL_END:TOTAL_ROWS]
    if not test["timestamp"].is_monotonic_increasing or test["timestamp"].duplicated().any():
        raise ValueError("Invalid test timestamp order.")
    feature_scaler = pickle.loads((cache_dir / "feature_scaler.pkl").read_bytes())
    target_scaler = pickle.loads((cache_dir / "target_scaler.pkl").read_bytes())
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
    parser.add_argument("--release-test", action="store_true", help="Explicitly release test access after all training finishes.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.release_test:
        raise SystemExit("Test access is locked; --release-test and all ten completed experiments are required.")
    records = release_settings()
    record = records[args.family, args.run]
    identity = record["cache_identity"]
    cache_dir = CACHE_ROOT / f"horizon_15/run_{args.run}"
    verify_cache(cache_dir, identity)
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
    dataset, target_scaler = load_released_test(args.processed_csv, identity, cache_dir)
    torch.set_num_threads(args.threads)
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
