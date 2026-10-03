"""Evaluate a selected Proposed-RVQC checkpoint; never train or refit scalers."""

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from configs import config
from evaluation.evaluator import Evaluator
from evaluation.plots import EvaluationPlotter
from evaluation.evaluate import _save_metrics, _save_predictions
from training.metrics import compute_metrics
from experiments.residual_learning.extract_residual_dataset import (
    RECOVERY_TARGET_ATOL, RECOVERY_TARGET_RMSE,
)
from experiments.residual_learning.run_residual_audit import forecast_metrics
from models.proposed_rvqc import ProposedRVQC
from . import run_experiment as training
from .run_experiment import (
    ROOT, FEATURES, VAL_END, EXPECTED_ROWS, WindowDataset,
    validate_frame, fingerprint, sha256, resolve_processed_csv,
)


def restore_scaler(statistics: dict, feature_names: list[str] | None = None) -> StandardScaler:
    """Restore fitted attributes recorded by training, without reading fitting rows."""
    scaler = StandardScaler()
    for key in ("mean_", "scale_", "var_"):
        setattr(scaler, key, np.asarray(statistics[key], dtype=np.float64))
    scaler.n_samples_seen_ = np.asarray(statistics["n_samples_seen_"]).item()
    scaler.n_features_in_ = len(scaler.mean_)
    if (not all(np.isfinite(getattr(scaler, key)).all() for key in ("mean_", "scale_", "var_"))
            or (scaler.scale_ <= 0).any() or (scaler.var_ < 0).any()):
        raise ValueError("Invalid saved scaler statistics.")
    if feature_names is not None:
        scaler.feature_names_in_ = np.asarray(feature_names, dtype=object)
    return scaler


def load_evaluation_frame(csv_path: Path, settings: dict, smoke: bool) -> pd.DataFrame:
    """Normal evaluation seeks directly to test; smoke uses 34 non-test fixture rows."""
    with csv_path.open("rb") as handle:
        header = handle.readline()
        if smoke:
            contents = header + b"".join(handle.readline() for _ in range(34))
        else:
            handle.seek(settings["test_row_byte_offset"])
            contents = header + handle.read()
    frame = pd.read_csv(io.BytesIO(contents), parse_dates=["timestamp"])
    if list(frame.columns) != settings["csv_columns"]:
        raise ValueError("CSV columns differ from training.")
    validate_frame(frame, 34 if smoke else EXPECTED_ROWS - VAL_END)
    return frame


def write_results(output: Path, metrics: dict, truth: np.ndarray, predictions: np.ndarray) -> None:
    """Write standard results and plots, refusing existing outputs."""
    metrics = dict(metrics)
    metrics["mape"] = compute_metrics(predictions, truth)["mape"]
    metrics = {key: metrics[key] for key in ("rmse", "mae", "mape", "r2", "nrmse")}
    if (output / "results").exists() or (output / "plots").exists():
        raise FileExistsError(f"Standard evaluation outputs already exist: {output}")
    _save_predictions(output / "results/predictions.csv", predictions.reshape(-1), truth.reshape(-1))
    saved = pd.read_csv(output / "results/predictions.csv", float_precision="round_trip")
    np.testing.assert_array_equal(saved.Actual.to_numpy(), truth.reshape(-1))
    np.testing.assert_array_equal(saved.Predicted.to_numpy(), predictions.reshape(-1))
    _save_metrics(output / "results/evaluation_metrics.json", metrics)
    if json.loads((output / "results/evaluation_metrics.json").read_text()) != metrics:
        raise RuntimeError("Saved evaluation metrics differ.")
    plotter = EvaluationPlotter(output / "plots", config.MAX_PLOT_SAMPLES)
    plotter.plot_predictions(predictions.reshape(-1), truth.reshape(-1))
    plotter.plot_residuals(predictions.reshape(-1), truth.reshape(-1))
    plotter.plot_prediction_scatter(predictions.reshape(-1), truth.reshape(-1))


def format_existing(output: Path) -> None:
    """Convert saved results only; never load a checkpoint or access the dataset."""
    summary_path = output / "summary.json"
    predictions_path = output / "predictions.npz"
    if not summary_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"Format-only conversion requires existing summary.json and predictions.npz in {output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256(predictions_path) != summary["predictions_sha256"]:
        raise ValueError("Saved prediction archive hash differs from the evaluation record.")
    with np.load(predictions_path, allow_pickle=False) as arrays:
        truth = arrays["y_true_original"]
        predictions = arrays["prediction_proposed_rvqc"]
        verified = forecast_metrics(truth, predictions)
        for scope in ("aggregate", "per_output"):
            if scope == "aggregate":
                pairs = [(verified[scope], summary["metrics"]["proposed_rvqc"][scope])]
            else:
                pairs = [(verified[scope][str(h)], summary["metrics"]["proposed_rvqc"][scope][str(h)]) for h in range(3)]
            for actual, recorded in pairs:
                for key in actual:
                    np.testing.assert_allclose(actual[key], recorded[key], rtol=1e-12, atol=1e-12)
        write_results(output, summary["metrics"]["proposed_rvqc"]["aggregate"], truth, predictions)
    # Retain legacy evidence after verifying a lossless CSV round trip.
    saved = pd.read_csv(output / "results/predictions.csv", float_precision="round_trip")
    np.testing.assert_array_equal(saved.Actual.to_numpy(), truth.reshape(-1))
    np.testing.assert_array_equal(saved.Predicted.to_numpy(), predictions.reshape(-1))
    logging.info("Reformatted saved evaluation; metrics unchanged; no training or inference: %s", output)


def run(args: argparse.Namespace, model_class=ProposedRVQC, family: str = "proposed_rvqc") -> None:
    if not 1 <= args.run_number <= config.NUM_RUNS:
        raise ValueError("Run number outside project NUM_RUNS.")
    suffix = Path(f"horizon_15/run_{args.run_number}")
    if args.smoke_test:
        suffix = Path("_smoke") / suffix
    if args.format_existing:
        if family != "proposed_rvqc":
            raise ValueError("Legacy conversion applies only to Proposed-RVQC.")
        format_existing(ROOT / "evaluation" / family / suffix)
        return
    destination = ROOT / "experiments" / family / suffix
    checkpoint = destination / "checkpoints/best_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint missing; run training first: {checkpoint}")
    settings_path = destination / "settings.json"
    history_path = destination / "training_history.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    completion = json.loads(history_path.read_text(encoding="utf-8"))["completion"]
    if (settings["experiment"] != f"end-to-end {family}"
            or settings["seed"] != config.RANDOM_SEED + args.run_number - 1
            or settings["benchmark_only"]
            or settings["run_number"] != args.run_number or settings["smoke_test"] != args.smoke_test
            or completion["test_accessed"] or completion["settings_sha256"] != fingerprint(settings_path)
            or sha256(checkpoint) != completion["training_files_sha256"]["checkpoints/best_checkpoint.pt"]):
        raise ValueError("Training completion/checkpoint provenance conflicts.")
    for name, digest in settings["source_sha256"].items():
        if fingerprint(ROOT / name) != digest:
            raise ValueError(f"Training source changed: {name}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    torch.set_num_threads(settings["cpu_threads"])
    csv_path = resolve_processed_csv(args.processed_csv)
    output = ROOT / "evaluation" / family / suffix
    baseline_root = ROOT / f"evaluation/proposed/horizon_15/run_{args.run_number}/results"
    if output.exists() and any(p.is_file() for p in output.rglob("*")):
        raise FileExistsError(f"Evaluation output exists; refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    protected = training.historical_snapshot()
    protected.update({baseline_root / name: sha256(baseline_root / name)
                      for name in ("predictions.csv", "evaluation_metrics.json")} if not args.smoke_test else {})
    try:
        frame = load_evaluation_frame(csv_path, settings, args.smoke_test)
        feature_scaler = restore_scaler(settings["feature_scaler"], FEATURES)
        target_scaler = restore_scaler(settings["target_scaler"])
        features = feature_scaler.transform(frame[FEATURES]).astype(np.float32)
        targets = target_scaler.transform(frame.Active_Power.to_numpy().reshape(-1, 1))[:, 0].astype(np.float32)
        dataset = WindowDataset(features, targets)
        result = Evaluator(model_class(), DataLoader(dataset, batch_size=settings["configuration"]["BATCH_SIZE"]),
                           checkpoint, target_scaler, device).evaluate()
        corrected = result["predictions"].reshape(-1, 3)
        truth = result["targets"].reshape(-1, 3)
        if not args.smoke_test:
            historical = pd.read_csv(baseline_root / "predictions.csv")
            if list(historical.columns) != ["Actual", "Predicted"] or len(historical) != len(dataset) * 3:
                raise ValueError("Historical baseline sample/output contract differs.")
            saved_truth = historical.Actual.to_numpy().reshape(-1, 3)
            difference = truth.astype(np.float64) - saved_truth
            if (not np.isfinite(difference).all() or np.abs(difference).max() > RECOVERY_TARGET_ATOL
                    or np.sqrt(np.mean(difference ** 2)) > RECOVERY_TARGET_RMSE):
                raise ValueError("Test targets differ from authoritative baseline.")
            truth = saved_truth
        write_results(output, forecast_metrics(truth, corrected)["aggregate"], truth, corrected)
        logging.info("Evaluation complete: %s (test accessed=%s); no training performed.", output, not args.smoke_test)
    finally:
        training.assert_protected(protected)


def main(model_class=ProposedRVQC, family: str = "proposed_rvqc") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-number", type=int, required=True, choices=range(1, config.NUM_RUNS + 1))
    parser.add_argument("--processed-csv", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Backbone device; RVQC stays CPU.")
    parser.add_argument("--smoke-test", action="store_true", help="Load disposable smoke checkpoint; evaluate eight non-test windows.")
    parser.add_argument("--format-existing", action="store_true", help="Verify/convert saved summary/NPZ only; retain originals; no inference.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args, model_class, family)


if __name__ == "__main__":
    main()
