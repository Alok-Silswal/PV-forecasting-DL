"""Fit classical residual controls and assess them once on a held-out temporal block."""

import argparse
import csv
import json
import logging
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from experiments.residual_learning.extract_residual_dataset import (
    CHECKPOINT, CSV, ROOT, WARNINGS, extract_residual_dataset, output_paths,
    save_json, scaler_stats, setup, sha256,
)
from models.residual_learning.residual_mlp import ResidualMLP

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def load_partition(directory: Path, name: str, manifest: dict) -> dict[str, np.ndarray]:
    """Verify extraction integrity before using an array."""
    path = directory / f"{name}.npz"
    if sha256(path) != manifest["partitions"][name]["sha256"]:
        raise ValueError(f"Extracted partition checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    n = manifest["partitions"][name]["samples"]
    if data["z"].shape != (n, 128):
        raise ValueError(f"Invalid latent shape for {name}: {data['z'].shape}")
    for key in ("y_true_original", "y_hat_original", "residual_original"):
        if data[key].shape != (n, 3) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} in {name}.")
    if not np.isfinite(data["z"]).all():
        raise ValueError(f"Non-finite latent in {name}.")
    return data


def metric_values(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute metrics in original units; pooled R² uses the pooled target mean."""
    truth, prediction = truth.astype(np.float64).reshape(-1), prediction.astype(np.float64).reshape(-1)
    error = truth - prediction
    rmse = float(np.sqrt(np.mean(error ** 2)))
    span = float(np.ptp(truth))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(error))),
            "r2": float(r2_score(truth, prediction)), "nrmse": rmse / span if span else 0.0}


def forecast_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Preserve three horizons before computing pooled aggregate metrics."""
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("Forecast metrics require matching [N,3] arrays.")
    per_output = {str(h): metric_values(truth[:, h], prediction[:, h]) for h in range(3)}
    return {"aggregate": metric_values(truth, prediction), "per_output": per_output}


def deltas(current: dict, baseline: dict) -> dict:
    """Negative deltas indicate lower error; positive improvements indicate benefit."""
    result = {}
    for key in ("rmse", "mae"):
        delta = current[key] - baseline[key]
        result[key] = {"delta": delta,
                       "delta_percent": 100 * delta / baseline[key] if baseline[key] else None,
                       "improvement": -delta,
                       "improvement_percent": -100 * delta / baseline[key] if baseline[key] else None}
    return result


def mlp_predict(model: ResidualMLP, features: np.ndarray, device: torch.device,
                batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def fit_mlp(train_z: np.ndarray, train_r: np.ndarray, tune_z: np.ndarray,
            tune_r: np.ndarray, args: argparse.Namespace,
            device: torch.device) -> tuple[ResidualMLP, int, list[dict]]:
    """Select weights using tuning loss only; assessment arrays are never supplied."""
    model = ResidualMLP(args.hidden_size, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_z), torch.from_numpy(train_r)),
                        batch_size=args.mlp_batch_size, shuffle=True, generator=generator)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite residual MLP training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(features)
        tuning_prediction = mlp_predict(model, tune_z, device, args.mlp_batch_size)
        tuning_loss = float(np.mean((tuning_prediction.astype(np.float64) - tune_r) ** 2))
        if not np.isfinite(tuning_loss):
            raise RuntimeError("Non-finite residual MLP tuning loss.")
        history.append({"epoch": epoch, "train_mse": total / len(train_z), "tuning_mse": tuning_loss})
        logging.info("MLP epoch %d train_mse=%.6g tuning_mse=%.6g", epoch, total / len(train_z), tuning_loss)
        if tuning_loss < best_loss:
            best_loss, best_epoch, stale = tuning_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, history


def diagnostics(data: dict, predictions: dict[str, np.ndarray], directory: Path) -> dict:
    """Save two compact plots and descriptive row-lag autocorrelations."""
    residual = data["residual_original"].astype(np.float64)
    acf = {}
    for lag in (1, 3, 12):
        values = []
        for h in range(3):
            left, right = residual[:-lag, h], residual[lag:, h]
            values.append(float(np.corrcoef(left, right)[0, 1])
                          if len(left) > 1 and left.std() > 0 and right.std() > 0 else None)
        acf[str(lag)] = values
    selected = np.unique(np.linspace(0, len(residual) - 1, min(1200, len(residual)), dtype=int))
    for kind in ("residual_errors", "forecasts"):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for h, ax in enumerate(axes):
            times = data[f"target_{h}_timestamp"][selected]
            if kind == "forecasts":
                ax.plot(times, data["y_true_original"][selected, h], label="Actual", linewidth=0.8)
            for name, prediction in predictions.items():
                values = (data["y_true_original"] - prediction) if kind == "residual_errors" else prediction
                ax.plot(times, values[selected, h], label=name, linewidth=0.7, alpha=0.8)
            ax.set_ylabel(f"Output {h}")
            if kind == "residual_errors":
                ax.axhline(0, color="black", linewidth=0.5)
        axes[0].legend(ncol=4)
        axes[0].set_title(f"Assessment {kind.replace('_', ' ')} (original Active_Power units)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(directory / f"{kind}.png", dpi=120)
        plt.close(fig)
    return {"residual_mean_per_output": residual.mean(axis=0).tolist(),
            "residual_std_per_output": residual.std(axis=0).tolist(),
            "autocorrelation_by_window_start_lag": acf,
            "autocorrelation_note": "Overlapping, irregularly timed windows; descriptive correlations, not independent evidence."}


def run_audit(args: argparse.Namespace) -> None:
    """Lock fitted controls before reading the assessment dataset."""
    if args.max_samples is not None and args.max_samples < 32:
        raise ValueError("Smoke extraction needs at least 32 samples per block for diagnostics.")
    if min(args.epochs, args.patience, args.mlp_batch_size, args.extraction_batch_size) < 1:
        raise ValueError("Epochs, patience and batch sizes must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    started = time.perf_counter()
    artifact_dir, evaluation_dir = output_paths(args.max_samples)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        if not args.extract_if_missing:
            raise FileNotFoundError(f"Missing completed extraction: {manifest_path}. Run extractor or use --extract-if-missing.")
        extract_residual_dataset(args.extraction_batch_size, args.max_samples, args.seed, args.cpu_threads)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["max_samples_per_partition"] != args.max_samples or manifest["schema_version"] != 1:
        raise ValueError("Extraction mode/schema mismatch.")
    for path, key in ((CSV, "processed_csv_sha256"), (CHECKPOINT, "checkpoint_sha256")):
        if not path.is_file() or sha256(path) != manifest[key]:
            raise ValueError(f"Historical source changed or missing since extraction: {path}")
    fit_dir = artifact_dir / "residual_controls"
    if evaluation_dir.exists() or fit_dir.exists():
        raise FileExistsError("Refusing to overwrite existing audit outputs; select a fresh smoke cap or archive only new experiment outputs.")
    device = setup(args.seed, args.cpu_threads)
    train = load_partition(artifact_dir, "train", manifest)
    tuning = load_partition(artifact_dir, "tuning", manifest)
    latent_scaler = StandardScaler().fit(train["z"])
    residual_scaler = StandardScaler().fit(train["residual_original"])
    train_z = latent_scaler.transform(train["z"]).astype(np.float32)
    tune_z = latent_scaler.transform(tuning["z"]).astype(np.float32)
    train_r = residual_scaler.transform(train["residual_original"]).astype(np.float32)
    tune_r = residual_scaler.transform(tuning["residual_original"]).astype(np.float32)
    selected_ridge, selected_alpha, best_rmse = None, None, float("inf")
    ridge_scores = []
    for alpha in ALPHAS:
        ridge = Ridge(alpha=alpha).fit(train_z.astype(np.float64), train_r.astype(np.float64))
        prediction = residual_scaler.inverse_transform(ridge.predict(tune_z.astype(np.float64)))
        rmse = metric_values(tuning["residual_original"], prediction)["rmse"]
        ridge_scores.append({"alpha": alpha, "tuning_residual_rmse_original": rmse})
        logging.info("Ridge alpha=%g tuning residual RMSE=%g", alpha, rmse)
        if rmse < best_rmse:
            selected_ridge, selected_alpha, best_rmse = ridge, alpha, rmse
    model, best_epoch, history = fit_mlp(train_z, train_r, tune_z, tune_r, args, device)
    fit_dir.mkdir(parents=True, exist_ok=False)
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    for name, value in [("latent_scaler", latent_scaler), ("residual_scaler", residual_scaler), ("ridge", selected_ridge)]:
        with (fit_dir / f"{name}.pkl").open("xb") as handle:
            pickle.dump(value, handle)
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "hidden_size": args.hidden_size, "dropout": args.dropout, "selected_epoch": best_epoch}, fit_dir / "mlp.pt")
    settings = {**vars(args), "ridge_alphas": ALPHAS, "selected_ridge_alpha": selected_alpha,
                "selected_mlp_epoch": best_epoch, "partitions": manifest["partitions"],
                "latent_scaler": scaler_stats(latent_scaler), "residual_scaler": scaler_stats(residual_scaler),
                "scaler_fit_partition": "train", "residual_scaler_units": "original Active_Power",
                "mlp_selection": "minimum standardized residual MSE on tuning; no assessment selection",
                "extraction_manifest_sha256": sha256(manifest_path)}
    save_json(fit_dir / "settings.json", settings)
    save_json(fit_dir / "mlp_history.json", history)
    save_json(fit_dir / "ridge_tuning.json", ridge_scores)
    logging.info("Configuration locked; reading assessment block for the first time.")
    assessment = load_partition(artifact_dir, "assessment", manifest)
    assess_z = latent_scaler.transform(assessment["z"]).astype(np.float32)
    corrections = {
        "baseline": np.zeros_like(assessment["y_hat_original"], dtype=np.float64),
        "ridge": residual_scaler.inverse_transform(selected_ridge.predict(assess_z.astype(np.float64))),
        "mlp": residual_scaler.inverse_transform(mlp_predict(model, assess_z, device, args.mlp_batch_size)),
    }
    baseline = assessment["y_hat_original"].astype(np.float64)
    predictions = {name: baseline + residual for name, residual in corrections.items()}
    metrics = {name: forecast_metrics(assessment["y_true_original"], prediction) for name, prediction in predictions.items()}
    improvements, residual_metrics = {}, {}
    for name in predictions:
        np.testing.assert_array_equal(predictions[name], baseline + corrections[name])
        improvements[name] = {"aggregate": deltas(metrics[name]["aggregate"], metrics["baseline"]["aggregate"]),
                              "per_output": {str(h): deltas(metrics[name]["per_output"][str(h)], metrics["baseline"]["per_output"][str(h)]) for h in range(3)}}
        residual_metrics[name] = forecast_metrics(assessment["residual_original"], corrections[name])
    periods = []
    for i, indices in enumerate(np.array_split(np.arange(len(baseline)), 3)):
        periods.append({"period": i, "samples": len(indices),
                        "first_target": str(assessment["target_0_timestamp"][indices[0]]),
                        "last_target": str(assessment["target_2_timestamp"][indices[-1]]),
                        "metrics": {name: forecast_metrics(assessment["y_true_original"][indices], pred[indices]) for name, pred in predictions.items()}})
    diagnostic = diagnostics(assessment, predictions, evaluation_dir)
    rows = []
    for name, result in metrics.items():
        for scope, values in [("aggregate", result["aggregate"]), *[(f"output_{h}", result["per_output"][str(h)]) for h in range(3)]]:
            reference = metrics["baseline"]["aggregate"] if scope == "aggregate" else metrics["baseline"]["per_output"][scope[-1]]
            delta = deltas(values, reference)
            rows.append({"model": name, "scope": scope, **values,
                         "delta_rmse": delta["rmse"]["delta"], "delta_rmse_percent": delta["rmse"]["delta_percent"],
                         "delta_mae": delta["mae"]["delta"], "delta_mae_percent": delta["mae"]["delta_percent"]})
    with (evaluation_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(evaluation_dir / "assessment_predictions.npz",
                        sample_index=assessment["sample_index"],
                        y_true_original=assessment["y_true_original"],
                        **{key: assessment[key] for key in assessment if key.endswith("timestamp")},
                        **{f"prediction_{name}": value for name, value in predictions.items()},
                        **{f"predicted_residual_{name}": value for name, value in corrections.items()})
    warnings = list(WARNINGS)
    if args.max_samples is not None:
        warnings.append("SMOKE TEST: limited samples/epochs; not evidence for the research decision.")
    summary = {
        "experiment": "Stage 0 exploratory classical residual predictability", "assessment_metrics": metrics,
        "deltas_and_improvements": improvements, "residual_prediction_metrics": residual_metrics,
        "selected_ridge_alpha": selected_alpha, "selected_mlp_epoch": best_epoch,
        "partitions": manifest["partitions"], "assessment_chronological_thirds": periods,
        "diagnostics": diagnostic, "runtime_seconds": time.perf_counter() - started,
        "extraction_runtime_seconds": manifest["runtime_seconds"], "device": str(device),
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "sklearn_version": sklearn.__version__, "seed": args.seed, "warnings": warnings,
        "deterministic_algorithms": True,
        "historical_reproduction_reference": manifest.get("historical_reproduction_reference"),
        "regression_checks": {**manifest["regression_checks"], "additive_prediction_identity_exact": True,
                              "original_csv_and_checkpoint_hashes_match_extraction": True},
        "decision": "No automatic GO/NO-GO threshold. Review aggregate, all three outputs and chronological thirds.",
        "evidence": {name: {"aggregate_rmse_lower": metrics[name]["aggregate"]["rmse"] < metrics["baseline"]["aggregate"]["rmse"],
                            "aggregate_mae_lower": metrics[name]["aggregate"]["mae"] < metrics["baseline"]["aggregate"]["mae"],
                            "outputs_with_lower_rmse": sum(metrics[name]["per_output"][str(h)]["rmse"] < metrics["baseline"]["per_output"][str(h)]["rmse"] for h in range(3)),
                            "periods_with_lower_rmse": sum(p["metrics"][name]["aggregate"]["rmse"] < p["metrics"]["baseline"]["aggregate"]["rmse"] for p in periods)} for name in ("ridge", "mlp")},
    }
    save_json(evaluation_dir / "stage0_summary.json", summary)
    lines = ["Stage 0: exploratory residual predictability", "", "Model       RMSE          MAE           R2            nRMSE"]
    for name, result in metrics.items():
        v = result["aggregate"]
        lines.append(f"{name:10s} {v['rmse']:13.6f} {v['mae']:13.6f} {v['r2']:13.6f} {v['nrmse']:13.6f}")
    for name in ("ridge", "mlp"):
        lines.append(f"{name} improvements: {json.dumps(improvements[name]['aggregate'])}")
        lines.append(f"{name} residual RMSE: {residual_metrics[name]['aggregate']['rmse']:.6f}")
        lines.append(f"{name} consistency: {json.dumps(summary['evidence'][name])}")
    lines += [f"Ridge alpha: {selected_alpha}; MLP selected epoch: {best_epoch}",
              f"Partitions: {json.dumps(manifest['partitions'])}",
              f"Runtime: {summary['runtime_seconds']:.2f}s; device: {device}; hardware: {summary['hardware']}",
              "Per-output metrics: metrics.csv; chronological thirds: stage0_summary.json",
              summary["decision"], "", "Warnings:", *warnings]
    (evaluation_dir / "stage0_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Audit complete: %s", evaluation_dir)
    print("\n".join(lines[:6]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-if-missing", action="store_true")
    parser.add_argument("--extraction-batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, help="Cap each block; use isolated _smoke/max_N outputs.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.smoke_test and args.max_samples is None:
        args.max_samples = 1024
    if args.epochs is None:
        args.epochs = 3 if args.max_samples is not None else 100
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_audit(args)


if __name__ == "__main__":
    main()
