"""Stage 0.5: training-only bias and a six-dimensional centered residual model.

Reads completed Stage-0 artifacts only. No extraction, backbone inference or HPO.
"""

import argparse
import csv
import json
import logging
import math
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from experiments.residual_learning.extract_residual_dataset import (
    ROOT, output_paths, save_json, scaler_stats, setup, sha256,
)
from experiments.residual_learning.run_residual_audit import (
    deltas, forecast_metrics, load_partition, mlp_predict,
)
from models.residual_learning.bottleneck_residual_mlp import BottleneckResidualMLP


def fit_bottleneck(train_z: np.ndarray, train_r: np.ndarray, tune_z: np.ndarray,
                   tune_r: np.ndarray, args: argparse.Namespace,
                   device: torch.device) -> tuple[BottleneckResidualMLP, int, list[dict]]:
    """Fit on training, select weights on tuning; no assessment data is accepted."""
    model = BottleneckResidualMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_z), torch.from_numpy(train_r)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite bottleneck training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(features)
        prediction = mlp_predict(model, tune_z, device, args.batch_size)
        tune_loss = float(np.mean((prediction.astype(np.float64) - tune_r) ** 2))
        if not np.isfinite(tune_loss):
            raise RuntimeError("Non-finite bottleneck tuning loss.")
        history.append({"epoch": epoch, "training_mse": total / len(train_z),
                        "tuning_mse": tune_loss, "epoch_seconds": time.perf_counter() - started})
        logging.info("6-D epoch %d train_mse=%.6g tuning_mse=%.6g", epoch, total / len(train_z), tune_loss)
        if tune_loss < best_loss:
            best_loss, best_epoch, stale = tune_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, history


def gain_fraction(improvement: float, baseline: float, stage0_mlp: float) -> dict:
    """Avoid unstable ratios or treating a worsening Stage-0 result as a gain."""
    denominator = baseline - stage0_mlp
    tolerance = float(np.sqrt(np.finfo(np.float64).eps) * max(1.0, abs(baseline), abs(stage0_mlp)))
    valid = denominator > tolerance
    fraction = improvement / denominator if valid else None
    return {"fraction": fraction, "percent": 100 * fraction if valid else None,
            "stage0_mlp_improvement": denominator, "numerical_denominator_tolerance": tolerance,
            "note": None if valid else "Stage-0 MLP gain is non-positive or numerically too small for a stable ratio."}


def check_baseline(recomputed: dict, recorded: dict) -> None:
    """Require like-for-like assessment metrics before comparing stored controls."""
    for scope in ("aggregate", "0", "1", "2"):
        current = recomputed[scope] if scope == "aggregate" else recomputed["per_output"][scope]
        reference = recorded[scope] if scope == "aggregate" else recorded["per_output"][scope]
        for metric in ("rmse", "mae", "r2", "nrmse"):
            if not np.isclose(current[metric], reference[metric], atol=1e-10, rtol=1e-10):
                raise ValueError(f"Stage-0 baseline mismatch for {scope}/{metric}; inputs and summary are not comparable.")


def predict_controls(data: dict, bias: np.ndarray, model: BottleneckResidualMLP,
                     latent_scaler: StandardScaler, residual_scaler: StandardScaler,
                     device: torch.device, batch_size: int) -> tuple[dict, np.ndarray]:
    baseline = data["y_hat_original"].astype(np.float64)
    latent = latent_scaler.transform(data["z"]).astype(np.float32)
    centered = residual_scaler.inverse_transform(mlp_predict(model, latent, device, batch_size)).astype(np.float64)
    predictions = {"baseline": baseline, "bias_only": baseline + bias,
                   "bias_plus_6d_mlp": baseline + bias + centered}
    np.testing.assert_array_equal(predictions["bias_plus_6d_mlp"], baseline + bias + centered)
    return predictions, centered


def run(args: argparse.Namespace) -> None:
    """Lock bias, scalers and checkpoint before reading assessment arrays."""
    if min(args.epochs, args.patience, args.batch_size, args.smoke_samples) < 1:
        raise ValueError("Epochs, patience, batch size and smoke sample cap must be positive.")
    started = time.perf_counter()
    sources = args.stage0_artifacts.resolve()
    stage0_summary_path = args.stage0_evaluation.resolve() / "stage0_summary.json"
    manifest_path = sources / "manifest.json"
    required = [manifest_path, stage0_summary_path, *[sources / f"{name}.npz" for name in ("train", "tuning", "assessment")]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required completed Stage-0 input missing: {path}. Run Stage 0 first; Stage 0.5 never extracts data.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage0 = json.loads(stage0_summary_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["latent_shape"] != [128] or manifest["output_length"] != 3:
        raise ValueError("Expected the verified Stage-0 128-latent/three-output contract.")
    if stage0["partitions"] != manifest["partitions"]:
        raise ValueError("Stage-0 summary and extraction partition provenance differ.")
    source_is_smoke = manifest["max_samples_per_partition"] is not None
    if source_is_smoke and not args.smoke_test:
        raise ValueError("Full Stage 0.5 requires full Stage-0 artifacts, not smoke partitions.")
    for name in ("baseline", "ridge", "mlp"):
        if name not in stage0["assessment_metrics"]:
            raise ValueError(f"Stage-0 summary lacks {name} metrics.")
    saved_periods = stage0["assessment_chronological_thirds"]
    if len(saved_periods) != 3 or sum(p["samples"] for p in saved_periods) != manifest["partitions"]["assessment"]["samples"]:
        raise ValueError("Stage-0 chronological thirds do not cover assessment exactly.")
    artifact_root, evaluation_root = output_paths(None)
    artifact_dir = artifact_root / "stage0_5"
    evaluation_dir = evaluation_root / "stage0_5"
    if args.smoke_test:
        suffix = Path(f"_smoke/stage0_5/max_{args.smoke_samples}/proposed/horizon_15/run_1")
        artifact_dir = ROOT / "artifacts/residual_learning" / suffix
        evaluation_dir = ROOT / "evaluation/residual_learning" / suffix
    for directory in (artifact_dir, evaluation_dir):
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite Stage-0.5 outputs: {directory}")
    source_hashes = {str(path): sha256(path) for path in required if path.name != "assessment.npz"}
    source_hashes[str(sources / "assessment.npz")] = manifest["partitions"]["assessment"]["sha256"]
    device = setup(args.seed, args.cpu_threads)
    train = load_partition(sources, "train", manifest)
    tuning = load_partition(sources, "tuning", manifest)
    # Original-unit subtraction avoids inheriting float32 inverse-scaling roundoff.
    train_residual = train["y_true_original"].astype(np.float64) - train["y_hat_original"].astype(np.float64)
    bias = train_residual.mean(axis=0)
    centered_train = train_residual - bias
    centered_tuning = tuning["y_true_original"].astype(np.float64) - tuning["y_hat_original"].astype(np.float64) - bias
    np.testing.assert_allclose(centered_train.mean(axis=0), 0, atol=1e-10)
    latent_scaler = StandardScaler().fit(train["z"])
    residual_scaler = StandardScaler().fit(centered_train)
    train_z = latent_scaler.transform(train["z"]).astype(np.float32)
    tune_z = latent_scaler.transform(tuning["z"]).astype(np.float32)
    train_r = residual_scaler.transform(centered_train).astype(np.float32)
    tune_r = residual_scaler.transform(centered_tuning).astype(np.float32)
    fit_train_z, fit_train_r, fit_tune_z, fit_tune_r = train_z, train_r, tune_z, tune_r
    if args.smoke_test:
        fit_train_z, fit_train_r = train_z[:args.smoke_samples], train_r[:args.smoke_samples]
        fit_tune_z, fit_tune_r = tune_z[:args.smoke_samples], tune_r[:args.smoke_samples]
    model, epoch, history = fit_bottleneck(fit_train_z, fit_train_r, fit_tune_z, fit_tune_r, args, device)
    with torch.no_grad():
        bottleneck_shape = list(model.projection(torch.from_numpy(fit_train_z[:8]).to(device)).shape)
    if bottleneck_shape != [min(8, len(fit_train_z)), 6]:
        raise RuntimeError(f"Invalid bottleneck shape: {bottleneck_shape}")
    for directory in (artifact_dir, evaluation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    settings = {"bottleneck_dim": 6, "hidden_dim": 16, "dropout": 0.1, "optimizer": "Adam",
                "learning_rate": 1e-3, "weight_decay": 1e-5, "loss": "standardized centered residual MSE",
                "seed": args.seed, "maximum_epochs": args.epochs, "patience": args.patience,
                "batch_size": args.batch_size, "selected_epoch": epoch, "bias": bias.tolist(),
                "bias_fit_partition": "train", "scaler_fit_partition": "train",
                "latent_scaler": scaler_stats(latent_scaler), "centered_residual_scaler": scaler_stats(residual_scaler),
                "partitions": manifest["partitions"], "source_sha256": source_hashes,
                "smoke_test": args.smoke_test, "source_is_smoke": source_is_smoke,
                "model_fit_samples": len(fit_train_z), "model_tuning_samples": len(fit_tune_z),
                "assessment_policy": "Entire saved assessment; exactly the saved chronological thirds"}
    for name, value in (("latent_scaler", latent_scaler), ("centered_residual_scaler", residual_scaler)):
        with (artifact_dir / f"{name}.pkl").open("xb") as handle:
            pickle.dump(value, handle)
    with (artifact_dir / "bias.npy").open("xb") as handle:
        np.save(handle, bias)
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "bias": bias.tolist(), "bottleneck_dim": 6, "hidden_dim": 16, "dropout": 0.1,
                "selected_epoch": epoch}, artifact_dir / "bottleneck_mlp.pt")
    save_json(artifact_dir / "settings.json", settings)
    save_json(evaluation_dir / "training_history.json", history)
    logging.info("Bias, scalers and model locked. Reading complete assessment; no extraction or Stage-0 refitting.")
    assessment = load_partition(sources, "assessment", manifest)
    predictions, centered_predictions = predict_controls(assessment, bias, model, latent_scaler, residual_scaler, device, args.batch_size)
    metrics = {name: forecast_metrics(assessment["y_true_original"], pred) for name, pred in predictions.items()}
    check_baseline(metrics["baseline"], stage0["assessment_metrics"]["baseline"])
    metrics["stage0_ridge"] = stage0["assessment_metrics"]["ridge"]
    metrics["stage0_mlp"] = stage0["assessment_metrics"]["mlp"]
    tuning_predictions, _ = predict_controls(tuning, bias, model, latent_scaler, residual_scaler, device, args.batch_size)
    tuning_metrics = {name: forecast_metrics(tuning["y_true_original"], pred) for name, pred in tuning_predictions.items()}
    improvements, incremental = {}, {}
    for name, values in metrics.items():
        improvements[name] = {"aggregate": deltas(values["aggregate"], metrics["baseline"]["aggregate"]),
                              "per_output": {str(h): deltas(values["per_output"][str(h)], metrics["baseline"]["per_output"][str(h)]) for h in range(3)}}
    incremental = {"aggregate": deltas(metrics["bias_plus_6d_mlp"]["aggregate"], metrics["bias_only"]["aggregate"]),
                   "per_output": {str(h): deltas(metrics["bias_plus_6d_mlp"]["per_output"][str(h)], metrics["bias_only"]["per_output"][str(h)]) for h in range(3)}}
    fractions = {name: {metric: gain_fraction(improvements[name]["aggregate"][metric]["improvement"],
                                             metrics["baseline"]["aggregate"][metric], metrics["stage0_mlp"]["aggregate"][metric])
                       for metric in ("rmse", "mae")} for name in ("bias_only", "bias_plus_6d_mlp")}
    periods, offset = [], 0
    for saved in saved_periods:
        end = offset + saved["samples"]
        if (str(assessment["target_0_timestamp"][offset]) != saved["first_target"]
                or str(assessment["target_2_timestamp"][end - 1]) != saved["last_target"]):
            raise ValueError("Assessment timestamps do not match the saved Stage-0 chronological thirds.")
        period_metrics = {name: forecast_metrics(assessment["y_true_original"][offset:end], pred[offset:end]) for name, pred in predictions.items()}
        check_baseline(period_metrics["baseline"], saved["metrics"]["baseline"])
        periods.append({"period": saved["period"], "samples": saved["samples"],
                        "first_target": saved["first_target"], "last_target": saved["last_target"], "metrics": period_metrics})
        offset = end
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("Stage-0 source changed during Stage 0.5.")
    batch_ratio = ((math.ceil(len(train_z) / args.batch_size) + math.ceil(len(tune_z) / args.batch_size)) /
                   (math.ceil(len(fit_train_z) / args.batch_size) + math.ceil(len(fit_tune_z) / args.batch_size)))
    runtime_estimate = {"mean_smoke_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])),
                        "estimated_full_source_training_seconds_at_100_epochs": float(np.mean([h["epoch_seconds"] for h in history]) * batch_ratio * 100),
                        "note": "Rough batch-count extrapolation on this hardware; excludes loading and startup; source may itself be smoke data."}
    warnings = list(stage0.get("warnings", []))
    if args.smoke_test:
        warnings.append("Stage-0.5 smoke: capped optimizer/tuning samples and epochs; not research evidence. Bias/scalers still fit the entire input training partition; assessment is not sliced.")
    for name in fractions:
        for metric, result in fractions[name].items():
            if result["fraction"] is None:
                warnings.append(f"{name}/{metric}: {result['note']}")
    summary = {"experiment": "Stage 0.5 exploratory bias and six-dimensional centered residual learning",
               "learned_bias_original": bias.tolist(), "assessment_metrics": metrics, "tuning_metrics": tuning_metrics,
               "deltas_and_improvements": improvements, "incremental_6d_over_bias": incremental,
               "bias_fraction_of_stage0_mlp_gain": fractions["bias_only"],
               "six_dimensional_fraction_of_stage0_mlp_gain_retained": fractions["bias_plus_6d_mlp"],
               "assessment_chronological_thirds": periods, "partitions": manifest["partitions"],
               "selected_epoch": epoch, "runtime_seconds": time.perf_counter() - started,
               "runtime_estimate": runtime_estimate, "device": str(device),
               "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
               "torch_version": str(torch.__version__), "sklearn_version": sklearn.__version__,
               "python_version": platform.python_version(), "seed": args.seed, "warnings": warnings,
               "residual_prediction_metrics": forecast_metrics(assessment["y_true_original"].astype(np.float64) - assessment["y_hat_original"].astype(np.float64), bias + centered_predictions),
               "assessment_residual_means": (assessment["y_true_original"].astype(np.float64) - assessment["y_hat_original"].astype(np.float64)).mean(axis=0).tolist(),
               "regression_checks": {"bias_and_scalers_training_only": True, "bottleneck_shape": bottleneck_shape,
                                     "additive_identity_exact": True, "assessment_loaded_after_configuration_lock": True,
                                     "stage0_source_hashes_unchanged": True, "stage0_baseline_and_thirds_match": True},
               "decision": "No automatic GO/NO-GO. Compare bias gain, incremental 6-D gain, retained gain, all outputs and saved thirds."}
    summary["evidence"] = {
        reference: {metric: {"outputs_improved": sum(metrics["bias_plus_6d_mlp"]["per_output"][str(h)][metric] < metrics[reference]["per_output"][str(h)][metric] for h in range(3)),
                             "thirds_improved": sum(p["metrics"]["bias_plus_6d_mlp"]["aggregate"][metric] < p["metrics"][reference]["aggregate"][metric] for p in periods)} for metric in ("rmse", "mae")}
        for reference in ("baseline", "bias_only")}
    order = ["baseline", "bias_only", "stage0_ridge", "stage0_mlp", "bias_plus_6d_mlp"]
    rows = []
    for name in order:
        for scope in ("aggregate", "0", "1", "2"):
            values = metrics[name]["aggregate"] if scope == "aggregate" else metrics[name]["per_output"][scope]
            change = improvements[name]["aggregate"] if scope == "aggregate" else improvements[name]["per_output"][scope]
            rows.append({"model": name, "scope": scope, **values,
                         **{f"{metric}_{key}": change[metric][key] for metric in ("rmse", "mae") for key in ("delta", "delta_percent", "improvement", "improvement_percent")}})
    with (evaluation_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(evaluation_dir / "stage0_5_summary.json", summary)
    lines = ["Stage 0.5: exploratory bias and 6-D residual controls", f"Training bias: {bias.tolist()}",
             "Model                RMSE         MAE          R2           nRMSE"]
    for name in order:
        value = metrics[name]["aggregate"]
        lines.append(f"{name:20s} {value['rmse']:12.6f} {value['mae']:12.6f} {value['r2']:12.6f} {value['nrmse']:12.6f}")
    lines += [f"Bias fraction of Stage-0 MLP gain: {json.dumps(fractions['bias_only'])}",
              f"6-D retained fraction: {json.dumps(fractions['bias_plus_6d_mlp'])}",
              f"Incremental 6-D improvement over bias: {json.dumps(incremental)}",
              f"Per-output changes: {json.dumps({n: improvements[n]['per_output'] for n in ('bias_only', 'bias_plus_6d_mlp')})}",
              f"Chronological thirds RMSE/MAE: {json.dumps([{**{k: p[k] for k in ('period', 'samples', 'first_target', 'last_target')}, 'metrics': {n: {k: p['metrics'][n]['aggregate'][k] for k in ('rmse', 'mae')} for n in predictions}} for p in periods])}",
              f"Evidence: {json.dumps(summary['evidence'])}", f"Selected epoch: {epoch}; runtime: {summary['runtime_seconds']:.2f}s; device: {device}",
              summary["decision"], "Warnings:", *warnings]
    with (evaluation_dir / "stage0_5_summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    logging.info("Stage 0.5 complete: %s", evaluation_dir)
    print("\n".join(lines[:8]))


def main() -> None:
    artifacts, evaluation = output_paths(None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-artifacts", type=Path, default=artifacts, help="Existing completed Stage-0 artifact directory.")
    parser.add_argument("--stage0-evaluation", type=Path, default=evaluation, help="Existing Stage-0 summary directory.")
    parser.add_argument("--smoke-test", action="store_true", help="Cap model fitting/tuning; keep bias/scalers and assessment complete.")
    parser.add_argument("--smoke-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (3 if args.smoke_test else 100)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
