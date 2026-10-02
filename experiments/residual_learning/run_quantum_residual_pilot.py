"""Isolated ideal six-qubit pilot; reads completed Stage 0/0.5, never extracts data."""

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
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.residual_learning.extract_residual_dataset import (
    ROOT, output_paths, save_json, setup, sha256, require_new_files, verify_recorded_file,
    compare_scaler_stats, scaler_stats, RECOVERY_BIAS_ATOL,
)
from experiments.residual_learning.run_residual_audit import (
    deltas, forecast_metrics, load_partition, mlp_predict,
)
from experiments.residual_learning.run_stage0_5 import check_baseline
from models.residual_learning.quantum_residual_vqc import QuantumResidualVQC


def synchronize(device: torch.device) -> None:
    """Make CUDA timing include completed simulator work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def verify_model(model: QuantumResidualVQC, features: np.ndarray,
                 device: torch.device) -> dict:
    """Check broadcasting, bounded angles and finite gradient paths."""
    batch = torch.from_numpy(features[:min(4, len(features))]).to(device)
    model.eval()
    angles = math.pi * torch.tanh(model.projection(batch))
    quantum = model.quantum_features(batch)
    output = model(batch)
    assert list(angles.shape) == [len(batch), 6]
    assert list(quantum.shape) == [len(batch), 6]
    assert list(output.shape) == [len(batch), 3]
    assert torch.isfinite(output).all() and (angles.abs() <= math.pi).all()
    output.square().mean().backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing/nonfinite gradient: {name}")
        gradients[name] = float(parameter.grad.norm().item())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        serial = torch.cat([model(row[None]) for row in batch])
        torch.testing.assert_close(output.detach(), serial, rtol=1e-5, atol=1e-6)
    counts = model.parameter_counts()
    assert counts == {"projection": 774, "quantum": 24, "readout": 21, "total": 819}
    return {"input_shape": list(batch.shape), "angle_shape": list(angles.shape),
            "quantum_output_shape": list(quantum.shape), "residual_shape": list(output.shape),
            "gradient_norms": gradients, "broadcast_matches_serial": True,
            "parameter_counts": counts}


def fit(model: QuantumResidualVQC, train_z: np.ndarray, train_r: np.ndarray,
        tune_z: np.ndarray, tune_r: np.ndarray, args: argparse.Namespace,
        device: torch.device) -> tuple[int, list[dict]]:
    """Optimize projection/circuit/readout; select only by tuning MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_z), torch.from_numpy(train_r)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        synchronize(device)
        started = time.perf_counter()
        total, forward_seconds, backward_seconds = 0.0, 0.0, 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            tick = time.perf_counter()
            loss = torch.nn.functional.mse_loss(model(features), targets)
            synchronize(device)
            forward_seconds += time.perf_counter() - tick
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite quantum training loss.")
            tick = time.perf_counter()
            loss.backward()
            optimizer.step()
            synchronize(device)
            backward_seconds += time.perf_counter() - tick
            total += loss.item() * len(features)
        train_seconds = time.perf_counter() - started
        tick = time.perf_counter()
        prediction = mlp_predict(model, tune_z, device, args.batch_size)
        synchronize(device)
        tune_seconds = time.perf_counter() - tick
        tuning_loss = float(np.mean((prediction.astype(np.float64) - tune_r) ** 2))
        if not np.isfinite(tuning_loss):
            raise RuntimeError("Nonfinite quantum tuning loss.")
        history.append({"epoch": epoch, "train_mse": total / len(train_z), "tuning_mse": tuning_loss,
                        "forward_seconds": forward_seconds, "backward_and_optimizer_seconds": backward_seconds,
                        "training_seconds": train_seconds, "tuning_seconds": tune_seconds,
                        "epoch_seconds": train_seconds + tune_seconds})
        logging.info("Quantum epoch %d train=%.6g tuning=%.6g time=%.2fs", epoch,
                     total / len(train_z), tuning_loss, train_seconds + tune_seconds)
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
    return best_epoch, history


def run(args: argparse.Namespace) -> None:
    """Validate provenance, lock a pilot, then optionally assess it once."""
    if min(args.batch_size, args.epochs, args.patience, args.subset_samples) < 1 or args.learning_rate <= 0:
        raise ValueError("Batch size, epochs, patience, subset size and learning rate must be positive.")
    started = time.perf_counter()
    source, classical = args.stage0_artifacts.resolve(), args.stage0_5_artifacts.resolve()
    manifest_path = source / "manifest.json"
    artifacts, evaluation = output_paths(None)
    if args.smoke_test or args.benchmark:
        suffix = Path("_smoke/quantum_pilot")
        if args.benchmark:
            suffix = suffix / "_benchmark"
        suffix = suffix / f"max_{args.subset_samples}/proposed/horizon_15/run_1"
        if args.benchmark:
            suffix = suffix / args.device
        artifacts = ROOT / "artifacts/residual_learning" / suffix
        evaluation = ROOT / "evaluation/residual_learning" / suffix
    else:
        artifacts, evaluation = artifacts / "quantum_pilot", evaluation / "quantum_pilot"
    summary_path = evaluation / ("benchmark.json" if args.benchmark else "quantum_pilot_summary.json")
    if summary_path.is_file():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if not args.benchmark and (not completed["assessment_evaluated"] or "quantum_6q_angle" not in completed["assessment_metrics"]):
            raise ValueError("Existing quantum summary is not a completed assessment.")
        if manifest_path.is_file() and completed["partitions"] != json.loads(manifest_path.read_text(encoding="utf-8"))["partitions"]:
            raise ValueError("Completed quantum pilot and current extraction provenance differ.")
        for path in [manifest_path, args.stage0_evaluation / "stage0_summary.json",
                     args.stage0_5_evaluation / "stage0_5_summary.json", classical / "settings.json", classical / "bias.npy",
                     classical / "latent_scaler.pkl", classical / "centered_residual_scaler.pkl",
                     *[source / f"{name}.npz" for name in ("train", "tuning", "assessment")]]:
            if path.is_file():
                verify_recorded_file(path, completed["source_sha256"])
        logging.info("Existing quantum pilot result found; no retraining, copied scalers or checkpoint required: %s", summary_path)
        return
    summary0_path = args.stage0_evaluation.resolve() / "stage0_summary.json"
    summary5_path = args.stage0_5_evaluation.resolve() / "stage0_5_summary.json"
    required = [manifest_path, summary0_path, summary5_path, classical / "settings.json",
                classical / "bias.npy", classical / "latent_scaler.pkl",
                classical / "centered_residual_scaler.pkl", source / "train.npz", source / "tuning.npz"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required pilot input missing: {path}. Run Stage 0 and Stage 0.5 first; this script never regenerates them.")
    manifest = json.loads(manifest_path.read_text())
    stage0 = json.loads(summary0_path.read_text())
    stage5 = json.loads(summary5_path.read_text())
    settings5 = json.loads((classical / "settings.json").read_text())
    if manifest["latent_shape"] != [128] or manifest["output_length"] != 3:
        raise ValueError("Expected Stage-0 [N,128] -> [N,3] residual contract.")
    for record in (stage0, stage5, settings5):
        if record["partitions"] != manifest["partitions"]:
            raise ValueError("Stage-0/0.5 partition provenance differs.")
    limited = args.smoke_test or args.benchmark
    if not limited and (manifest["max_samples_per_partition"] is not None or settings5["smoke_test"]):
        raise ValueError("Full quantum pilot requires full Stage-0 and Stage-0.5 artifacts.")
    if not args.benchmark and not (source / "assessment.npz").is_file():
        raise FileNotFoundError("Stage-0 assessment.npz missing; run Stage 0 first.")
    require_new_files([artifacts / name for name in ("best_checkpoint.pt", "latent_scaler.pkl", "centered_residual_scaler.pkl", "settings.json")]
                      + [evaluation / name for name in ("benchmark.json", "benchmark.txt", "quantum_pilot_summary.json", "quantum_pilot_summary.txt", "metrics.csv", "training_history.json")])
    source_hashes = {str(path): sha256(path) for path in required}
    setup(args.seed, args.cpu_threads)
    device_warning = "This pilot's default.qubit implementation is CPU-only; --device cuda falls back to CPU for the entire residual model."
    if args.device == "cuda":
        logging.warning(device_warning)
    device = torch.device("cpu")
    train = load_partition(source, "train", manifest)
    tuning = load_partition(source, "tuning", manifest)
    bias = np.load(classical / "bias.npy", allow_pickle=False)
    train_residual = train["y_true_original"].astype(np.float64) - train["y_hat_original"].astype(np.float64)
    np.testing.assert_allclose(bias, train_residual.mean(0), atol=RECOVERY_BIAS_ATOL, rtol=0)
    np.testing.assert_array_equal(bias, stage5["learned_bias_original"])
    with (classical / "latent_scaler.pkl").open("rb") as handle:
        latent_scaler = pickle.load(handle)
    with (classical / "centered_residual_scaler.pkl").open("rb") as handle:
        residual_scaler = pickle.load(handle)
    from sklearn.preprocessing import StandardScaler
    for saved, expected in ((latent_scaler, StandardScaler().fit(train["z"])),
                            (residual_scaler, StandardScaler().fit(train_residual - bias))):
        compare_scaler_stats(scaler_stats(saved), scaler_stats(expected))
    train_z = latent_scaler.transform(train["z"]).astype(np.float32)
    tune_z = latent_scaler.transform(tuning["z"]).astype(np.float32)
    train_r = residual_scaler.transform(train_residual - bias).astype(np.float32)
    tune_r = residual_scaler.transform(tuning["y_true_original"].astype(np.float64) - tuning["y_hat_original"].astype(np.float64) - bias).astype(np.float32)
    if limited:
        train_z, train_r = train_z[:args.subset_samples], train_r[:args.subset_samples]
        tune_z, tune_r = tune_z[:min(args.subset_samples, 1024)], tune_r[:min(args.subset_samples, 1024)]
    model = QuantumResidualVQC().to(device)
    import pennylane as qml
    checks = verify_model(model, train_z, device)
    logging.info("default.qubit / backprop / shots=None / %s; parameters=%s", device, checks["parameter_counts"])
    epoch, history = fit(model, train_z, train_r, tune_z, tune_r, args, device)
    for directory in (artifacts, evaluation):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts / "best_checkpoint.pt"
    torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "selected_epoch": epoch, "bias": bias.tolist()}, checkpoint)
    restored = QuantumResidualVQC().to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    restored.eval()
    probe = torch.from_numpy(train_z[:4]).to(device)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    checks["checkpoint_reload_exact"] = True
    for name, scaler in (("latent_scaler", latent_scaler), ("centered_residual_scaler", residual_scaler)):
        with (artifacts / f"{name}.pkl").open("xb") as handle:
            pickle.dump(scaler, handle)
    train_batches = math.ceil(len(train_z) / args.batch_size)
    tune_batches = math.ceil(len(tune_z) / args.batch_size)
    mean_train = float(np.mean([h["training_seconds"] for h in history]))
    mean_tune = float(np.mean([h["tuning_seconds"] for h in history]))
    estimated_epoch = (mean_train * math.ceil(manifest["partitions"]["train"]["samples"] / args.batch_size) / train_batches
                       + mean_tune * math.ceil(manifest["partitions"]["tuning"]["samples"] / args.batch_size) / tune_batches)
    report = {"backend": "default.qubit", "interface": "torch", "differentiation": "backprop", "shots": None,
              "device": str(device), "requested_device": args.device, "hardware": platform.processor(),
              "pennylane_version": qml.__version__, "torch_version": str(torch.__version__), "python_version": platform.python_version(),
              "configuration": vars(args) | {"stage0_artifacts": str(source), "stage0_evaluation": str(args.stage0_evaluation),
                                              "stage0_5_artifacts": str(classical), "stage0_5_evaluation": str(args.stage0_5_evaluation)},
              "train_samples": len(train_z), "tuning_samples": len(tune_z), "partitions": manifest["partitions"],
              "selected_epoch": epoch, "training_bias_original": bias.tolist(), "source_sha256": source_hashes,
              "scaling": "Reused Stage-0.5 train-only latent and centered-residual scalers, verified against entire training partition",
              "encoding": "One RY encoding per qubit; angles=pi*tanh(trainable Linear(128,6))",
              "variational_layer": "RZ(theta_z), RY(theta_y), CNOT ring; repeated twice",
              "projection_policy": "Trained from scratch; same bottleneck width, not the same learned coordinates as Stage 0.5",
              "checks": checks, "history": history, "estimated_source_epoch_seconds": estimated_epoch,
              "estimated_source_30_epoch_seconds": estimated_epoch * 30,
              "estimate_note": "Batch-count extrapolation; excludes loading/assessment; source may be smoke; not a measured full run",
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "assessment_evaluated": False, "warnings": list(stage0.get("warnings", [])) + [
                  *([device_warning] if args.device == "cuda" else []),
                  "One independently trained 128->6 projection; not the exact Stage-0.5 compressed coordinates.",
                  *(["Smoke/benchmark uses limited data and is not scientific assessment evidence."] if limited else []),
              ],
              "decision": "No automatic GO/NO-GO or superiority claim; a single seed cannot establish reproducibility."}
    save_json(artifacts / "settings.json", report["configuration"] | {
        "source_sha256": dict(source_hashes), "scaling": report["scaling"],
        "bias": bias.tolist(), "selected_epoch": epoch, "parameter_counts": model.parameter_counts(),
        "backend": "default.qubit", "differentiation": "backprop", "shots": None,
    })
    if not args.benchmark:
        logging.info("Configuration/checkpoint locked; now reading assessment once.")
        assessment = load_partition(source, "assessment", manifest)
        source_hashes[str(source / "assessment.npz")] = sha256(source / "assessment.npz")
        centered = residual_scaler.inverse_transform(mlp_predict(model, latent_scaler.transform(assessment["z"]).astype(np.float32), device, args.batch_size)).astype(np.float64)
        baseline = assessment["y_hat_original"].astype(np.float64)
        corrected = baseline + bias + centered
        np.testing.assert_array_equal(corrected, baseline + bias + centered)
        metrics = {"baseline": forecast_metrics(assessment["y_true_original"], baseline),
                   "bias_only": forecast_metrics(assessment["y_true_original"], baseline + bias),
                   "quantum_6q_angle": forecast_metrics(assessment["y_true_original"], corrected),
                   "stage0_5_6d_mlp": stage5["assessment_metrics"]["bias_plus_6d_mlp"],
                   "stage0_ridge": stage0["assessment_metrics"]["ridge"],
                   "stage0_mlp": stage0["assessment_metrics"]["mlp"]}
        check_baseline(metrics["baseline"], stage0["assessment_metrics"]["baseline"])
        check_baseline(metrics["bias_only"], stage5["assessment_metrics"]["bias_only"])
        periods, offset = [], 0
        saved5 = stage5["assessment_chronological_thirds"]
        saved0 = stage0["assessment_chronological_thirds"]
        if len(saved0) != 3 or len(saved5) != 3:
            raise ValueError("Expected the three saved assessment thirds.")
        for old, classical_period in zip(saved0, saved5):
            end = offset + old["samples"]
            for key in ("period", "samples", "first_target", "last_target"):
                if old[key] != classical_period[key]:
                    raise ValueError("Stage-0/0.5 chronological thirds differ.")
            if str(assessment["target_0_timestamp"][offset]) != old["first_target"] or str(assessment["target_2_timestamp"][end-1]) != old["last_target"]:
                raise ValueError("Assessment timestamp mapping differs from saved thirds.")
            period_metrics = {name: forecast_metrics(assessment["y_true_original"][offset:end], pred[offset:end])
                              for name, pred in (("baseline", baseline), ("bias_only", baseline+bias), ("quantum_6q_angle", corrected))}
            check_baseline(period_metrics["baseline"], old["metrics"]["baseline"])
            period_metrics["stage0_5_6d_mlp"] = classical_period["metrics"]["bias_plus_6d_mlp"]
            periods.append({key: old[key] for key in ("period", "samples", "first_target", "last_target")} | {"metrics": period_metrics})
            offset = end
        assert offset == len(baseline)
        report.update({"assessment_evaluated": True, "assessment_metrics": metrics,
                       "assessment_chronological_thirds": periods,
                       "quantum_comparisons": {name: {"aggregate": deltas(metrics["quantum_6q_angle"]["aggregate"], values["aggregate"]),
                            "per_output": {str(h): deltas(metrics["quantum_6q_angle"]["per_output"][str(h)], values["per_output"][str(h)]) for h in range(3)}}
                                               for name, values in metrics.items() if name != "quantum_6q_angle"},
                       "residual_prediction_metrics": forecast_metrics(assessment["y_true_original"].astype(np.float64)-baseline, bias+centered)})
        checks["additive_identity_exact"] = True
        rows = [{"model": name, "scope": scope, **(values["aggregate"] if scope == "aggregate" else values["per_output"][scope])}
                for name, values in metrics.items() for scope in ("aggregate", "0", "1", "2")]
        with (evaluation / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("Stage-0/0.5 inputs changed during pilot.")
    checks["source_hashes_unchanged"] = True
    report["runtime_seconds"] = time.perf_counter() - started
    save_json(evaluation / "training_history.json", history)
    filename = "benchmark" if args.benchmark else "quantum_pilot_summary"
    save_json(evaluation / f"{filename}.json", report)
    with (evaluation / f"{filename}.txt").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logging.info("Finished %s on %s: %.2fs total; extrapolated source epoch %.2fs; outputs %s",
                 filename, device, report["runtime_seconds"], estimated_epoch, evaluation)


def main() -> None:
    artifacts, evaluation = output_paths(None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-artifacts", type=Path, default=artifacts)
    parser.add_argument("--stage0-evaluation", type=Path, default=evaluation)
    parser.add_argument("--stage0-5-artifacts", type=Path, default=artifacts / "stage0_5")
    parser.add_argument("--stage0-5-evaluation", type=Path, default=evaluation / "stage0_5")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--benchmark", action="store_true")
    parser.add_argument("--subset-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="CPU-only default.qubit pilot; cuda warns and falls back to CPU; auto also uses CPU.")
    args = parser.parse_args()
    args.subset_samples = args.subset_samples if args.subset_samples is not None else (32 if args.smoke_test else 4096)
    args.epochs = args.epochs if args.epochs is not None else (1 if args.smoke_test else 2 if args.benchmark else 30)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
