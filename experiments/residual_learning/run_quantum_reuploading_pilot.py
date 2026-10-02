"""Isolated CPU re-uploading pilot; no extraction or earlier-model retraining.

Phase A verified current saved comparators internally and independently.
Earlier quoted values cannot be reproduced from available repository history;
their provenance remains unresolved and they are excluded from comparisons.
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
import torch
from sklearn.preprocessing import StandardScaler

from experiments.residual_learning.extract_residual_dataset import (
    ROOT, output_paths, save_json, setup, sha256, require_new_files, verify_recorded_file,
    compare_scaler_stats, scaler_stats, RECOVERY_BIAS_ATOL,
)
from experiments.residual_learning.run_residual_audit import (
    deltas, forecast_metrics, load_partition, mlp_predict,
)
from experiments.residual_learning.run_stage0_5 import check_baseline
from experiments.residual_learning.run_quantum_residual_pilot import fit, verify_model
from models.residual_learning.quantum_residual_vqc import QuantumResidualVQC
from models.residual_learning.quantum_residual_reupload_vqc import QuantumResidualReuploadVQC

HISTORICAL_NOTE = (
    "Current full comparator artifacts were internally consistent and independently verified in Phase A. "
    "Earlier quoted values cannot be reproduced from available repository history; their provenance "
    "remains unresolved. They must not be used in further comparisons. No rerun explanation is claimed."
)


def verify_snapshot(path: Path, recorded: dict[str, str]) -> None:
    """Accept only the recorded snapshot, allowing Git's JSON newline conversion."""
    verify_recorded_file(path, recorded)


def verify_reuploading(model: QuantumResidualReuploadVQC, features: np.ndarray) -> dict:
    """Verify encoding placement, initialization fairness and observable sensitivity."""
    checks = verify_model(model, features, torch.device("cpu"))
    batch = torch.from_numpy(features[:3])
    with torch.no_grad():
        angles = math.pi * torch.tanh(model.projection(batch))
    tape = model.circuit.construct((angles, model.weights), {})
    expected = []
    for _ in range(2):
        expected += [("RY", [wire]) for wire in range(6)]
        expected += [(name, [wire]) for wire in range(6) for name in ("RZ", "RY")]
        expected += [("CNOT", [wire, (wire + 1) % 6]) for wire in range(6)]
    if [(op.name, list(op.wires)) for op in tape.operations] != expected:
        raise RuntimeError("Circuit must contain exactly two Encode/RZ/RY/ring blocks.")
    for start in (0, 24):
        for wire in range(6):
            torch.testing.assert_close(tape.operations[start + wire].data[0], angles[:, wire], rtol=0, atol=0)
    sensitivity = torch.zeros(24, dtype=torch.float64)
    # Separate generator preserves the first pilot's training RNG sequence.
    generator = torch.Generator().manual_seed(42)
    for _ in range(3):
        probe_angles = (2 * torch.rand(3, 6, generator=generator, dtype=torch.float64) - 1) * math.pi
        probe_weights = (2 * torch.rand(2, 6, 2, generator=generator, dtype=torch.float64) - 1) * 0.1
        jacobian = torch.autograd.functional.jacobian(
            lambda weights: torch.stack(model.circuit(probe_angles, weights), dim=-1), probe_weights)
        if not torch.isfinite(jacobian).all():
            raise RuntimeError("Nonfinite quantum observable Jacobian.")
        sensitivity = torch.maximum(sensitivity, jacobian.reshape(18, 24).abs().amax(0))
    if not torch.all(sensitivity > 1e-10):
        raise RuntimeError("A quantum parameter has zero observable sensitivity across smoke probes.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = QuantumResidualVQC()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    checks.update({"encoding_applications": 2, "input_encoding_rotations": 12,
                   "encoding_angles_reused_exactly": True, "initialization_matches_first_pilot": True,
                   "all_24_observable_jacobian_columns_nonzero": True,
                   "max_absolute_sensitivity_per_parameter": sensitivity.tolist()})
    return checks


def run(args: argparse.Namespace) -> None:
    """Train from scratch, lock checkpoint, then read assessment once."""
    started = time.perf_counter()
    if args.smoke_samples < 3:
        raise ValueError("Smoke verification needs at least three fitting samples.")
    source = args.stage0_artifacts.resolve()
    classical = args.stage0_5_artifacts.resolve()
    artifacts, evaluation = output_paths(None)
    artifacts, evaluation = artifacts / "quantum_reuploading_pilot", evaluation / "quantum_reuploading_pilot"
    if args.smoke_test:
        suffix = Path(f"_smoke/quantum_reuploading_pilot/max_{args.smoke_samples}/proposed/horizon_15/run_1")
        artifacts, evaluation = ROOT / "artifacts/residual_learning" / suffix, ROOT / "evaluation/residual_learning" / suffix
    summary_path = evaluation / "quantum_reuploading_summary.json"
    if summary_path.is_file():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if "quantum_6q_reuploading" not in completed["assessment_metrics"]:
            raise ValueError("Existing re-uploading summary is not a completed assessment.")
        manifest_path = source / "manifest.json"
        if manifest_path.is_file() and completed["partitions"] != json.loads(manifest_path.read_text(encoding="utf-8"))["partitions"]:
            raise ValueError("Completed re-uploading pilot and current extraction provenance differ.")
        for path in [manifest_path, args.stage0_evaluation / "stage0_summary.json",
                     args.stage0_5_evaluation / "stage0_5_summary.json", args.quantum_pilot_evaluation / "quantum_pilot_summary.json",
                     classical / "bias.npy", classical / "latent_scaler.pkl", classical / "centered_residual_scaler.pkl",
                     *[source / f"{name}.npz" for name in ("train", "tuning", "assessment")]]:
            if path.is_file():
                verify_recorded_file(path, completed["source_sha256"])
        logging.info("Existing re-uploading assessment found; no automatic retraining or checkpoint required: %s", summary_path)
        return
    paths = {"manifest": source / "manifest.json",
             "stage0": args.stage0_evaluation.resolve() / "stage0_summary.json",
             "stage5": args.stage0_5_evaluation.resolve() / "stage0_5_summary.json",
             "quantum": args.quantum_pilot_evaluation.resolve() / "quantum_pilot_summary.json"}
    required = [*paths.values(), classical / "bias.npy",
                classical / "latent_scaler.pkl", classical / "centered_residual_scaler.pkl",
                *[source / f"{name}.npz" for name in ("train", "tuning", "assessment")]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required existing experiment input missing: {path}. Complete Stage 0/0.5 and the first quantum pilot first; this script never regenerates them.")
    documents = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    manifest, stage0, stage5, quantum = [documents[name] for name in ("manifest", "stage0", "stage5", "quantum")]
    if manifest["schema_version"] != 1 or manifest["latent_shape"] != [128] or manifest["output_length"] != 3:
        raise ValueError("Expected the verified Stage-0 [N,128] / three-output contract.")
    for record in (stage0, stage5, quantum):
        if record["partitions"] != manifest["partitions"]:
            raise ValueError("Comparator partition provenance differs.")
    if not args.smoke_test and (manifest["max_samples_per_partition"] is not None
                               or quantum["configuration"]["smoke_test"]):
        raise ValueError("Full re-uploading pilot requires full, completed comparator artifacts.")
    if not quantum["assessment_evaluated"] or quantum["backend"] != "default.qubit" or quantum["device"] != "cpu":
        raise ValueError("Expected a completed CPU default.qubit single-encoding comparator.")
    for key, expected in (("seed", 42), ("learning_rate", 1e-3), ("batch_size", 128), ("patience", 8)):
        if quantum["configuration"][key] != expected:
            raise ValueError(f"First pilot's {key} differs from the fixed re-uploading protocol.")
    if quantum["differentiation"] != "backprop" or quantum["shots"] is not None or quantum["pennylane_version"] != "0.45.1":
        raise ValueError("First pilot must use PennyLane 0.45.1, backprop and analytic expectations.")
    for name in ("manifest", "stage0", "stage5"):
        verify_snapshot(paths[name], quantum["source_sha256"])
    for name in ("bias.npy", "latent_scaler.pkl", "centered_residual_scaler.pkl"):
        verify_snapshot(classical / name, quantum["source_sha256"])
    metrics = {"baseline": stage0["assessment_metrics"]["baseline"],
               "bias_only": stage5["assessment_metrics"]["bias_only"],
               "stage0_ridge": stage0["assessment_metrics"]["ridge"],
               "stage0_mlp": stage0["assessment_metrics"]["mlp"],
               "stage0_5_6d_mlp": stage5["assessment_metrics"]["bias_plus_6d_mlp"],
               "quantum_6q_single_encoding": quantum["assessment_metrics"]["quantum_6q_angle"]}
    for name in ("baseline", "bias_only", "stage0_ridge", "stage0_mlp", "stage0_5_6d_mlp"):
        if metrics[name] != quantum["assessment_metrics"][name]:
            raise ValueError(f"Current authoritative {name} differs from the first pilot's comparator snapshot.")
    require_new_files([artifacts / name for name in ("best_checkpoint.pt", "latent_scaler.pkl", "centered_residual_scaler.pkl", "settings.json")]
                      + [evaluation / name for name in ("quantum_reuploading_summary.json", "quantum_reuploading_summary.txt", "metrics.csv", "training_history.json", "assessment_predictions.npz")])
    source_hashes = {str(path): sha256(path) for path in required if path.name != "assessment.npz"}
    device = torch.device("cpu")
    setup(42, 1)
    train = load_partition(source, "train", manifest)
    tuning = load_partition(source, "tuning", manifest)
    bias = np.load(classical / "bias.npy", allow_pickle=False)
    train_residual = train["y_true_original"].astype(np.float64) - train["y_hat_original"].astype(np.float64)
    np.testing.assert_allclose(bias, train_residual.mean(0), atol=RECOVERY_BIAS_ATOL, rtol=0)
    np.testing.assert_array_equal(bias, stage5["learned_bias_original"])
    np.testing.assert_array_equal(bias, quantum["training_bias_original"])
    scaler_bytes = {name: (classical / f"{name}.pkl").read_bytes()
                    for name in ("latent_scaler", "centered_residual_scaler")}
    latent_scaler = pickle.loads(scaler_bytes["latent_scaler"])
    residual_scaler = pickle.loads(scaler_bytes["centered_residual_scaler"])
    for saved, expected in ((latent_scaler, StandardScaler().fit(train["z"])),
                            (residual_scaler, StandardScaler().fit(train_residual - bias))):
        compare_scaler_stats(scaler_stats(saved), scaler_stats(expected))
    train_z = latent_scaler.transform(train["z"]).astype(np.float32)
    tune_z = latent_scaler.transform(tuning["z"]).astype(np.float32)
    train_r = residual_scaler.transform(train_residual - bias).astype(np.float32)
    tune_r = residual_scaler.transform(tuning["y_true_original"].astype(np.float64) - tuning["y_hat_original"].astype(np.float64) - bias).astype(np.float32)
    if args.smoke_test:
        train_z, train_r = train_z[:args.smoke_samples], train_r[:args.smoke_samples]
        tune_z, tune_r = tune_z[:args.smoke_samples], tune_r[:args.smoke_samples]
    import pennylane as qml
    if qml.__version__ != "0.45.1":
        raise RuntimeError("This controlled pilot requires PennyLane 0.45.1; install pennylane==0.45.1.")
    model = QuantumResidualReuploadVQC()
    checks = verify_reuploading(model, train_z) if args.smoke_test else verify_model(model, train_z, device)
    logging.info("CPU default.qubit / backprop / shots=None; two encodings; parameters=%s", model.parameter_counts())
    epoch, history = fit(model, train_z, train_r, tune_z, tune_r, args, device)
    settings = {"seed": 42, "optimizer": "Adam", "learning_rate": 1e-3, "batch_size": 128,
                "maximum_epochs": args.epochs, "patience": 8, "loss": "standardized centered-residual MSE",
                "device": "cpu", "cpu_threads": 1, "backend": "default.qubit", "interface": "torch",
                "differentiation": "backprop", "shots": None, "pennylane_version": qml.__version__,
                "encoding_applications": 2, "parameter_counts": model.parameter_counts(),
                "selected_epoch": epoch, "bias": bias.tolist(), "partitions": manifest["partitions"],
                "source_sha256": dict(source_hashes), "scaling": "Stage-0.5 scalers; exact historical hashes or validated numerical recovery; verified against training only",
                "historical_discrepancy": HISTORICAL_NOTE, "smoke_test": args.smoke_test}
    for directory in (artifacts, evaluation):
        directory.mkdir(parents=True, exist_ok=True)
    for name, raw in scaler_bytes.items():
        with (artifacts / f"{name}.pkl").open("xb") as handle:
            handle.write(raw)
    checkpoint = artifacts / "best_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "selected_epoch": epoch, "encoding_applications": 2}, checkpoint)
    restored = QuantumResidualReuploadVQC()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        probe = torch.from_numpy(train_z[:4])
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    checks["checkpoint_reload_exact"] = True
    save_json(artifacts / "settings.json", settings)
    logging.info("Checkpoint/configuration locked; now reading assessment once.")
    assessment = load_partition(source, "assessment", manifest)
    source_hashes[str(source / "assessment.npz")] = sha256(source / "assessment.npz")
    baseline = assessment["y_hat_original"].astype(np.float64)
    centered = residual_scaler.inverse_transform(mlp_predict(model, latent_scaler.transform(assessment["z"]).astype(np.float32), device, 128)).astype(np.float64)
    corrected = baseline + bias + centered
    np.testing.assert_array_equal(corrected, baseline + bias + centered)
    checks["additive_identity_exact"] = True
    check_baseline(forecast_metrics(assessment["y_true_original"], baseline), metrics["baseline"])
    check_baseline(forecast_metrics(assessment["y_true_original"], baseline + bias), metrics["bias_only"])
    metrics["quantum_6q_reuploading"] = forecast_metrics(assessment["y_true_original"], corrected)
    periods, offset = [], 0
    saved_periods = [record["assessment_chronological_thirds"] for record in (stage0, stage5, quantum)]
    if any(len(value) != 3 for value in saved_periods):
        raise ValueError("Expected exactly the three existing assessment thirds.")
    for old, classical_period, first_period in zip(*saved_periods):
        for key in ("period", "samples", "first_target", "last_target"):
            if not old[key] == classical_period[key] == first_period[key]:
                raise ValueError("Existing chronological thirds differ.")
        end = offset + old["samples"]
        if str(assessment["target_0_timestamp"][offset]) != old["first_target"] or str(assessment["target_2_timestamp"][end-1]) != old["last_target"]:
            raise ValueError("Assessment timestamps differ from existing thirds.")
        check_baseline(forecast_metrics(assessment["y_true_original"][offset:end], baseline[offset:end]), old["metrics"]["baseline"])
        period_metrics = {"baseline": old["metrics"]["baseline"], "bias_only": classical_period["metrics"]["bias_only"],
                          "quantum_6q_single_encoding": first_period["metrics"]["quantum_6q_angle"],
                          "stage0_5_6d_mlp": classical_period["metrics"]["bias_plus_6d_mlp"],
                          "quantum_6q_reuploading": forecast_metrics(assessment["y_true_original"][offset:end], corrected[offset:end])}
        periods.append({key: old[key] for key in ("period", "samples", "first_target", "last_target")} | {"metrics": period_metrics})
        offset = end
    if offset != len(baseline):
        raise ValueError("Saved thirds do not cover the assessment partition.")
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("Protected input changed during re-uploading pilot.")
    checks["protected_sources_unchanged"] = True
    estimates = {"mean_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])),
                 "estimated_source_epoch_seconds": float(np.mean([h["training_seconds"] for h in history])) * math.ceil(manifest["partitions"]["train"]["samples"] / 128) / math.ceil(len(train_z) / 128)
                    + float(np.mean([h["tuning_seconds"] for h in history])) * math.ceil(manifest["partitions"]["tuning"]["samples"] / 128) / math.ceil(len(tune_z) / 128),
                 "note": "Rough batch-count extrapolation; smoke sources may themselves be limited; excludes startup/loading/assessment"}
    summary = {"experiment": "Two-encoding six-qubit residual pilot", "assessment_metrics": metrics,
               "assessment_chronological_thirds": periods, "partitions": manifest["partitions"],
               "reuploading_comparisons": {name: {"aggregate": deltas(metrics["quantum_6q_reuploading"]["aggregate"], values["aggregate"]),
                    "per_output": {str(h): deltas(metrics["quantum_6q_reuploading"]["per_output"][str(h)], values["per_output"][str(h)]) for h in range(3)}}
                                          for name, values in metrics.items() if name != "quantum_6q_reuploading"},
               "residual_prediction_metrics": forecast_metrics(assessment["y_true_original"].astype(np.float64) - baseline, bias + centered),
               "selected_epoch": epoch, "parameter_counts": model.parameter_counts(), "checks": checks,
               "source_sha256": source_hashes, "configuration": settings, "runtime_estimate": estimates,
               "runtime_seconds": time.perf_counter() - started, "hardware": platform.processor(),
               "torch_version": str(torch.__version__), "python_version": platform.python_version(),
               "warnings": list(stage0.get("warnings", [])) + [HISTORICAL_NOTE] + (["Smoke test only; not scientific evidence."] if args.smoke_test else []),
               "decision": "No automatic GO/NO-GO or superiority claim; compare existing controls and saved thirds."}
    rows = [{"model": name, "scope": scope, **(values["aggregate"] if scope == "aggregate" else values["per_output"][scope])}
            for name, values in metrics.items() for scope in ("aggregate", "0", "1", "2")]
    with (evaluation / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(evaluation / "assessment_predictions.npz", sample_index=assessment["sample_index"],
                        y_true_original=assessment["y_true_original"], baseline_prediction=baseline,
                        predicted_centered_residual=centered, prediction_quantum_6q_reuploading=corrected,
                        **{key: value for key, value in assessment.items() if key.endswith("timestamp")})
    save_json(evaluation / "training_history.json", history)
    save_json(evaluation / "quantum_reuploading_summary.json", summary)
    lines = ["Six-qubit data re-uploading residual pilot", HISTORICAL_NOTE,
             "Model                         RMSE          MAE           R2            nRMSE"]
    for name, values in metrics.items():
        v = values["aggregate"]
        lines.append(f"{name:28s} {v['rmse']:13.9f} {v['mae']:13.9f} {v['r2']:13.9f} {v['nrmse']:13.9f}")
    lines += [f"Re-uploading comparisons: {json.dumps(summary['reuploading_comparisons'])}",
              f"Residual metrics: {json.dumps(summary['residual_prediction_metrics'])}",
              f"Chronological thirds: {json.dumps(periods)}", f"Epoch: {epoch}; CPU runtime: {summary['runtime_seconds']:.2f}s",
              summary["decision"]]
    with (evaluation / "quantum_reuploading_summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    logging.info("Re-uploading pilot complete: %s", evaluation)
    print("\n".join(lines[:10]))


def main() -> None:
    artifacts, evaluation = output_paths(None)
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("stage0-artifacts", artifacts), ("stage0-evaluation", evaluation),
                          ("stage0-5-artifacts", artifacts / "stage0_5"), ("stage0-5-evaluation", evaluation / "stage0_5"),
                          ("quantum-pilot-evaluation", evaluation / "quantum_pilot")):
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=32)
    args = parser.parse_args()
    args.seed, args.learning_rate, args.batch_size, args.patience = 42, 1e-3, 128, 8
    args.epochs = 1 if args.smoke_test else 30
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
