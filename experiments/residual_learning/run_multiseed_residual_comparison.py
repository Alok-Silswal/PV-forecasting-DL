"""Fixed-protocol, paired five-seed residual comparison; CPU execution only.

Recovery may inspect historical assessment anchors for provenance. New residual
assessment predictions are computed only after each selected checkpoint is saved.
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler

from experiments.residual_learning.extract_residual_dataset import (
    ROOT, CHECKPOINT, RECOVERY_BIAS_ATOL, compare_scaler_stats,
    extract_residual_dataset, output_paths, require_new_files, save_json,
    scaler_stats, setup, sha256, verify_recorded_file,
)
from experiments.residual_learning.run_residual_audit import forecast_metrics, load_partition, mlp_predict
from experiments.residual_learning.run_stage0_5 import check_baseline, fit_bottleneck, run as restore_stage05
from experiments.residual_learning.run_quantum_residual_pilot import fit as fit_quantum, verify_model
from models.residual_learning.bottleneck_residual_mlp import BottleneckResidualMLP
from models.residual_learning.quantum_residual_vqc import QuantumResidualVQC
from models.residual_learning.quantum_residual_reupload_vqc import QuantumResidualReuploadVQC
from models.residual_learning.quantum_fixed_reupload_feature_map import QuantumFixedReuploadFeatureMap

SEEDS = (42, 123, 456, 789, 2026)
ORIGINAL_MODELS = ("bottleneck_mlp", "quantum_single", "quantum_reuploading")
MODELS = ("bottleneck_mlp", "quantum_fixed_reuploading", "quantum_single", "quantum_reuploading")
SMOKE_SAMPLES = 32
METRICS = ("rmse", "mae", "r2", "nrmse")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required persisted experiment record missing: {path}. Persist the completed historical experiment first; this runner never retrains it.")
    return json.loads(path.read_text(encoding="utf-8"))


def portable_hash(path: Path) -> str:
    """Keep immutable text provenance stable across Git's LF/CRLF conversion."""
    if path.suffix in {".json", ".py"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return sha256(path)


def protected_snapshot() -> dict[Path, str]:
    """Exclude this experiment's namespace, including its smoke outputs."""
    paths = [CHECKPOINT, *ROOT.glob("models/**/*.py"),
             *ROOT.glob("experiments/residual_learning/*.py"),
             *ROOT.glob("notebooks/*.ipynb")]
    paths += [ROOT / p for p in ("main.py", "evaluation/evaluate.py", "evaluation/evaluator.py",
                                "training/trainer.py", "configs/config.py", "data/processed/DKASC_Preprocessed.csv")]
    for directory in (ROOT / "artifacts/residual_learning", ROOT / "evaluation/residual_learning"):
        paths += [p for p in directory.rglob("*") if p.is_file() and "multiseed" not in p.parts
                  and "__pycache__" not in p.parts]
    return {p: sha256(p) for p in paths if p.is_file()}


def assert_unchanged(snapshot: dict[Path, str]) -> None:
    for path, digest in snapshot.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Protected historical file changed: {path}")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse existing recovery; fit no new bias, scalers or historical controls."""
    artifacts, evaluation = output_paths(None)
    protocol_settings = read_json(artifacts / "stage0_5/settings.json")
    historical_single = read_json(evaluation / "quantum_pilot/quantum_pilot_summary.json")
    mlp_protocol = {key: protocol_settings[key] for key in
                    ("bottleneck_dim", "hidden_dim", "dropout", "optimizer", "learning_rate",
                     "weight_decay", "maximum_epochs", "patience", "batch_size")}
    expected = dict(bottleneck_dim=6, hidden_dim=16, dropout=0.1, optimizer="Adam",
                    learning_rate=0.001, weight_decay=1e-5, maximum_epochs=100, patience=10, batch_size=512)
    if mlp_protocol != expected:
        raise ValueError("Stage-0.5 saved protocol differs from its reusable fit_bottleneck helper; inspect before running.")
    quantum_protocol = dict(optimizer="Adam", learning_rate=0.001, weight_decay=0.0,
                            maximum_epochs=30, patience=8, batch_size=128, qubits=6, layers=2,
                            backend="default.qubit", interface="torch", differentiation="backprop", shots=None)
    for key, value in (("learning_rate", 0.001), ("batch_size", 128), ("epochs", 30), ("patience", 8)):
        if historical_single["configuration"][key] != value:
            raise ValueError(f"Historical quantum protocol drift: {key}")
    for key in ("backend", "interface", "differentiation", "shots"):
        if historical_single[key] != quantum_protocol[key]:
            raise ValueError(f"Historical quantum backend drift: {key}")
    if historical_single["device"] != "cpu" or historical_single["pennylane_version"] != "0.45.1":
        raise ValueError("Expected the completed CPU PennyLane 0.45.1 pilot.")
    stage05_artifacts, stage05_evaluation = artifacts / "stage0_5", evaluation / "stage0_5"
    single_evaluation, reupload_evaluation = evaluation / "quantum_pilot", evaluation / "quantum_reuploading_pilot"
    # Local smoke can reuse the project's existing small verified extraction.
    smoke_artifacts, smoke_evaluation = output_paths(1024)
    smoke05 = ROOT / "artifacts/residual_learning/_smoke/stage0_5/max_512/proposed/horizon_15/run_1"
    smoke05_eval = ROOT / "evaluation/residual_learning/_smoke/stage0_5/max_512/proposed/horizon_15/run_1"
    smoke_single = ROOT / "evaluation/residual_learning/_smoke/quantum_pilot/max_16/proposed/horizon_15/run_1"
    smoke_reupload = ROOT / "evaluation/residual_learning/_smoke/quantum_reuploading_pilot/max_32/proposed/horizon_15/run_1"
    if (args.smoke_test and not (artifacts / "train.npz").is_file()
            and all(p.is_file() for p in (smoke_artifacts / "manifest.json", smoke_artifacts / "train.npz",
                                         smoke05 / "latent_scaler.pkl", smoke_single / "quantum_pilot_summary.json",
                                         smoke_reupload / "quantum_reuploading_summary.json"))):
        artifacts, evaluation = smoke_artifacts, smoke_evaluation
        stage05_artifacts, stage05_evaluation = smoke05, smoke05_eval
        single_evaluation, reupload_evaluation = smoke_single, smoke_reupload
        logging.info("Smoke uses existing verified 1024-sample artifacts; full protocols remain locked.")
    paths = {"manifest": artifacts / "manifest.json", "stage0": evaluation / "stage0_summary.json",
             "stage05": stage05_evaluation / "stage0_5_summary.json",
             "single": single_evaluation / "quantum_pilot_summary.json",
             "reupload": reupload_evaluation / "quantum_reuploading_summary.json",
             "scaler_settings": stage05_artifacts / "settings.json", "bias": stage05_artifacts / "bias.npy"}
    documents = {name: read_json(path) for name, path in paths.items() if path.suffix == ".json"}
    manifest = documents["manifest"]
    if not args.smoke_test:
        if manifest["max_samples_per_partition"] is not None or {k: v["samples"] for k, v in manifest["partitions"].items()} != {"train": 95860, "tuning": 31927, "assessment": 31928}:
            raise ValueError("Full comparison requires the locked full historical residual partitions.")
        if documents["reupload"]["configuration"]["maximum_epochs"] != 30:
            raise ValueError("Full historical re-uploading protocol must have maximum_epochs=30.")
        if protocol_settings["seed"] != 42 or historical_single["configuration"]["seed"] != 42 or documents["reupload"]["configuration"]["seed"] != 42:
            raise ValueError("Historical reproducibility references must be seed 42.")
    references = {"baseline": documents["stage0"]["assessment_metrics"]["baseline"],
                  "bias_only": documents["stage05"]["assessment_metrics"]["bias_only"],
                  "stage0_ridge": documents["stage0"]["assessment_metrics"]["ridge"],
                  "stage0_mlp": documents["stage0"]["assessment_metrics"]["mlp"],
                  "stage0_5_6d_mlp": documents["stage05"]["assessment_metrics"]["bias_plus_6d_mlp"]}
    for record in (documents["single"], documents["reupload"]):
        if any(record["assessment_metrics"][key] != value for key, value in references.items()):
            raise ValueError("Historical comparator metrics are not internally consistent.")
    for name in ("stage0", "stage05", "single", "reupload", "scaler_settings"):
        if documents[name]["partitions"] != manifest["partitions"]:
            raise ValueError(f"Historical partition provenance differs: {name}")
    if any(not (artifacts / f"{name}.npz").is_file() for name in ("train", "tuning", "assessment")):
        extract_residual_dataset(processed_csv=args.processed_csv,
                                 max_samples=manifest["max_samples_per_partition"])
    if any(not (stage05_artifacts / f"{name}.pkl").is_file() for name in ("latent_scaler", "centered_residual_scaler")):
        if manifest["max_samples_per_partition"] is not None:
            raise FileNotFoundError("Smoke source scalers missing; restore its Stage-0.5 artifacts before running.")
        restore_stage05(argparse.Namespace(stage0_artifacts=artifacts, stage0_evaluation=evaluation,
                                          smoke_test=False, smoke_samples=512, epochs=100, patience=10,
                                          batch_size=512, seed=42, cpu_threads=1))
    train, tuning = (load_partition(artifacts, name, manifest) for name in ("train", "tuning"))
    bias = np.load(paths["bias"], allow_pickle=False)
    train_residual = train["y_true_original"].astype(np.float64) - train["y_hat_original"].astype(np.float64)
    np.testing.assert_allclose(bias, train_residual.mean(0), atol=RECOVERY_BIAS_ATOL, rtol=0)
    np.testing.assert_array_equal(bias, documents["stage05"]["learned_bias_original"])
    scalers = {}
    for name, values in (("latent_scaler", train["z"]), ("centered_residual_scaler", train_residual - bias)):
        path = stage05_artifacts / f"{name}.pkl"
        verify_recorded_file(path, documents["single"]["source_sha256"])
        with path.open("rb") as handle:
            scalers[name] = pickle.load(handle)
        compare_scaler_stats(scaler_stats(scalers[name]), scaler_stats(StandardScaler().fit(values)))
        paths[name] = path
    third_records = documents["stage0"]["assessment_chronological_thirds"]
    for name in ("stage05", "single", "reupload"):
        periods = documents[name]["assessment_chronological_thirds"]
        if len(periods) != 3 or any(any(a[k] != b[k] for k in ("period", "samples", "first_target", "last_target"))
                                    for a, b in zip(third_records, periods)):
            raise ValueError(f"Chronological thirds differ: {name}")
    if sum(p["samples"] for p in third_records) != manifest["partitions"]["assessment"]["samples"]:
        raise ValueError("Saved thirds do not cover assessment.")
    arrays = {}
    for name, data in (("train", train), ("tuning", tuning)):
        residual = data["y_true_original"].astype(np.float64) - data["y_hat_original"].astype(np.float64) - bias
        arrays[name + "_z"] = scalers["latent_scaler"].transform(data["z"]).astype(np.float32)
        arrays[name + "_r"] = scalers["centered_residual_scaler"].transform(residual).astype(np.float32)
        if args.smoke_test:
            arrays[name + "_z"] = arrays[name + "_z"][:SMOKE_SAMPLES]
            arrays[name + "_r"] = arrays[name + "_r"][:SMOKE_SAMPLES]
    import pennylane as qml
    if qml.__version__ != "0.45.1":
        raise RuntimeError("Locked experiment requires PennyLane 0.45.1.")
    versions = dict(python=platform.python_version(), torch=str(torch.__version__),
                    numpy=np.__version__, sklearn=sklearn.__version__, pennylane=qml.__version__)
    code_paths = [Path(__file__), *[ROOT / "experiments/residual_learning" / name for name in
                 ("extract_residual_dataset.py", "run_residual_audit.py", "run_stage0_5.py", "run_quantum_residual_pilot.py", "run_quantum_reuploading_pilot.py")],
                 *[ROOT / "models/residual_learning" / name for name in
                 ("bottleneck_residual_mlp.py", "quantum_residual_vqc.py", "quantum_residual_reupload_vqc.py", "quantum_fixed_reupload_feature_map.py")]]
    provenance = {"immutable_source_sha256": {name: portable_hash(path) for name, path in paths.items() if path.suffix != ".pkl"},
                  "code_sha256": {p.relative_to(ROOT).as_posix(): portable_hash(p) for p in code_paths},
                  "source_sha256": {name: sha256(path) for name, path in paths.items()},
                  "partition_file_sha256": {name: sha256(artifacts / f"{name}.npz") for name in ("train", "tuning", "assessment")},
                  "partition_metadata": manifest["partitions"], "manifest": manifest,
                  "bias": bias.tolist(), "scaler_statistics": {name: scaler_stats(value) for name, value in scalers.items()},
                  "versions": versions}
    return dict(artifacts=artifacts, evaluation=evaluation, documents=documents, manifest=manifest,
                bias=bias, scalers=scalers, arrays=arrays, thirds=third_records, provenance=provenance,
                full_partitions=historical_single["partitions"],
                protocols={"bottleneck_mlp": mlp_protocol, "quantum_single": quantum_protocol,
                           "quantum_reuploading": dict(quantum_protocol, encoding_applications=2),
                           "quantum_fixed_reuploading": dict(quantum_protocol, encoding_applications=2,
                                                              fixed_quantum_angles=0.0, trainable_quantum_parameters=0)})


def verify_fixed_map(model: QuantumFixedReuploadFeatureMap, features: np.ndarray, seed: int) -> dict:
    """Check topology, classical gradients, nonlinearity and matched initialization."""
    batch = torch.from_numpy(features[:4])
    angles = math.pi * torch.tanh(model.projection(batch))
    quantum, output = model.quantum_features(batch), model(batch)
    assert angles.shape == quantum.shape == (len(batch), 6) and output.shape == (len(batch), 3)
    counts = model.parameter_counts()
    assert counts == {"projection": 774, "quantum": 0, "readout": 21, "total": 795}
    assert "weights" not in dict(model.named_parameters()) and not model.weights.requires_grad
    output.square().mean().backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not parameter.grad.abs().max().item():
            raise RuntimeError(f"Fixed-map classical gradient missing/nonfinite/zero: {name}")
        gradients[name] = float(parameter.grad.norm())
    assert model.weights.grad is None
    model.zero_grad(set_to_none=True)
    tape = model.circuit.construct((angles, model.weights), {})
    expected = []
    for _ in range(2):
        expected += [("RY", [wire]) for wire in range(6)]
        expected += [(gate, [wire]) for wire in range(6) for gate in ("RZ", "RY")]
        expected += [("CNOT", [wire, (wire + 1) % 6]) for wire in range(6)]
    assert [(op.name, list(op.wires)) for op in tape.operations] == expected
    assert [(measurement.obs.name, list(measurement.wires)) for measurement in tape.measurements] == [("PauliZ", [wire]) for wire in range(6)]
    for start in (0, 24):
        for wire in range(6):
            torch.testing.assert_close(tape.operations[start + wire].data[0], angles[:, wire], rtol=0, atol=0)
            for index in (start + 6 + 2 * wire, start + 7 + 2 * wire):
                assert tape.operations[index].data[0].item() == 0 and not tape.operations[index].data[0].requires_grad
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        comparator = QuantumResidualReuploadVQC()
        for name in ("projection", "readout"):
            for key, value in getattr(model, name).state_dict().items():
                torch.testing.assert_close(value, getattr(comparator, name).state_dict()[key], rtol=0, atol=0)
    generator = torch.Generator().manual_seed(2026)
    u, v = torch.randn(2, 128, generator=generator).split(1)
    with torch.no_grad():
        serial = torch.cat([model(row[None]) for row in batch])
        torch.testing.assert_close(output.detach(), serial, rtol=1e-5, atol=1e-6)
        dependence = float((model.quantum_features(u) - model.quantum_features(v)).abs().max())
        nonlinearity = float((model((u + v) / 2) - (model(u) + model(v)) / 2).abs().max())
    if dependence <= 1e-6 or nonlinearity <= 1e-6:
        raise RuntimeError("Fixed-map input-dependence/non-affinity sanity check failed.")
    return dict(input_shape=list(batch.shape), angle_shape=list(angles.shape), quantum_shape=list(quantum.shape),
                output_shape=list(output.shape), parameter_counts=counts, gradient_norms=gradients,
                fixed_angles_have_no_gradients=True, encoding_applications=2, cnot_rings=2,
                broadcast_matches_serial=True, classical_initialization_matches_trainable_comparator=True,
                maximum_input_response=dependence, midpoint_affine_identity_violation=nonlinearity)


def verify_mlp(features: np.ndarray) -> dict:
    """Probe outside the training RNG stream, including dropout consumption."""
    with torch.random.fork_rng(devices=[]):
        model = BottleneckResidualMLP()
        batch = torch.from_numpy(features[:4])
        output = model(batch)
        assert model.projection(batch).shape == (len(batch), 6) and output.shape == (len(batch), 3)
        output.square().mean().backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite/missing classical gradient.")
        assert sum(p.numel() for p in model.parameters()) == 937
    return dict(input_shape=list(batch.shape), bottleneck_shape=[len(batch), 6], output_shape=list(output.shape),
                finite_gradients=True, parameter_counts={"total": 937})


def validate_reuse(settings: dict, summary: dict, expected: dict, artifacts: Path, evaluation: Path) -> None:
    for key, value in expected.items():
        if key != "provenance" and settings.get(key) != value:
            raise ValueError(f"Completed model/seed configuration conflict: {key}")
    old, current = settings["provenance"], expected["provenance"]
    for key in ("immutable_source_sha256", "partition_metadata", "bias", "versions"):
        if old[key] != current[key]:
            raise ValueError(f"Completed model/seed provenance conflict: {key}")
    # Additive orchestration/control changes must not invalidate the original
    # replicates. Training helpers and applicable model implementations stay fixed.
    for path, digest in current["code_sha256"].items():
        if Path(path).name == Path(__file__).name:
            continue
        if Path(path).name == "quantum_fixed_reupload_feature_map.py" and expected["model"] != "quantum_fixed_reuploading":
            continue
        if old["code_sha256"].get(path) != digest:
            raise ValueError(f"Completed training/model implementation conflict: {path}")
    for name, stats in current["scaler_statistics"].items():
        compare_scaler_stats(old["scaler_statistics"][name], stats)
    if summary["settings_fingerprint"] != hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest():
        raise ValueError("Completed settings and summary disagree.")
    if not summary["assessment_evaluated_after_checkpoint_lock"] or not 1 <= summary["selected_epoch"] <= settings["effective_epochs"]:
        raise ValueError("Invalid completed checkpoint-selection record.")
    if any(summary[key] != settings[key] for key in ("model", "seed", "selected_epoch", "parameter_counts")):
        raise ValueError("Completed summary and settings disagree.")
    checkpoint = artifacts / "best_checkpoint.pt"
    if checkpoint.exists() and sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint conflicts with summary.")
    prediction_path = evaluation / "assessment_predictions.npz"
    if prediction_path.exists():
        if sha256(prediction_path) != summary["assessment_predictions_sha256"]:
            raise ValueError("Completed assessment predictions conflict with summary.")
        with np.load(prediction_path, allow_pickle=False) as saved:
            if forecast_metrics(saved["y_true_original"], saved["prediction_corrected"]) != summary["assessment_metrics"]:
                raise ValueError("Completed metrics differ from saved predictions.")
            if forecast_metrics(saved["y_true_original"] - saved["prediction_baseline"], saved["predicted_residual"]) != summary["residual_prediction_metrics"]:
                raise ValueError("Completed residual metrics differ from saved predictions.")
    history_path = evaluation / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best = min(history, key=lambda row: row["tuning_mse"])["epoch"]
        if best != summary["selected_epoch"] or history != summary["training_history"]:
            raise ValueError("Completed training history conflicts with selected epoch.")


def run_replicate(context: dict, seed: int, name: str, artifacts: Path,
                  evaluation: Path, smoke: bool) -> dict:
    """Train from scratch; publish summary last as the completion marker."""
    protocol = context["protocols"][name]
    expected = dict(model=name, seed=seed, declared_seeds=list(SEEDS), protocol=protocol, smoke_test=smoke,
                    effective_epochs=1 if smoke else protocol["maximum_epochs"], cpu_threads=1, device="cpu",
                    loss="standardized centered-residual MSE", provenance=context["provenance"],
                    train_samples=len(context["arrays"]["train_z"]), tuning_samples=len(context["arrays"]["tuning_z"]),
                    assessment_source_thirds=[{key: p[key] for key in ("period", "samples", "first_target", "last_target")} for p in context["thirds"]])
    summary_path, settings_path = evaluation / "summary.json", artifacts / "settings.json"
    if summary_path.is_file():
        summary, settings = read_json(summary_path), read_json(settings_path)
        validate_reuse(settings, summary, expected, artifacts, evaluation)
        logging.info("Reusing completed %s seed=%d; checkpoint/prediction archive not required if absent.", name, seed)
        return summary
    require_new_files([settings_path, artifacts / "best_checkpoint.pt", evaluation / "training_history.json",
                       evaluation / "assessment_predictions.npz", summary_path])
    artifacts.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    logging.info("Training %s seed=%d from scratch; no mid-training resume is claimed.", name, seed)
    started = time.perf_counter()
    setup(seed, 1)
    device = torch.device("cpu")
    arrays = context["arrays"]
    args = argparse.Namespace(seed=seed, epochs=expected["effective_epochs"], batch_size=protocol["batch_size"],
                              patience=protocol["patience"], learning_rate=protocol["learning_rate"])
    if name == "bottleneck_mlp":
        checks = verify_mlp(arrays["train_z"]) if smoke else {}
        model, epoch, history = fit_bottleneck(arrays["train_z"], arrays["train_r"], arrays["tuning_z"], arrays["tuning_r"], args, device)
        counts = {"total": sum(p.numel() for p in model.parameters())}
        assert counts["total"] == 937
    else:
        model = {"quantum_single": QuantumResidualVQC, "quantum_reuploading": QuantumResidualReuploadVQC,
                 "quantum_fixed_reuploading": QuantumFixedReuploadFeatureMap}[name]()
        checks = (verify_fixed_map(model, arrays["train_z"], seed) if name == "quantum_fixed_reuploading"
                  else verify_model(model, arrays["train_z"], device)) if smoke else {}
        counts = model.parameter_counts()
        quantum_count = 0 if name == "quantum_fixed_reuploading" else 24
        if counts != {"projection": 774, "quantum": quantum_count, "readout": 21, "total": 795 + quantum_count}:
            raise ValueError("Quantum architecture parameter count drift.")
        epoch, history = fit_quantum(model, arrays["train_z"], arrays["train_r"], arrays["tuning_z"], arrays["tuning_r"], args, device)
    checkpoint = artifacts / "best_checkpoint.pt"
    with checkpoint.open("xb") as handle:
        torch.save(dict(model_state_dict=model.state_dict(), selected_epoch=epoch, seed=seed, model=name), handle)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        restored = {"bottleneck_mlp": BottleneckResidualMLP, "quantum_single": QuantumResidualVQC,
                    "quantum_reuploading": QuantumResidualReuploadVQC,
                    "quantum_fixed_reuploading": QuantumFixedReuploadFeatureMap}[name]()
    restored.load_state_dict(state["model_state_dict"])
    restored.eval()
    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(arrays["train_z"][:4])
        torch.testing.assert_close(model(batch), restored(batch), rtol=0, atol=0)
    settings = dict(expected, selected_epoch=epoch, parameter_counts=counts)
    save_json(settings_path, settings)
    save_json(evaluation / "training_history.json", history)
    logging.info("Checkpoint locked: %s seed=%d epoch=%d; now reading assessment.", name, seed, epoch)
    assessment = load_partition(context["artifacts"], "assessment", context["manifest"])
    offset, selected, periods = 0, [], []
    for period in context["thirds"]:
        end = offset + period["samples"]
        if str(assessment["target_0_timestamp"][offset]) != period["first_target"] or str(assessment["target_2_timestamp"][end - 1]) != period["last_target"]:
            raise ValueError("Assessment third boundaries changed.")
        indices = np.arange(offset, min(end, offset + 8) if smoke else end)
        selected.extend(indices.tolist())
        periods.append(dict(period=period["period"], source_samples=period["samples"],
                            first_target=period["first_target"], last_target=period["last_target"], samples=len(indices)))
        offset = end
    indices = np.asarray(selected, dtype=np.int64)
    truth = assessment["y_true_original"][indices].astype(np.float64)
    baseline = assessment["y_hat_original"][indices].astype(np.float64)
    features = context["scalers"]["latent_scaler"].transform(assessment["z"][indices]).astype(np.float32)
    centered = context["scalers"]["centered_residual_scaler"].inverse_transform(mlp_predict(model, features, device, protocol["batch_size"])).astype(np.float64)
    predicted_residual = context["bias"] + centered
    corrected = baseline + context["bias"] + centered
    np.testing.assert_array_equal(corrected, baseline + context["bias"] + centered)
    metrics = forecast_metrics(truth, corrected)
    residual_metrics = forecast_metrics(truth - baseline, predicted_residual)
    if not smoke:
        check_baseline(forecast_metrics(truth, baseline), context["documents"]["stage0"]["assessment_metrics"]["baseline"])
        check_baseline(forecast_metrics(truth, baseline + context["bias"]), context["documents"]["stage05"]["assessment_metrics"]["bias_only"])
    offset = 0
    for period in periods:
        end = offset + period["samples"]
        period["metrics"] = forecast_metrics(truth[offset:end], corrected[offset:end])
        offset = end
    prediction_path = evaluation / "assessment_predictions.npz"
    data = dict(y_true_original=truth, prediction_baseline=baseline, prediction_corrected=corrected,
                predicted_centered_residual=centered, predicted_residual=predicted_residual,
                sample_index=assessment["sample_index"][indices])
    for key in ("input_start_timestamp", "input_end_timestamp", "target_0_timestamp", "target_1_timestamp", "target_2_timestamp"):
        data[key] = assessment[key][indices]
    with prediction_path.open("xb") as handle:
        np.savez_compressed(handle, **data)
    historical_keys = {"bottleneck_mlp": ("stage05", "bias_plus_6d_mlp"),
                       "quantum_single": ("single", "quantum_6q_angle"),
                       "quantum_reuploading": ("reupload", "quantum_6q_reuploading")}
    reproducibility = None
    if seed == 42 and not smoke and name in historical_keys:
        reference_name, reference_key = historical_keys[name]
        reference = context["documents"][reference_name]
        old = reference["assessment_metrics"][reference_key]["aggregate"]
        differences = {key: metrics["aggregate"][key] - old[key] for key in METRICS}
        review = any(abs(differences[key]) > 1e-5 for key in ("rmse", "mae"))
        reproducibility = dict(historical=old, current=metrics["aggregate"], differences=differences,
                               historical_selected_epoch=reference["selected_epoch"], current_selected_epoch=epoch,
                               exceeds_direct_recovery_error_bound=review,
                               investigation="Compare selected epochs, saved training histories, software/hardware and raw recovered-data hashes; nonconvex training may amplify tiny input changes. No causal explanation is assumed.")
        if review:
            logging.warning("Seed-42 %s differs beyond direct reconstruction variation: %s; diagnostics saved for review.", name, differences)
    # Count-based extrapolation is deliberately separate from measured runtime.
    full = context["full_partitions"]
    train_ratio = math.ceil(full["train"]["samples"] / protocol["batch_size"]) / math.ceil(len(arrays["train_z"]) / protocol["batch_size"])
    tune_ratio = math.ceil(full["tuning"]["samples"] / protocol["batch_size"]) / math.ceil(len(arrays["tuning_z"]) / protocol["batch_size"])
    if name == "bottleneck_mlp":
        epoch_estimate = float(np.mean([h["epoch_seconds"] for h in history])) * (train_ratio + tune_ratio) / 2
    else:
        epoch_estimate = float(np.mean([h["training_seconds"] for h in history])) * train_ratio + float(np.mean([h["tuning_seconds"] for h in history])) * tune_ratio
    summary = dict(model=name, seed=seed, selected_epoch=epoch, parameter_counts=counts, assessment_metrics=metrics,
                   residual_prediction_metrics=residual_metrics, assessment_chronological_thirds=periods,
                   training_history=history, runtime_seconds=time.perf_counter() - started,
                   hardware=dict(platform=platform.platform(), processor=platform.processor(), device="cpu", cpu_threads=1),
                   assessment_evaluated_after_checkpoint_lock=True, checks=dict(checks, checkpoint_reload_exact=True, additive_identity_exact=True),
                   settings_fingerprint=hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
                   checkpoint_sha256=sha256(checkpoint),
                   assessment_predictions_sha256=sha256(prediction_path), historical_seed42_comparison=reproducibility,
                   estimated_full_epoch_seconds=epoch_estimate,
                   estimate_note="Batch-count extrapolation; tiny batches/hardware can distort estimates; excludes recovery/loading/assessment.")
    save_json(summary_path, summary)
    return summary


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return dict(mean=float(array.mean()), sample_sd=float(array.std(ddof=1)), median=float(np.median(array)),
                minimum=float(array.min()), maximum=float(array.max()))


def summarize_metrics(records: list[dict]) -> dict:
    return {"aggregate": {key: describe([r["aggregate"][key] for r in records]) for key in METRICS},
            "per_output": {str(h): {key: describe([r["per_output"][str(h)][key] for r in records]) for key in METRICS} for h in range(3)}}


def paired(records: dict, seeds: list[int], comparator: str, period: int | None = None,
           left_model: str = "quantum_reuploading") -> dict:
    differences = []
    for seed in seeds:
        left, right = records[(seed, left_model)], records[(seed, comparator)]
        a = left["assessment_metrics"] if period is None else left["assessment_chronological_thirds"][period]["metrics"]
        b = right["assessment_metrics"] if period is None else right["assessment_chronological_thirds"][period]["metrics"]
        differences.append(dict(seed=seed, **{key: a["aggregate"][key] - b["aggregate"][key] for key in ("rmse", "mae")}))
    return dict(differences=differences, statistics={key: describe([d[key] for d in differences]) for key in ("rmse", "mae")},
                lower_error_seed_counts={key: sum(d[key] < 0 for d in differences) for key in ("rmse", "mae")},
                left_model=left_model, right_model=comparator,
                sign="left model minus right model; negative means lower error")


def aggregate(context: dict, records: dict, seeds: list[int], smoke: bool) -> dict:
    references = {"baseline": context["documents"]["stage0"]["assessment_metrics"]["baseline"],
                  "bias_only": context["documents"]["stage05"]["assessment_metrics"]["bias_only"],
                  "ridge": context["documents"]["stage0"]["assessment_metrics"]["ridge"],
                  "stage0_mlp": context["documents"]["stage0"]["assessment_metrics"]["mlp"]}
    models = {}
    for name in MODELS:
        rows = [records[(seed, name)] for seed in seeds]
        models[name] = dict(assessment=summarize_metrics([r["assessment_metrics"] for r in rows]),
                            residual_prediction=summarize_metrics([r["residual_prediction_metrics"] for r in rows]),
                            chronological_thirds=[dict(period=context["thirds"][p]["period"],
                                first_target=context["thirds"][p]["first_target"], last_target=context["thirds"][p]["last_target"],
                                statistics=summarize_metrics([r["assessment_chronological_thirds"][p]["metrics"] for r in rows])) for p in range(3)],
                            runtime_seconds=sum(r["runtime_seconds"] for r in rows),
                            parameter_counts=rows[0]["parameter_counts"],
                            estimated_five_seed_max_epoch_seconds=5 * context["protocols"][name]["maximum_epochs"] * float(np.mean([r["estimated_full_epoch_seconds"] for r in rows])))
    comparisons = {"bottleneck_mlp": ("quantum_reuploading", "bottleneck_mlp", "Complete trainable quantum branch versus compressed classical control."),
                   "quantum_single": ("quantum_reuploading", "quantum_single", "Contribution of data re-uploading in the chosen trainable VQC.")}
    if "quantum_fixed_reuploading" in MODELS:
        comparisons.update({"quantum_fixed_reuploading": ("quantum_reuploading", "quantum_fixed_reuploading", "Contribution of trainable variational rotations within the chosen re-uploading topology."),
                            "fixed_map_vs_bottleneck_mlp": ("quantum_fixed_reuploading", "bottleneck_mlp", "Fixed quantum feature transformation versus compressed classical control.")})
    return dict(experiment="Locked-protocol paired residual reproducibility", declared_seeds=list(SEEDS), aggregate_seeds=seeds,
                smoke_test=smoke, models=models, fixed_references=references,
                descriptive_differences_vs_fixed_references=None if smoke else {
                    name: {reference: {key: describe([records[(seed, name)]["assessment_metrics"]["aggregate"][key] - values["aggregate"][key] for seed in seeds])
                                       for key in ("rmse", "mae")} for reference, values in references.items()} for name in MODELS},
                paired_comparisons={name: dict(paired(records, seeds, right, left_model=left), question=question)
                                    for name, (left, right, question) in comparisons.items()},
                paired_chronological_thirds={name: [dict(paired(records, seeds, right, p, left), question=question) for p in range(3)]
                                            for name, (left, right, question) in comparisons.items()},
                historical_seed42_checks={name: records[(42, name)]["historical_seed42_comparison"] for name in MODELS},
                protocols=context["protocols"], provenance={key: value for key, value in context["provenance"].items()
                    if key not in ("source_sha256", "partition_file_sha256", "scaler_statistics")},
                protected_historical_files_unchanged=True,
                fixed_control_design="Fixed quantum re-uploading feature map: two RY encodings, RZ(0)/RY(0), two CNOT rings, six Z measurements; projection/readout trainable; zero trainable quantum parameters.",
                notes=["Seed 42 is retrained here; historical metrics never enter the replicate distributions.",
                       "No seed or architecture selection; sample SD uses ddof=1; no significance test.",
                       "Historical validation previously selected the backbone; this remains exploratory.",
                       "Smoke metrics use 32 train/tuning rows and 8 assessment rows from each saved third; not scientific results."] if smoke else
                      ["Seed 42 is retrained here; historical metrics never enter the replicate distributions.",
                       "No seed or architecture selection; sample SD uses ddof=1; no significance test.",
                       "Historical validation previously selected the backbone; this remains exploratory."])


def write_aggregate(evaluation: Path, summary: dict, records: dict, seeds: list[int]) -> None:
    """Archive a verified three-model aggregate before adding the fourth control."""
    def text(value: dict) -> str:
        lines = ["Paired residual reproducibility experiment", f"Seeds: {seeds}; smoke={value['smoke_test']}",
                 "Aggregate forecast mean +/- sample SD:"]
        for name, record in value["models"].items():
            lines.append(name + ": " + "; ".join(f"{key}={values['mean']:.9f} +/- {values['sample_sd']:.9f}" for key, values in record["assessment"]["aggregate"].items()))
        sign = "re-uploading minus comparator" if len(value["models"]) == 3 else "left model minus right model"
        lines += [f"Paired differences ({sign}):", json.dumps(value["paired_comparisons"], indent=2),
                  "Historical seed-42 checks:", json.dumps(value["historical_seed42_checks"], indent=2),
                  "Per-output, residual and chronological statistics are in multiseed_summary.json.",
                  "Runtime extrapolations exclude recovery/loading/assessment and may be inaccurate on tiny smoke batches."]
        return "\n".join(lines) + "\n"

    def table(names: tuple) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["seed", "model", "selected_epoch", *METRICS, "residual_rmse", "residual_mae", "residual_r2", "runtime_seconds"])
        for seed in seeds:
            for name in names:
                row = records[(seed, name)]
                writer.writerow([seed, name, row["selected_epoch"], *[row["assessment_metrics"]["aggregate"][k] for k in METRICS],
                                 *[row["residual_prediction_metrics"]["aggregate"][k] for k in ("rmse", "mae", "r2")], row["runtime_seconds"]])
        return buffer.getvalue()

    evaluation.mkdir(parents=True, exist_ok=True)
    json_path = evaluation / "multiseed_summary.json"
    text_outputs = {evaluation / "multiseed_summary.txt": text(summary), evaluation / "per_seed_metrics.csv": table(MODELS)}
    upgrade = False
    if json_path.exists():
        old = read_json(json_path)
        if old != summary:
            if set(old["models"]) != set(ORIGINAL_MODELS) or set(summary["models"]) != set(MODELS) or len(MODELS) != 4:
                raise ValueError("Existing aggregate conflicts with completed model/seed results.")
            for key in ("declared_seeds", "aggregate_seeds", "smoke_test", "fixed_references"):
                if old[key] != summary[key]:
                    raise ValueError(f"Three-model aggregate conflict: {key}")
            for key in ("immutable_source_sha256", "partition_metadata", "manifest", "bias", "versions"):
                if old["provenance"][key] != summary["provenance"][key]:
                    raise ValueError(f"Three-model aggregate provenance conflict: {key}")
            for path, digest in old["provenance"]["code_sha256"].items():
                if Path(path).name != Path(__file__).name and summary["provenance"]["code_sha256"].get(path) != digest:
                    raise ValueError(f"Three-model aggregate implementation conflict: {path}")
            for name in ORIGINAL_MODELS:
                if any(summary["models"][name].get(key) != value for key, value in old["models"][name].items()) or old["protocols"][name] != summary["protocols"][name]:
                    raise ValueError(f"Completed three-model aggregate changed: {name}")
            for name, values in old["paired_comparisons"].items():
                if any(values[key] != summary["paired_comparisons"][name][key] for key in ("differences", "statistics", "lower_error_seed_counts")):
                    raise ValueError(f"Completed paired statistics changed: {name}")
            old_text = {evaluation / "multiseed_summary.txt": text(old), evaluation / "per_seed_metrics.csv": table(ORIGINAL_MODELS)}
            for path, value in old_text.items():
                if path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") != value.replace("\r\n", "\n"):
                    raise ValueError(f"Conflicting three-model aggregate text/table: {path}")
            archive = evaluation / "aggregate_history/three_models"
            archive.mkdir(parents=True, exist_ok=True)
            for path in (json_path, *old_text):
                if path.exists():
                    target = archive / path.name
                    raw = path.read_bytes()
                    if target.exists() and target.read_bytes().replace(b"\r\n", b"\n") != raw.replace(b"\r\n", b"\n"):
                        raise ValueError(f"Conflicting archived aggregate: {target}")
                    if not target.exists():
                        with target.open("xb") as handle:
                            handle.write(raw)
            logging.info("Verified three-model aggregate archived; updating derived four-model tables only.")
            upgrade = True
    for path, value in text_outputs.items():
        if not upgrade and path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") != value.replace("\r\n", "\n"):
            raise ValueError(f"Existing aggregate output conflicts: {path}")
    if upgrade:
        json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    elif not json_path.exists():
        save_json(json_path, summary)
    for path, value in text_outputs.items():
        if upgrade or not path.exists():
            with path.open("w" if upgrade else "x", encoding="utf-8", newline="") as handle:
                handle.write(value)


def run(args: argparse.Namespace) -> None:
    requested = args.seeds or ([42, 123] if args.smoke_test else list(SEEDS))
    if len(set(requested)) != len(requested) or any(seed not in SEEDS for seed in requested):
        raise ValueError(f"Only predeclared seeds {SEEDS} are permitted; duplicates are invalid.")
    if args.smoke_test and requested != [42, 123]:
        raise ValueError("Smoke uses exactly the first two declared seeds: 42 123.")
    targets = [42, 123] if args.smoke_test else list(SEEDS)
    protected = protected_snapshot()
    try:
        context = prepare(args)
        assert_unchanged(protected)
        protected.update(protected_snapshot())  # Also protect newly restored ignored artifacts.
        artifact_root, evaluation_root = output_paths(None)
        if args.smoke_test:
            suffix = Path("_smoke/multiseed/max_32/proposed/horizon_15/run_1")
            artifact_root, evaluation_root = ROOT / "artifacts/residual_learning" / suffix, ROOT / "evaluation/residual_learning" / suffix
        else:
            artifact_root, evaluation_root = artifact_root / "multiseed", evaluation_root / "multiseed"
        records = {}
        for seed in targets:
            for name in MODELS:
                artifacts = artifact_root / f"seed_{seed}" / name
                evaluation = evaluation_root / f"seed_{seed}" / name
                if seed in requested or (evaluation / "summary.json").is_file():
                    records[(seed, name)] = run_replicate(context, seed, name, artifacts, evaluation, args.smoke_test)
                    assert_unchanged(protected)
        missing = [(seed, name) for seed in targets for name in MODELS if (seed, name) not in records]
        if missing:
            logging.info("Aggregate deferred until all predeclared runs exist; missing: %s", missing)
            return
        summary = aggregate(context, records, targets, args.smoke_test)
        assert_unchanged(protected)
        write_aggregate(evaluation_root, summary, records, targets)
        logging.info("Aggregate saved: %s; descriptive statistics only.", evaluation_root)
    finally:
        assert_unchanged(protected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", help="Schedule a subset of the five declared seeds; full aggregation still requires all five.")
    parser.add_argument("--processed-csv", type=Path, help="Forwarded to existing recovery only when extraction files are missing.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
