"""Recover the historical contract and extract purged validation blocks.

Run from project root with python -m experiments.residual_learning.extract_residual_dataset.
Existing forecasting artifacts are read only; no test data is extracted.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import platform
import pickle
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CSV_PATHS = (ROOT / "data/processed/DKASC_Preprocessed.csv",
                       Path("/kaggle/working/Processed.csv"))
FEATURES = [
    "Weather_Temperature_Celsius", "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation", "Diffuse_Horizontal_Radiation",
    "Radiation_Global_Tilted", "Radiation_Diffuse_Tilted", "Active_Power",
]
CHECKPOINT = ROOT / "experiments/proposed/horizon_15/run_1/checkpoints/best_checkpoint.pt"
TRAIN_END = 745702
EXPECTED_ROWS = 1065289
LOOKBACK, OUTPUTS, PURGE = 24, 3, 26
WARNINGS = [
    "Exploratory only: the historical validation period selected the backbone checkpoint.",
    "Historical timestamp gaps and pre-split interpolation are preserved; horizons are row based.",
    "No original test data is used for residual fitting, tuning, or assessment.",
    "Overlapping windows within blocks make samples statistically dependent.",
    "CPU subnormal flushing is enabled; previously checked against the historical checkpoint.",
]

# Bounds follow the measured independent-session recovery diagnostic; every
# mismatch also requires authoritative anchors and row-wise frozen-model replay.
RECOVERY_PREDICTION_ATOL = 1e-4
RECOVERY_PREDICTION_RMSE = 1e-5
RECOVERY_STAT_ATOL = 1e-6
RECOVERY_STAT_RTOL = 2e-7
RECOVERY_BIAS_ATOL = 1e-6
RECOVERY_TARGET_ATOL = 5e-5
RECOVERY_TARGET_RMSE = 1e-5
_VALIDATED_RECOVERIES: set[tuple] = set()


def compare_scaler_stats(actual: dict, expected: dict) -> None:
    """Allow measured floating drift, never a different sample count."""
    np.testing.assert_array_equal(actual["n_samples_seen_"], expected["n_samples_seen_"])
    latent = len(actual["mean_"]) == 128
    for key in ("mean_", "scale_", "var_"):
        np.testing.assert_allclose(actual[key], expected[key],
                                   atol=1e-9 if latent else RECOVERY_STAT_ATOL,
                                   rtol=1e-6 if latent else RECOVERY_STAT_RTOL)


def path_diagnostics(processed_csv: Path | None = None) -> str:
    """Describe the two notebook output paths without searching or writing files."""
    paths = [processed_csv.expanduser().resolve()] if processed_csv is not None else []
    paths += list(PROCESSED_CSV_PATHS)
    return "\n".join([
        f"Project root: {ROOT}", f"Current working directory: {Path.cwd()}",
        *[f"{path}: exists={path.exists()}, is_file={path.is_file()}" for path in paths],
        "Select an existing file with --processed-csv PATH; .pt artifacts are not substitutes.",
    ])


def resolve_processed_csv(processed_csv: Path | None = None) -> Path:
    """Honor an explicit file; otherwise require one unique known notebook output."""
    paths = ([processed_csv.expanduser().resolve()] if processed_csv is not None
             else list(PROCESSED_CSV_PATHS))
    matches = [path for path in paths if path.is_file()]
    if not matches:
        raise FileNotFoundError("Processed CSV missing.\n" + path_diagnostics(processed_csv))
    if len(matches) > 1:
        raise ValueError("Ambiguous processed CSV; use --processed-csv PATH.\n" + path_diagnostics())
    selected = matches[0].resolve()
    logging.info("Selected processed CSV: %s", selected)
    return selected


def sha256(path: Path) -> str:
    """Hash an input without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def require_new_files(paths: list[Path]) -> None:
    """Protect named outputs while allowing existing directories."""
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Incomplete experiment output already exists; refusing to overwrite: {path}")


def verify_recorded_file(path: Path, recorded: dict[str, str]) -> None:
    """Check a source snapshot, allowing only Git's JSON newline conversion."""
    matches = [value for key, value in recorded.items()
               if key.replace("\\", "/").rsplit("/", 1)[-1] == path.name]
    if len(matches) != 1:
        raise ValueError(f"Missing/ambiguous recorded hash for {path.name}.")
    hashes = {sha256(path)}
    if path.suffix == ".json":
        hashes.add(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    if matches[0] not in hashes:
        if path.suffix == ".npz" and path.name in {"train.npz", "tuning.npz", "assessment.npz"}:
            manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
            if matches[0] != manifest["partitions"][path.stem]["sha256"]:
                raise ValueError(f"Snapshot and manifest disagree: {path}")
            recover_extraction(path.parent, manifest)
            return
        if path.suffix == ".pkl" and path.stem in {"latent_scaler", "centered_residual_scaler"}:
            settings_path = path.parent / "settings.json"
            verify_recorded_file(settings_path, recorded)
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            scaler = pickle.loads(path.read_bytes())
            if (not isinstance(scaler, StandardScaler) or not scaler.with_mean or not scaler.with_std
                    or not scaler.copy or scaler.n_features_in_ != len(settings[path.stem]["mean_"])):
                raise ValueError(f"Invalid recovered scaler: {path}")
            compare_scaler_stats(scaler_stats(scaler), settings[path.stem])
            logging.warning("Numerically equivalent scaler; historical SHA retained: %s", path)
            return
        raise ValueError(f"Existing source conflicts with recorded SHA256: {path}")


def recover_extraction(directory: Path, manifest: dict, repair: bool = False,
                       processed_csv: Path | None = None) -> None:
    """Validate replicas against tracked anchors and a row-wise frozen-model replay.

    No candidate is published until every available check passes. Validation is
    cached only within this process and keyed by file hashes, never directory existence.
    """
    names = ("train", "tuning", "assessment")
    if processed_csv is None:
        recorded_csv = Path(manifest["source_processed_csv"])
        if not recorded_csv.is_absolute():
            recorded_csv = ROOT / recorded_csv
        if recorded_csv.is_file():
            processed_csv = recorded_csv
    csv_path = resolve_processed_csv(processed_csv)
    _, evaluation = output_paths(manifest["max_samples_per_partition"])
    settings_path = directory / "stage0_5/settings.json"
    if not settings_path.is_file():
        settings_path = ROOT / "artifacts/residual_learning/proposed/horizon_15/run_1/stage0_5/settings.json"
    bias_path = settings_path.parent / "bias.npy"
    assessment_path = evaluation / "assessment_predictions.npz"
    if sha256(CHECKPOINT) != manifest["checkpoint_sha256"]:
        raise ValueError(f"Recovery source SHA256 differs: {CHECKPOINT}")
    current_csv_sha = sha256(csv_path)
    csv_sha_differs = current_csv_sha != manifest["processed_csv_sha256"]
    if csv_sha_differs:
        if manifest["max_samples_per_partition"] is not None:
            raise ValueError(f"Recovery source SHA256 differs: {csv_path}")
        logging.warning(
            "Processed CSV byte SHA differs from historical manifest; proceeding only with full "
            "numerical/semantic recovery validation. Historical manifest will remain unchanged. "
            "historical=%s current=%s", manifest["processed_csv_sha256"], current_csv_sha)
    if manifest["max_samples_per_partition"] is not None:
        # Smoke manifests need not have later-stage anchors: exact historical
        # archive hashes remain sufficient, without granting numerical fallback.
        with tempfile.TemporaryDirectory(prefix="pv_residual_smoke_recovery_") as temporary:
            reference = Path(temporary)
            extract_residual_dataset(manifest["batch_size"], manifest["max_samples_per_partition"],
                                     manifest["seed"], manifest["cpu_threads"], csv_path,
                                     _destination=reference, _device=manifest["device"])
            for name in names:
                path = directory / f"{name}.npz"
                if sha256(reference / path.name) != manifest["partitions"][name]["sha256"]:
                    raise ValueError("Smoke recovery lacks matching anchors; exact historical hashes required.")
                if path.exists() and sha256(path) != manifest["partitions"][name]["sha256"]:
                    raise ValueError(f"Conflicting smoke artifact: {path}")
            if repair:
                for name in names:
                    path = directory / f"{name}.npz"
                    if not path.exists():
                        with path.open("xb") as handle:
                            handle.write((reference / path.name).read_bytes())
        return
    for path in (settings_path, bias_path, assessment_path):
        if not path.is_file():
            raise FileNotFoundError(f"Numerical recovery requires an authoritative anchor: {path}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if (settings["partitions"] != manifest["partitions"] or settings["scaler_fit_partition"] != "train"
            or settings["bias_fit_partition"] != "train"):
        raise ValueError("Recovery anchors belong to a different extraction.")
    signature = lambda: (str(directory.resolve()), json.dumps(manifest, sort_keys=True),
                         *[sha256(p) for p in (csv_path, CHECKPOINT, settings_path, bias_path, assessment_path)],
                         *[sha256(directory / f"{n}.npz") if (directory / f"{n}.npz").is_file() else None for n in names])
    if signature() in _VALIDATED_RECOVERIES:
        return
    with tempfile.TemporaryDirectory(prefix="pv_residual_recovery_") as temporary:
        reference_dir = Path(temporary)
        extract_residual_dataset(manifest["batch_size"], manifest["max_samples_per_partition"],
                                 manifest["seed"], manifest["cpu_threads"], csv_path,
                                 _destination=reference_dir, _device=manifest["device"])
        reference_manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
        for name in ("feature_scaler", "target_scaler"):
            for key, value in reference_manifest[name].items():
                np.testing.assert_allclose(value, manifest[name][key], atol=1e-12, rtol=1e-12)
        candidates = {}
        for name in names:
            reference = dict(np.load(reference_dir / f"{name}.npz", allow_pickle=False))
            candidate_path = directory / f"{name}.npz"
            candidate = dict(np.load(candidate_path, allow_pickle=False)) if candidate_path.is_file() else reference
            if set(candidate) != set(reference):
                raise ValueError(f"Recovery field contract differs: {name}")
            for key, values in reference.items():
                if candidate[key].shape != values.shape or candidate[key].dtype != values.dtype:
                    raise ValueError(f"Recovery shape/dtype differs: {name}/{key}")
                if key not in {"z", "y_hat_normalized", "residual_normalized", "y_hat_original", "residual_original"}:
                    np.testing.assert_array_equal(candidate[key], values, err_msg=f"{name}/{key}")
                else:
                    # Row-wise replay prevents permutations/corruption passing aggregate anchors.
                    if not np.isfinite(candidate[key]).all():
                        raise ValueError(f"Non-finite recovery field: {name}/{key}")
                    atol = (1e-7 if key == "z" else RECOVERY_PREDICTION_ATOL if key.endswith("original")
                            else RECOVERY_PREDICTION_ATOL / manifest["target_scaler"]["scale_"][0])
                    np.testing.assert_allclose(candidate[key], values, atol=atol, rtol=1e-6 if key == "z" else 0,
                                               err_msg=f"Frozen-model replay differs: {name}/{key}")
            for key in ("samples", "first_sample_index", "last_sample_index", "latent_shape", "input_start", "last_input_end", "first_target", "last_target"):
                if reference_manifest["partitions"][name][key] != manifest["partitions"][name][key]:
                    raise ValueError(f"Recovery partition boundary differs: {name}/{key}")
            # Check residual construction, not only independently plausible arrays.
            np.testing.assert_array_equal(candidate["residual_normalized"], candidate["y_true_normalized"] - candidate["y_hat_normalized"])
            np.testing.assert_allclose(candidate["residual_original"],
                                       candidate["residual_normalized"] * np.float64(manifest["target_scaler"]["scale_"][0]),
                                       atol=1e-12, rtol=1e-12)
            candidates[name] = candidate
        bias = np.load(bias_path, allow_pickle=False)
        train = candidates["train"]
        residual = train["y_true_original"].astype(np.float64) - train["y_hat_original"].astype(np.float64)
        difference = np.abs(residual.mean(axis=0) - bias)
        logging.info("Recovery training bias absolute differences: %s", difference.tolist())
        np.testing.assert_allclose(residual.mean(axis=0), bias, atol=RECOVERY_BIAS_ATOL, rtol=0)
        stored_difference = np.abs(train["residual_original"].astype(np.float64).mean(axis=0) - bias)
        logging.info("Recovery stored residual mean absolute differences: %s", stored_difference.tolist())
        np.testing.assert_allclose(train["residual_original"].astype(np.float64).mean(axis=0), bias,
                                   atol=RECOVERY_BIAS_ATOL, rtol=0)
        for name, values in (("latent_scaler", train["z"]), ("centered_residual_scaler", residual - bias)):
            actual = scaler_stats(StandardScaler().fit(values))
            logging.info("Recovery %s maximum statistic differences: %s", name,
                         {key: float(np.max(np.abs(np.asarray(value) - settings[name][key]))) for key, value in actual.items()})
            compare_scaler_stats(actual, settings[name])
        with np.load(assessment_path, allow_pickle=False) as anchor:
            data = candidates["assessment"]
            for key in ("sample_index", "input_start_timestamp", "input_end_timestamp", "target_0_timestamp", "target_1_timestamp", "target_2_timestamp"):
                np.testing.assert_array_equal(data[key], anchor[key])
            difference = data["y_true_original"].astype(np.float64) - anchor["y_true_original"].astype(np.float64)
            maximum, mean, rmse = float(np.abs(difference).max()), float(np.abs(difference).mean()), float(np.sqrt(np.mean(difference ** 2)))
            logging.info("Recovery assessment target differences: max=%g mean=%g RMSE=%g", maximum, mean, rmse)
            if (not np.isfinite(data["y_true_original"]).all() or not np.isfinite(anchor["y_true_original"]).all()
                    or not np.isfinite(difference).all() or maximum > RECOVERY_TARGET_ATOL or rmse > RECOVERY_TARGET_RMSE):
                raise ValueError("Assessment targets differ beyond numerical recovery bounds.")
            difference = data["y_hat_original"].astype(np.float64) - anchor["prediction_baseline"]
            maximum, mean, rmse = float(np.abs(difference).max()), float(np.abs(difference).mean()), float(np.sqrt(np.mean(difference ** 2)))
            logging.info("Recovery assessment prediction differences: max=%g mean=%g RMSE=%g", maximum, mean, rmse)
            if not np.isfinite(difference).all() or maximum > RECOVERY_PREDICTION_ATOL or rmse > RECOVERY_PREDICTION_RMSE:
                raise ValueError("Assessment baseline differs beyond numerical recovery bounds.")
        if repair:
            for name in names:
                path = directory / f"{name}.npz"
                if not path.exists():
                    with path.open("xb") as handle:
                        handle.write((reference_dir / path.name).read_bytes())
                    logging.info("NUMERICALLY RECOVERED %s; historical=%s current=%s; manifest unchanged",
                                 path, manifest["partitions"][name]["sha256"], sha256(path))
    logging.info("NUMERICALLY VALIDATED extraction replicas; historical manifest hashes unchanged: %s", directory)
    if csv_sha_differs:
        logging.info("Processed CSV accepted only through numerical/semantic validation, not byte identity; "
                     "historical=%s current=%s; historical manifest retained unchanged",
                     manifest["processed_csv_sha256"], current_csv_sha)
    _VALIDATED_RECOVERIES.add(signature())


def validate_extraction(directory: Path, manifest: dict, existing_only: bool = False) -> None:
    """Accept exact archives or require source/anchor/replay validation for replicas."""
    from experiments.residual_learning.run_residual_audit import load_partition

    for name in ("train", "tuning", "assessment"):
        if existing_only and not (directory / f"{name}.npz").exists():
            continue
        load_partition(directory, name, manifest)


def setup(seed: int, cpu_threads: int) -> torch.device:
    """Configure deterministic execution without changing global project config."""
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_paths(max_samples: int | None) -> tuple[Path, Path]:
    suffix = Path("proposed/horizon_15/run_1")
    if max_samples is not None:
        suffix = Path(f"_smoke/max_{max_samples}") / suffix
    return (ROOT / "artifacts/residual_learning" / suffix,
            ROOT / "evaluation/residual_learning" / suffix)


def scaler_stats(scaler: StandardScaler) -> dict[str, Any]:
    return {name: np.asarray(getattr(scaler, name)).tolist()
            for name in ("mean_", "scale_", "var_", "n_samples_seen_")}


def partition_starts(n_windows: int, max_samples: int | None) -> dict[str, np.ndarray]:
    """Use 60/20/20 boundaries, dropping 26 starts at each later block's start."""
    a, b = int(0.6 * n_windows), int(0.8 * n_windows)
    ranges = {"train": (0, a), "tuning": (a + PURGE, b),
              "assessment": (b + PURGE, n_windows)}
    result = {}
    for name, (lo, hi) in ranges.items():
        if hi <= lo:
            raise ValueError("Validation period is too short for three purged blocks.")
        if max_samples is not None:
            hi = min(hi, lo + max_samples)
        result[name] = np.arange(lo, hi, dtype=np.int64)
    names = list(result)
    for left, right in zip(names, names[1:]):
        assert result[left][-1] + LOOKBACK + OUTPUTS - 1 < result[right][0]
    return result


def classical_model() -> torch.nn.Module:
    """Instantiate only the verified classical architecture, with strict guards."""
    from configs import config
    from models.proposed_model import ProposedModel

    if (config.ACTIVE_HORIZON != "15" or config.FEATURE_COLUMNS != FEATURES
            or config.FEATURE_ATTENTION_REDUCTION != 8):
        raise ValueError("Project config drift: expected horizon 15, verified seven features, reduction 8.")
    return ProposedModel(
        dcnn_filters=64, dcnn_kernel_size=3, dcnn_dilation_rate=2,
        dcnn_dropout_rate=0.2, bilstm_hidden_size=64, bilstm_dropout_rate=0.2,
        mlp_hidden_dim=64, mlp_dropout_rate=0.2,
        use_feature_attention=True, use_temporal_attention=True,
        use_scalar_gated_fusion=True, use_quantum_branch=False,
    )


def extract_residual_dataset(batch_size: int = 256, max_samples: int | None = None,
                             seed: int = 42, cpu_threads: int = 1,
                             processed_csv: Path | None = None, *,
                             _destination: Path | None = None, _device: str | None = None) -> Path:
    """Extract the three validation partitions and write a completion manifest last."""
    if batch_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError("batch_size and max_samples must be positive.")
    destination, _ = output_paths(max_samples)
    if _destination is not None:
        destination = _destination
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    recorded = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if recorded is not None:
        if (recorded["schema_version"] != 1 or recorded["max_samples_per_partition"] != max_samples
                or recorded["features"] != FEATURES or recorded["input_length"] != LOOKBACK
                or recorded["output_length"] != OUTPUTS or recorded["stride"] != 1):
            raise ValueError("Authoritative extraction manifest differs from the requested contract.")
        if all((destination / f"{name}.npz").is_file() for name in ("train", "tuning", "assessment")):
            if any(sha256(destination / f"{name}.npz") != recorded["partitions"][name]["sha256"]
                   for name in ("train", "tuning", "assessment")):
                recover_extraction(destination, recorded, processed_csv=processed_csv)
            else:
                validate_extraction(destination, recorded)
            logging.info("Existing extraction artifacts verified: %s", destination)
            return destination
        recover_extraction(destination, recorded, repair=True, processed_csv=processed_csv)
        return destination
    csv_path = resolve_processed_csv(processed_csv)
    for path in (csv_path, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"Required historical input missing: {path}")
    started = time.perf_counter()
    device = setup(seed, cpu_threads)
    if _device is not None:
        device = torch.device(_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Recorded extraction CUDA device is unavailable.")
    logging.info("Loading historical processed CSV; device=%s", device)
    frame = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if len(frame) != EXPECTED_ROWS or int(0.7 * len(frame)) != TRAIN_END:
        raise ValueError("Processed-data row count/boundary differs from the recovered contract.")
    if frame[FEATURES].isna().any().any() or not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError("Historical feature data contains missing/non-finite values.")
    if frame.timestamp.isna().any() or not frame.timestamp.is_monotonic_increasing or frame.timestamp.duplicated().any():
        raise ValueError("Historical timestamps must be unique, sorted and valid.")
    val_end = TRAIN_END + int(0.15 * len(frame))
    feature_scaler = StandardScaler().fit(frame[FEATURES].iloc[:TRAIN_END])
    target_scaler = StandardScaler().fit(frame.Active_Power.iloc[:TRAIN_END].to_numpy().reshape(-1, 1))
    features = feature_scaler.transform(frame[FEATURES].iloc[TRAIN_END:val_end])
    targets = target_scaler.transform(frame.Active_Power.iloc[TRAIN_END:val_end].to_numpy().reshape(-1, 1))[:, 0]
    timestamps = frame.timestamp.iloc[TRAIN_END:val_end].to_numpy(dtype="datetime64[ns]")
    starts = partition_starts(len(features) - LOOKBACK - OUTPUTS + 1, max_samples)
    model = classical_model().to(device)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        if output.ndim != 3 or output.shape[1:] != (24, 128):
            raise ValueError(f"Unexpected fusion output shape: {tuple(output.shape)}")
        captured.append(output.detach())
        return None

    probe = torch.tensor(np.stack([features[i:i + LOOKBACK] for i in starts["train"][:min(8, len(starts['train']))]]), dtype=torch.float32, device=device)
    with torch.no_grad():
        ordinary = model(probe)
    hook = model.scalar_gated_fusion.register_forward_hook(capture)
    partition_manifest = {}
    try:
        with torch.no_grad():
            observed = model(probe)
            if not torch.equal(ordinary, observed):
                raise RuntimeError("Read-only fusion hook changed model predictions.")
            assert len(captured) == 1 and captured[0].mean(dim=1).shape == (len(probe), 128)
            captured.clear()
            for name, indices in starts.items():
                output_path = destination / f"{name}.npz"
                latent_batches, prediction_batches = [], []
                for offset in tqdm(range(0, len(indices), batch_size), desc=f"Extract {name}"):
                    batch_indices = indices[offset:offset + batch_size]
                    inputs = torch.tensor(np.stack([features[i:i + LOOKBACK] for i in batch_indices]), dtype=torch.float32, device=device)
                    predictions = model(inputs)
                    if len(captured) != 1 or predictions.shape != (len(inputs), 3):
                        raise RuntimeError("Expected one fusion capture and three outputs per sample.")
                    latent_batches.append(captured.pop().mean(dim=1).cpu().numpy())
                    prediction_batches.append(predictions.cpu().numpy())
                latent = np.concatenate(latent_batches)
                predicted = np.concatenate(prediction_batches)
                truth = np.stack([targets[i + LOOKBACK:i + LOOKBACK + OUTPUTS] for i in indices]).astype(np.float32)
                residual = truth - predicted
                original_truth = target_scaler.inverse_transform(truth.reshape(-1, 1)).reshape(-1, 3)
                original_prediction = target_scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(-1, 3)
                original_residual = residual * target_scaler.scale_[0]
                np.testing.assert_allclose(original_residual, original_truth - original_prediction, atol=5e-5, rtol=1e-5)
                data = {
                    "sample_index": indices, "processed_input_start_row": TRAIN_END + indices,
                    "z": latent, "y_true_normalized": truth, "y_hat_normalized": predicted,
                    "residual_normalized": residual, "y_true_original": original_truth,
                    "y_hat_original": original_prediction, "residual_original": original_residual,
                }
                for key, shift in [("input_start_timestamp", 0), ("input_end_timestamp", 23),
                                   ("target_0_timestamp", 24), ("target_1_timestamp", 25), ("target_2_timestamp", 26)]:
                    data[key] = timestamps[indices + shift]
                with io.BytesIO() as archive:
                    np.savez_compressed(archive, **data)
                    digest = hashlib.sha256(archive.getbuffer()).hexdigest()
                    if output_path.exists():
                        if sha256(output_path) != digest:
                            raise ValueError(f"Existing extraction without manifest conflicts with generated {name}: {output_path}")
                    else:
                        with output_path.open("xb") as handle:
                            handle.write(archive.getbuffer())
                partition_manifest[name] = {
                    "samples": len(indices), "first_sample_index": int(indices[0]),
                    "last_sample_index": int(indices[-1]), "latent_shape": list(latent.shape),
                    "input_start": str(data["input_start_timestamp"][0]),
                    "last_input_end": str(data["input_end_timestamp"][-1]),
                    "first_target": str(data["target_0_timestamp"][0]),
                    "last_target": str(data["target_2_timestamp"][-1]),
                    "sha256": digest,
                }
    finally:
        hook.remove()
        captured.clear()
    manifest = {
        "schema_version": 1,
        "source_processed_csv": csv_path.relative_to(ROOT).as_posix() if csv_path.is_relative_to(ROOT) else csv_path.as_posix(),
        "processed_csv_sha256": sha256(csv_path), "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(CHECKPOINT), "checkpoint_epoch_zero_based": checkpoint["epoch"],
        "run_number": 1, "features": FEATURES, "target": "Active_Power",
        "training_scaler_row_range": [0, TRAIN_END],
        "split_row_boundaries": {"train": [0, TRAIN_END], "validation": [TRAIN_END, val_end], "test": [val_end, len(frame)]},
        "input_length": LOOKBACK, "output_length": OUTPUTS, "stride": 1,
        "latent_extraction_module": "model.scalar_gated_fusion", "latent_shape": [128],
        "pooling": "detached fusion output.mean(dim=1)",
        "timestamp_mapping": {"sample_index": "window start relative to historical validation rows",
                              "row_offsets": [0, 23, 24, 25, 26]},
        "feature_scaler": scaler_stats(feature_scaler), "target_scaler": scaler_stats(target_scaler),
        "partition_policy": "60/20/20 of validation window starts; discard first 26 starts of tuning and assessment",
        "purged_start_ranges": [[int(0.6 * (len(features) - 26)), int(0.6 * (len(features) - 26)) + 26],
                                [int(0.8 * (len(features) - 26)), int(0.8 * (len(features) - 26)) + 26]],
        "partitions": partition_manifest, "max_samples_per_partition": max_samples,
        "seed": seed, "batch_size": batch_size, "cpu_threads": cpu_threads,
        "torch_version": str(torch.__version__), "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(), "cuda_available": torch.cuda.is_available(),
        "device": str(device), "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "deterministic_algorithms": True,
        "historical_reproduction_reference": {
            "source": "previous full test provenance audit, not rerun during validation extraction",
            "maximum_prediction_difference_original": 0.00087738037109375,
            "prediction_difference_rmse_original": 0.00004419742796242756,
        },
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "regression_checks": {"hook_predictions_exact": True, "pooled_latent_shape": [len(probe), 128],
                              "parameters_frozen": all(not p.requires_grad for p in model.parameters()),
                              "hook_removed": hook.id not in model.scalar_gated_fusion._forward_hooks},
        "warnings": WARNINGS,
    }
    save_json(destination / "manifest.json", manifest)
    logging.info("Extraction complete: %s", destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, help="Cap each block; writes only to _smoke/max_N.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--processed-csv", type=Path, help="Existing processed CSV; relative paths resolve from cwd.")
    parser.add_argument("--diagnose-paths", action="store_true", help="Print read-only path/artifact diagnostics and exit.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.diagnose_paths:
        print(path_diagnostics(args.processed_csv))
        return
    cap = args.max_samples if args.max_samples is not None else (1024 if args.smoke_test else None)
    extract_residual_dataset(args.batch_size, cap, args.seed, args.cpu_threads, args.processed_csv)


if __name__ == "__main__":
    main()
