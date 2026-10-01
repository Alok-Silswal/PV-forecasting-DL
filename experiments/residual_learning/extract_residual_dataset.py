"""Recover the historical contract and extract purged validation blocks.

Run from project root with python -m experiments.residual_learning.extract_residual_dataset.
Existing forecasting artifacts are read only; no test data is extracted.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import random
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
                             processed_csv: Path | None = None) -> Path:
    """Extract the three validation partitions and write a completion manifest last."""
    if batch_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError("batch_size and max_samples must be positive.")
    csv_path = resolve_processed_csv(processed_csv)
    for path in (csv_path, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"Required historical input missing: {path}")
    destination, _ = output_paths(max_samples)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite extraction directory: {destination}")
    started = time.perf_counter()
    device = setup(seed, cpu_threads)
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
            destination.mkdir(parents=True, exist_ok=False)
            for name, indices in starts.items():
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
                np.savez_compressed(destination / f"{name}.npz", **data)
                partition_manifest[name] = {
                    "samples": len(indices), "first_sample_index": int(indices[0]),
                    "last_sample_index": int(indices[-1]), "latent_shape": list(latent.shape),
                    "input_start": str(data["input_start_timestamp"][0]),
                    "last_input_end": str(data["input_end_timestamp"][-1]),
                    "first_target": str(data["target_0_timestamp"][0]),
                    "last_target": str(data["target_2_timestamp"][-1]),
                    "sha256": sha256(destination / f"{name}.npz"),
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
