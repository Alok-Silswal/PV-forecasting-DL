"""Evaluate a selected Proposed-RVQC checkpoint; never train or refit scalers."""

import argparse
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from configs import config
from evaluation.evaluator import Evaluator
from experiments.residual_learning.extract_residual_dataset import (
    RECOVERY_TARGET_ATOL, RECOVERY_TARGET_RMSE,
)
from experiments.residual_learning.run_residual_audit import forecast_metrics, deltas
from models.proposed_rvqc import ProposedRVQC
from . import run_experiment as training
from .run_experiment import (
    ROOT, FEATURES, TRAIN_END, VAL_END, EXPECTED_ROWS, WindowDataset,
    validate_frame, fingerprint, sha256, save_json, resolve_processed_csv,
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


def aggregate(evaluation_root: Path) -> None:
    paths = [evaluation_root/f"run_{run}/summary.json" for run in range(1,config.NUM_RUNS+1)]
    if not all(p.is_file() for p in paths):
        return
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    cohort_settings = []
    for run, record in enumerate(records, 1):
        settings_path = training.TRAINING_ROOT/f'horizon_15/run_{run}/settings.json'
        if fingerprint(settings_path) != record['settings_sha256']:
            raise ValueError(f"Run {run} settings changed before aggregation.")
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
        if (record['run_number'] != run or record['seed'] != config.RANDOM_SEED+run-1
                or settings['smoke_test'] or settings['benchmark_only']):
            raise ValueError(f"Run {run} is not a valid full experimental replicate.")
        cohort_settings.append(settings)
    for key in ('configuration','source_sha256','fitting_csv_sha256','features','lookback','outputs',
                'stride','feature_scaler','target_scaler','split_rows','loss'):
        if any(settings[key] != cohort_settings[0][key] for settings in cohort_settings[1:]):
            raise ValueError(f"Five-run training/data protocols differ: {key}")
    def stats(values: list[float]) -> dict:
        return {"mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)),
                "median":float(np.median(values)),"minimum":float(np.min(values)),"maximum":float(np.max(values))}
    result: dict[str,Any] = {"runs":config.NUM_RUNS,"summary_sha256":{str(i+1):fingerprint(p) for i,p in enumerate(paths)}}
    for name in ("baseline","proposed_rvqc","paired_delta"):
        result[name] = {}
        for scope in ("aggregate","0","1","2"):
            result[name][scope] = {}
            for metric in ("rmse","mae","r2","nrmse"):
                def value(record: dict, model: str) -> float:
                    m = record['metrics'][model]
                    return m['aggregate'][metric] if scope=='aggregate' else m['per_output'][scope][metric]
                values = [(value(r,'proposed_rvqc')-value(r,'baseline')) if name=='paired_delta' else value(r,name) for r in records]
                result[name][scope][metric] = stats(values)
    result["runs_improved"] = {key:sum(r['paired_deltas']['aggregate'][key]['delta']<0 for r in records) for key in ('rmse','mae')}
    result["percentage_improvement"] = {key:stats([r['paired_deltas']['aggregate'][key]['improvement_percent'] for r in records]) for key in ('rmse','mae')}
    path=evaluation_root/'five_run_summary.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=result:
            raise ValueError("Conflicting existing five-run summary.")
    else:
        save_json(path,result)



def run(args: argparse.Namespace) -> None:
    suffix = Path(f"horizon_15/run_{args.run_number}")
    if args.smoke_test:
        suffix = Path("_smoke") / suffix
    destination = training.TRAINING_ROOT / suffix
    checkpoint = destination / "checkpoints/best_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint missing; run training first: {checkpoint}")
    settings_path = destination / "settings.json"
    history_path = destination / "training_history.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    completion = json.loads(history_path.read_text(encoding="utf-8"))["completion"]
    if (settings["run_number"] != args.run_number or settings["smoke_test"] != args.smoke_test
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
    output = training.EVALUATION_ROOT / suffix
    baseline_root = ROOT / f"evaluation/proposed/horizon_15/run_{args.run_number}/results"
    provenance = {"settings_sha256": fingerprint(settings_path), "checkpoint_sha256": sha256(checkpoint),
                  "training_history_sha256": fingerprint(history_path), "run_number": args.run_number,
                  "seed": settings["seed"], "test_accessed": not args.smoke_test}
    if not args.smoke_test:
        provenance["baseline_sources_sha256"] = {name: fingerprint(baseline_root / name)
            for name in ("predictions.csv", "evaluation_metrics.json")}
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (any(summary.get(key) != value for key, value in provenance.items())
                or sha256(output / "predictions.npz") != summary["predictions_sha256"]):
            raise ValueError("Completed evaluation provenance conflicts.")
        logging.info("Existing evaluation verified; no retraining or overwrite.")
        if not args.smoke_test:
            aggregate(output.parent)
        return
    if output.exists() and any(p.is_file() for p in output.rglob("*")):
        raise FileExistsError(f"Partial evaluation requires explicit cleanup: {output}")
    protected = training.historical_snapshot()
    protected.update({baseline_root / name: sha256(baseline_root / name)
                      for name in ("predictions.csv", "evaluation_metrics.json")} if not args.smoke_test else {})
    started = time.perf_counter()
    try:
        frame = load_evaluation_frame(csv_path, settings, args.smoke_test)
        feature_scaler = restore_scaler(settings["feature_scaler"], FEATURES)
        target_scaler = restore_scaler(settings["target_scaler"])
        features = feature_scaler.transform(frame[FEATURES]).astype(np.float32)
        targets = target_scaler.transform(frame.Active_Power.to_numpy().reshape(-1, 1))[:, 0].astype(np.float32)
        dataset = WindowDataset(features, targets)
        result = Evaluator(ProposedRVQC(), DataLoader(dataset, batch_size=settings["configuration"]["BATCH_SIZE"]),
                           checkpoint, target_scaler, device).evaluate()
        corrected = result["predictions"].reshape(-1, 3)
        truth = result["targets"].reshape(-1, 3)
        summary = dict(provenance)
        if not args.smoke_test:
            historical = pd.read_csv(baseline_root / "predictions.csv")
            if list(historical.columns) != ["Actual", "Predicted"] or len(historical) != len(dataset) * 3:
                raise ValueError("Historical baseline sample/output contract differs.")
            saved_truth = historical.Actual.to_numpy().reshape(-1, 3)
            baseline = historical.Predicted.to_numpy().reshape(-1, 3)
            difference = truth.astype(np.float64) - saved_truth
            if (not np.isfinite(difference).all() or np.abs(difference).max() > RECOVERY_TARGET_ATOL
                    or np.sqrt(np.mean(difference ** 2)) > RECOVERY_TARGET_RMSE):
                raise ValueError("Test targets differ from authoritative baseline.")
            truth = saved_truth
            summary["target_equivalence_max_abs"] = float(np.abs(difference).max())
            summary["target_equivalence_rmse"] = float(np.sqrt(np.mean(difference ** 2)))
            summary["metrics"] = {"baseline": forecast_metrics(truth, baseline),
                                  "proposed_rvqc": forecast_metrics(truth, corrected)}
            metrics = summary["metrics"]
            summary["paired_deltas"] = {"aggregate": deltas(metrics["proposed_rvqc"]["aggregate"], metrics["baseline"]["aggregate"]),
                "per_output": {str(h): deltas(metrics["proposed_rvqc"]["per_output"][str(h)], metrics["baseline"]["per_output"][str(h)]) for h in range(3)}}
        else:
            summary["metrics"] = {"proposed_rvqc": forecast_metrics(truth, corrected)}
        indices = np.arange(len(dataset))
        arrays = {"sample_index": indices, "processed_input_start_row": (0 if args.smoke_test else VAL_END) + indices,
                  "y_true_original": truth, "prediction_proposed_rvqc": corrected}
        timestamps = frame.timestamp.to_numpy(dtype="datetime64[ns]")
        for name, shift in (("input_start_timestamp", 0), ("input_end_timestamp", 23),
                            ("target_0_timestamp", 24), ("target_1_timestamp", 25), ("target_2_timestamp", 26)):
            arrays[name] = timestamps[indices + shift]
        output.mkdir(parents=True, exist_ok=True)
        with (output / "predictions.npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        summary.update(predictions_sha256=sha256(output / "predictions.npz"),
                       samples=len(dataset), selected_epoch=completion["selected_epoch"],
                       runtime_seconds=time.perf_counter() - started)
        save_json(summary_path, summary)
        if not args.smoke_test:
            aggregate(output.parent)
        logging.info("Evaluation complete: %s (test accessed=%s); no training performed.", output, not args.smoke_test)
    finally:
        training.assert_protected(protected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-number", type=int, required=True, choices=range(1, config.NUM_RUNS + 1))
    parser.add_argument("--processed-csv", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Backbone device; RVQC stays CPU.")
    parser.add_argument("--smoke-test", action="store_true", help="Load disposable smoke checkpoint; evaluate eight non-test windows.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
