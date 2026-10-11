"""One test dataset, base forward and inverse scaler for all three arms."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.proposed_rvqc.data import ROOT, file_hash
from models.proposed_rvqc.frozen_baseline import FrozenBaseline
from training.metrics import compute_metrics
from .model import ARMS, Correction
from .run_experiment import EXPERIMENT_ROOT, select_device


def inverse(values, scaler):
    return scaler.inverse_transform(np.asarray(values, dtype=np.float64).reshape(-1, 1)).reshape(values.shape)


def report(prediction, target, correction):
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Expected shared forecasts and targets [N,3].")
    error = prediction - target
    metrics = compute_metrics(prediction, target)
    metrics.pop("mape")
    metrics["mse"] = float(np.mean(error ** 2))
    metrics["nrmse_denominator"] = float(np.max(target) - np.min(target))
    metrics["nrmse_definition"] = "aggregate RMSE / (max shared target - min shared target), all horizons flattened"
    metrics["horizons"] = [{"steps": i + 1, "minutes": 5 * (i + 1),
                            "mse": float(np.mean(error[:, i] ** 2)),
                            "rmse": float(np.sqrt(np.mean(error[:, i] ** 2))),
                            "mae": float(np.mean(np.abs(error[:, i])))} for i in range(3)]
    metrics["correction"] = {"mean": float(correction.mean()), "std": float(correction.std()),
                             "mean_absolute": float(np.abs(correction).mean()),
                             "rms": float(np.sqrt(np.mean(correction ** 2))),
                             "p95_absolute": float(np.quantile(np.abs(correction), 0.95)),
                             "max_absolute": float(np.abs(correction).max())}
    return metrics


def evaluate_arrays(base, corrections, target, scaler):
    if set(corrections) != set(ARMS):
        raise ValueError("Exactly classical and quantum corrections required.")
    original_target = inverse(target, scaler)
    predictions = {"baseline": inverse(base, scaler)}
    magnitudes = {"baseline": np.zeros_like(base, dtype=np.float64)}
    for arm in ARMS:
        if corrections[arm].shape != base.shape:
            raise ValueError("Correction shape differs from shared base forecast.")
        predictions[arm] = inverse(base + corrections[arm], scaler)
        # A correction is a difference: multiply by scale without adding the mean.
        magnitudes[arm] = corrections[arm].astype(np.float64) * float(scaler.scale_[0])
    if not all(np.isfinite(p).all() for p in (*predictions.values(), original_target)):
        raise ValueError("Nonfinite physical-scale predictions or targets.")
    results = {arm: report(pred, original_target, magnitudes[arm]) for arm, pred in predictions.items()}
    for arm in ARMS:
        results[arm]["relative_to_baseline"] = {}
        for metric in ("rmse", "mae"):
            baseline_metric = results["baseline"][metric]
            delta = results[arm][metric] - baseline_metric
            results[arm]["relative_to_baseline"][metric] = {
                "absolute_difference": delta,
                "percentage_difference": 100 * delta / baseline_metric if baseline_metric else None}
    return predictions, original_target, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    directory = EXPERIMENT_ROOT / f"run_{args.run}"
    output = ROOT / f"evaluation/controlled_residual/horizon_15/run_{args.run}"
    if output.exists():
        raise FileExistsError(f"Refusing evaluation overwrite: {output}")
    settings = json.loads((directory / "settings.json").read_text())
    if settings["run"] != args.run or settings["seed"] != 41 + args.run:
        raise ValueError("Mismatched experiment run/seed.")
    identity = settings["cache_identity"]
    baseline = FrozenBaseline(args.run, device=device)
    history = ROOT / f"experiments/proposed/horizon_15/run_{args.run}/history.json"
    if (baseline.sha256 != identity["checkpoint_sha256"] or not identity["validation_gate"]["passed"] or
            file_hash(history) != identity["validation_gate"]["history_sha256"] or
            file_hash(ROOT / "models/proposed_rvqc/frozen_baseline.py") != identity["frozen_extractor_sha256"]):
        raise ValueError("Source checkpoint/history differs from accepted validation gate.")
    models, records = {}, {}
    for arm in ARMS:
        record = json.loads((directory / arm / "settings.json").read_text())
        path = directory / arm / "checkpoints/best_checkpoint.pt"
        if (not record["training_complete"] or record["arm"] != arm or
                record["cache_identity"] != identity or record["seed"] != settings["seed"] or
                file_hash(path) != record["best_checkpoint_sha256"]):
            raise ValueError(f"Incomplete or changed {arm} training artifacts.")
        for name in ("model.py", "simulator.py"):
            if file_hash(Path(__file__).parent / name) != record["protocol"]["module_hashes"][name]:
                raise ValueError(f"Model implementation changed: {name}")
        if file_hash(ROOT / "models/proposed_qtm/quantum_simulator.py") != record["protocol"]["rotation_primitives_sha256"]:
            raise ValueError("Reused quantum rotation primitives changed after training.")
        if record["backend"] == "pennylane" and device.type != "cpu":
            raise ValueError("Evaluate a PennyLane reference run with --device cpu.")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["selected_epoch"] != record["selected_epoch"] or checkpoint["settings"]["cache_identity"] != identity:
            raise ValueError("Checkpoint selection/identity mismatch.")
        models[arm] = Correction(arm, settings["seed"], record["backend"]).to(device).eval()
        models[arm].load_state_dict(checkpoint["model_state_dict"], strict=True)
        records[arm] = record
    from models.proposed_rvqc.evaluate import load_test
    dataset, scaler = load_test(args.processed_csv, identity)
    bases, targets = [], []
    residuals = {arm: [] for arm in ARMS}
    with torch.no_grad():
        for inputs, target in DataLoader(dataset, batch_size=256, shuffle=False,
                                        generator=torch.Generator().manual_seed(0)):
            pooled, base = baseline(inputs.to(device))
            bases.append(base.cpu())
            targets.append(target)
            for arm, model in models.items():
                residuals[arm].append(model(pooled).cpu())
    base, target = torch.cat(bases).numpy(), torch.cat(targets).numpy()
    corrections = {arm: torch.cat(values).numpy() for arm, values in residuals.items()}
    predictions, original_target, results = evaluate_arrays(base, corrections, target, scaler)
    metadata = {"experiment": settings, "correction_training": records, "evaluation_device": str(device),
                "target_and_base_shared": True, "test_windows": len(dataset), "units": "original target units",
                "inverse_scaling": "one train-fitted scaler; float64 inverse for all arms",
                "test_used_for_selection": False, "delta_sign": "negative means improvement",
                "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                "base_sha256": hashlib.sha256(base.tobytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", targets=original_target, base_predictions=predictions["baseline"],
                        classical_predictions=predictions["classical"], quantum_predictions=predictions["quantum"],
                        normalized_targets=target, normalized_base_predictions=base,
                        normalized_classical_correction=corrections["classical"],
                        normalized_quantum_correction=corrections["quantum"])
    for arm, prediction in predictions.items():
        pd.DataFrame({"Actual": original_target.reshape(-1), "Predicted": prediction.reshape(-1)}).to_csv(output / f"{arm}_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
