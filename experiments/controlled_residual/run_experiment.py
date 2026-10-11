"""Fixed paired correction training; no test data, resume or implicit overwrites."""

import argparse
import json
from importlib.metadata import version
import time
from pathlib import Path
import torch
from torch import nn
from models.proposed_rvqc.batch_order import batches
from models.proposed_rvqc.cache import CachedSplit
from models.proposed_rvqc.data import ROOT, file_hash
from training.metrics import compute_metrics
from utils.logger import get_logger
from .cache import prepare
from .model import ARMS, Correction

EXPERIMENT_ROOT = ROOT / "experiments/controlled_residual/horizon_15"


def protocol():
    return {
        "version": 1, "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-5,
        "batch_size": 256, "max_epochs": 100, "patience": 15, "gradient_clip_norm": 1.0,
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-6},
        "selection": "lowest_validation_standardized_final_forecast_MSE",
        "epoch_indexing": "0=untrained, 1..100=completed training epochs", "resume": False,
        "projection_initialization": "Uniform(-1/sqrt(128),1/sqrt(128)); private seed",
        "readout_initialization": "all weights and biases zero",
        "quantum_initialization": "Uniform(-0.1,0.1); private seed+10000",
        "classical_transform_initialization": "Uniform(-1/sqrt(6),1/sqrt(6)); private seed+11000",
        "permutation": "private seed+20000+zero_based_training_epoch",
        "torch_version": str(torch.__version__), "pennylane_version": version("pennylane"),
        "rotation_primitives_sha256": file_hash(ROOT / "models/proposed_qtm/quantum_simulator.py"),
        "module_hashes": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    return torch.device(name)


@torch.no_grad()
def validate(correction, cache, device):
    if correction is not None:
        correction.eval()
    predictions, targets = [], []
    for start in range(0, len(cache), 256):
        pooled, base, target = cache[slice(start, start + 256)]
        prediction = base if correction is None else base.to(device) + correction(pooled.to(device))
        predictions.append(prediction.cpu())
        targets.append(target)
    prediction, target = torch.cat(predictions), torch.cat(targets)
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite validation forecast.")
    loss = nn.functional.mse_loss(prediction, target).item()
    return loss, compute_metrics(prediction.numpy(), target.numpy())


def improved(loss, best):
    return loss < best


def train_arm(arm, train, validation, directory, settings, device, epochs=100, patience=15):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    logger = get_logger(directory)
    model = Correction(arm, settings["seed"], settings["backend"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    settings = {**settings, "arm": arm, "training_complete": False,
                "trainable_parameters": sum(p.numel() for p in model.parameters()), "resume_supported": False}
    (directory / "settings.json").write_text(json.dumps(settings, indent=2))
    start = time.perf_counter()
    best, metrics = validate(model, validation, device)
    base_loss, _ = validate(None, validation, device)
    if best != base_loss:
        raise AssertionError("Epoch-0 forecast differs from unchanged baseline.")
    selected_epoch, stale = 0, 0
    history = [{"epoch": 0, "train_loss": None, "val_loss": best, **metrics}]

    def save(name, epoch):
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best, "selected_epoch": selected_epoch, "history": history,
                    "settings": settings}, checkpoint_dir / name)

    save("best_checkpoint.pt", 0)
    save("last_checkpoint.pt", 0)
    (directory / "history.json").write_text(json.dumps(history, indent=2))
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for indices in batches(len(train), settings["seed"], epoch - 1):
            pooled, base, target = train[indices]
            pooled, base, target = pooled.to(device), base.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(base + model(pooled), target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(indices)
        val_loss, metrics = validate(model, validation, device)
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"], **metrics})
        scheduler.step()
        if improved(val_loss, best):
            best, selected_epoch, stale = val_loss, epoch, 0
            save("best_checkpoint.pt", epoch)
        else:
            stale += 1
        save("last_checkpoint.pt", epoch)
        (directory / "history.json").write_text(json.dumps(history, indent=2))
        logger.info("Epoch %d | train MSE %.7f | val MSE %.7f | RMSE %.7f | MAE %.7f | selected %d",
                    epoch, total / len(train), val_loss, metrics["rmse"], metrics["mae"], selected_epoch)
        if stale >= patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    settings.update({"training_complete": True, "selected_epoch": selected_epoch,
                     "no_correction_selected": selected_epoch == 0, "best_val_loss": best,
                     "training_seconds": time.perf_counter() - start, "epochs_completed": history[-1]["epoch"],
                     "best_checkpoint_sha256": file_hash(checkpoint_dir / "best_checkpoint.pt"),
                     "last_checkpoint_sha256": file_hash(checkpoint_dir / "last_checkpoint.pt")})
    (directory / "settings.json").write_text(json.dumps(settings, indent=2))
    logger.info("Selected epoch %d%s", selected_epoch, " (no correction selected)" if selected_epoch == 0 else "")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    return settings


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("torch", "pennylane"), default="torch")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--build-cache-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    if args.backend == "pennylane" and device.type != "cpu":
        raise ValueError("Use --device cpu for the PennyLane reference backend.")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    directory = EXPERIMENT_ROOT / f"run_{args.run}"
    if not args.build_cache_only and directory.exists():
        raise FileExistsError(f"Refusing overwrite/resume: {directory}")
    from models.proposed_rvqc.data import WINDOW_COUNTS
    from .cache import CACHE_ROOT
    print(json.dumps({"arms": ["unchanged baseline", *ARMS], "seed": 41 + args.run,
                      "correction_training_runs": 0 if args.build_cache_only else 2,
                      "cache_status": "validating existing" if (CACHE_ROOT / f"run_{args.run}").exists() else "being generated",
                      "cache_path": str(CACHE_ROOT / f"run_{args.run}"), "device": str(device),
                      "backend": args.backend, "sample_counts": WINDOW_COUNTS,
                      "checkpoint": str(ROOT / f"experiments/proposed/horizon_15/run_{args.run}/checkpoints/best_checkpoint.pt")}, indent=2), flush=True)
    # Verify the optimized circuit before cache generation or training.
    from .preflight import reference_agreement
    agreement = reference_agreement(device)
    cache_dir, identity, baseline, status, checks = prepare(args.processed_csv, args.run, device, build=True)
    settings = {"experiment": "controlled_residual", "run": args.run, "seed": 41 + args.run,
                "backend": args.backend, "device": str(device), "protocol": protocol(),
                "cache_identity": identity, "cache_path": str(cache_dir), "cache_live_checks": checks,
                "reference_agreement": agreement, "source_checkpoint": str(baseline.path),
                "frozen_backbone_parameters": sum(p.numel() for p in baseline.parameters()),
                "frozen_backbone_trainable_parameters": 0}
    print(json.dumps({"arms": ["A: unchanged pretrained", "B: classical", "C: quantum"],
                      "run": args.run, "seed": settings["seed"], "correction_training_runs": 0 if args.build_cache_only else 2,
                      "cache_status": status, "cache_path": str(cache_dir), "checkpoint": str(baseline.path),
                      "device": str(device), "backend": args.backend,
                      "train_samples": identity["provenance"]["window_counts"]["train"],
                      "validation_samples": identity["provenance"]["window_counts"]["validation"]}, indent=2), flush=True)
    if args.build_cache_only:
        return
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "settings.json").write_text(json.dumps(settings, indent=2))
    before = {k: v.clone() for k, v in baseline.state_dict().items()}
    train, validation = CachedSplit(cache_dir, "train"), CachedSplit(cache_dir, "validation")
    baseline_loss, baseline_metrics = validate(None, validation, device)
    (directory / "baseline_validation.json").write_text(json.dumps({"mse": baseline_loss, **baseline_metrics}, indent=2))
    for arm in ARMS:
        train_arm(arm, train, validation, directory / arm, settings, device)
    if any(not torch.equal(v, baseline.state_dict()[k]) for k, v in before.items()):
        raise AssertionError("Frozen backbone state changed.")
    if any(m.training for m in baseline.modules()):
        raise AssertionError("Frozen backbone left evaluation mode.")


if __name__ == "__main__":
    main()
