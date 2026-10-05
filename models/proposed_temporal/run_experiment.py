"""Validation-only A/B/C screening using the finalized CSV split and Trainer."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from configs import config
from models.proposed_rvqc.data import reconstruct
from models.proposed_rvqc.frozen_baseline import FrozenBaseline, validation_gate
from training.trainer import Trainer
from utils.logger import get_logger
from utils.seed import set_seed
from .cache import CachedBaseline, CachedTemporalModel, prepare_cache


def train_arm(baseline, datasets, arm, seed, history, provenance, gate, device, cache_identity):
    run = seed - config.RANDOM_SEED + 1
    family = f"proposed_temporal_{arm.lower()}"
    directory = config.EXPERIMENTS_DIR / family / "horizon_15" / f"history_{history}" / f"run_{run}"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    set_seed(seed)
    model = CachedTemporalModel(baseline.baseline, arm=arm, history=history, seed=seed).to(device)
    probe = torch.stack([datasets["validation"][i][0] for i in range(4)]).to(device)
    model.assert_initial_equivalence(probe)
    frozen_before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    # Separate sampler and loader generators isolate ordering from model/dropout RNG.
    train_loader = DataLoader(
        datasets["train"], batch_size=config.BATCH_SIZE,
        sampler=RandomSampler(datasets["train"], generator=torch.Generator().manual_seed(seed + 20000)),
        generator=torch.Generator().manual_seed(seed + 30000), num_workers=0)
    val_loader = DataLoader(datasets["validation"], batch_size=config.BATCH_SIZE,
                            shuffle=False, generator=torch.Generator().manual_seed(seed + 30000),
                            num_workers=0)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=config.SCHEDULER_MODE, factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE, min_lr=config.SCHEDULER_MIN_LR)
    directory.mkdir(parents=True)
    settings = {"arm": arm, "seed": seed, "history": history,
                "baseline_sha256": baseline.sha256, "provenance": provenance,
                "representation_cache": cache_identity,
                "validation_gate": gate, "test_used": False, "training_complete": False,
                "protocol": {"optimizer": "Adam", "learning_rate": config.LEARNING_RATE,
                             "weight_decay": config.WEIGHT_DECAY, "batch_size": config.BATCH_SIZE,
                             "epochs": config.NUM_EPOCHS, "patience": config.EARLY_STOPPING_PATIENCE,
                             "gradient_clip": config.GRADIENT_CLIP_VALUE,
                             "scheduler": "ReduceLROnPlateau", "factor": config.SCHEDULER_FACTOR,
                             "scheduler_patience": config.SCHEDULER_PATIENCE,
                             "min_lr": config.SCHEDULER_MIN_LR, "selection": "validation MSE",
                             "torch_version": str(torch.__version__)}}
    settings_path = directory / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))
    trainer = Trainer(model, train_loader, val_loader, nn.MSELoss(), optimizer,
                      scheduler, torch.device(device), get_logger(directory),
                      directory / "checkpoints", config.EARLY_STOPPING_PATIENCE,
                      gradient_clip_value=config.GRADIENT_CLIP_VALUE, num_epochs=config.NUM_EPOCHS)
    training_history = trainer.train()
    if not all(torch.equal(v, model.backbone.state_dict()[k]) for k, v in frozen_before.items()):
        raise AssertionError("Frozen backbone parameters or buffers changed.")
    (directory / "history.json").write_text(json.dumps(training_history, indent=2))
    best = min(range(len(training_history["val_loss"])), key=training_history["val_loss"].__getitem__)
    metrics = {key: values[best] for key, values in training_history.items() if key != "train_loss"}
    metrics["mse"] = metrics.pop("val_loss")
    metrics["epoch"] = best
    results = directory / "results"
    results.mkdir()
    (results / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    settings["training_complete"] = True
    settings_path.write_text(json.dumps(settings, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=range(42, 47),
                        default=config.TEMPORAL_SCREENING_SEEDS)
    parser.add_argument("--arms", nargs="+", choices=("A", "B", "C"), default=("A", "B", "C"))
    parser.add_argument("--history", type=int, choices=(12, 24), default=config.TEMPORAL_HISTORY)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if config.ACTIVE_HORIZON != "15":
        raise ValueError("Screening requires the finalized 15-minute configuration.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        raise ValueError("Seeds and arms must be unique.")
    for seed in args.seeds:
        for arm in args.arms:
            destination = (config.EXPERIMENTS_DIR / f"proposed_temporal_{arm.lower()}" /
                           "horizon_15" / f"history_{args.history}" / f"run_{seed - 41}")
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
    datasets, _, _, provenance = reconstruct(args.processed_csv)
    report = {}
    for seed in args.seeds:
        run = seed - 41
        baseline = FrozenBaseline(run)
        cache_dir = (config.ARTIFACT_DIR / "proposed_temporal" / "cache" /
                     f"history_{args.history}" / baseline.sha256)
        cached_datasets, cache_identity = prepare_cache(
            cache_dir, baseline, datasets, args.history, provenance, config.BATCH_SIZE)
        gate = validation_gate(CachedBaseline(baseline), cached_datasets["validation"], run)
        if not gate["passed"]:
            raise ValueError("Reconstructed validation does not reproduce baseline: " + json.dumps(gate))
        for arm in args.arms:
            metrics = train_arm(baseline, cached_datasets, arm, seed, args.history,
                                provenance, gate, args.device, cache_identity)
            report[f"{seed}/{arm}"] = metrics
            print(f"seed={seed} arm={arm}: " + json.dumps(metrics))
    print(json.dumps({"validation_only": report}, indent=2))


if __name__ == "__main__":
    main()
