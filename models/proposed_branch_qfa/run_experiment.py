"""CPU-safe, cached A/B/C screening; never loads test data or trains a VQC."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from configs import config
from models.model_factory import get_model
from models.proposed_rvqc.data import file_hash, reconstruct
from models.proposed_rvqc.frozen_baseline import FrozenBaseline, validation_gate
from training.trainer import Trainer
from utils.logger import get_logger
from utils.seed import set_seed
from .cache import prepare_cache


class CachedBaseline(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        self.head = baseline.baseline.mlp_head
        self.checkpoint = baseline.checkpoint
        self.sha256 = baseline.sha256

    @torch.no_grad()
    def forward(self, packed):
        return packed[:, 0], self.head(packed[:, 0])


class ScreeningTrainer(Trainer):
    """Retain the initial pretrained function as a valid checkpoint candidate."""
    def train(self):
        loss, metrics = self._validate_one_epoch(-1)
        self.initial_metrics = {"mse": loss, **metrics, "epoch": -1}
        self.best_val_loss = loss
        if self.enable_checkpointing:
            self._save_checkpoint(-1)
        return super().train()


def loaders(datasets, seed):
    train = DataLoader(
        datasets["train"], batch_size=config.BATCH_SIZE,
        sampler=RandomSampler(datasets["train"], generator=torch.Generator().manual_seed(seed + 20000)),
        generator=torch.Generator().manual_seed(seed + 30000), num_workers=0)
    validation = DataLoader(datasets["validation"], batch_size=config.BATCH_SIZE, shuffle=False,
                            generator=torch.Generator().manual_seed(seed + 30000), num_workers=0)
    return train, validation


def experiment_directory(arm, seed):
    return (config.EXPERIMENTS_DIR / f"proposed_branch_qfa_{arm.lower()}" /
            "horizon_15" / f"run_{seed - 41}")


def train_arm(baseline, datasets, arm, seed, identity, gate, device):
    if arm not in ("A", "B", "C"):
        raise ValueError("Stage 0 supports A/B/C only.")
    directory = experiment_directory(arm, seed)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    set_seed(seed)
    model = get_model(f"proposed_branch_qfa_{arm.lower()}", baseline=baseline.baseline,
                      seed=seed, cached=True).to(device)
    probe = torch.stack([datasets["validation"][i][0] for i in range(min(4, len(datasets["validation"])))])
    model.assert_initial_equivalence(probe.to(device))
    frozen_before = {key: value.clone() for key, value in model.frozen.state_dict().items()}
    train_loader, val_loader = loaders(datasets, seed)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=config.SCHEDULER_MODE, factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE, min_lr=config.SCHEDULER_MIN_LR)
    directory.mkdir(parents=True)
    settings = {
        "arm": arm, "seed": seed, "device": str(device), "cache_identity": identity,
        "validation_gate": gate, "test_used": False, "training_complete": False,
        "protocol": {"optimizer": "Adam", "learning_rate": config.LEARNING_RATE,
                     "weight_decay": config.WEIGHT_DECAY, "batch_size": config.BATCH_SIZE,
                     "max_epochs": config.NUM_EPOCHS, "patience": config.EARLY_STOPPING_PATIENCE,
                     "gradient_clip": config.GRADIENT_CLIP_VALUE,
                     "scheduler": "ReduceLROnPlateau", "factor": config.SCHEDULER_FACTOR,
                     "scheduler_patience": config.SCHEDULER_PATIENCE, "min_lr": config.SCHEDULER_MIN_LR,
                     "sampler_seed_offset": 20000, "loader_seed_offset": 30000,
                     "initial_checkpoint_candidate": True, "selection": "lowest validation MSE"},
        "manual_criterion": {"rmse_improvement_percent": 0.3, "mae_not_worse": True,
                             "B_mae_warning_percent": 2.0},
        "source_hashes": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
    }
    settings_path = directory / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))
    logger = get_logger(directory)
    try:
        trainer = ScreeningTrainer(model, train_loader, val_loader, nn.MSELoss(), optimizer,
                                   scheduler, torch.device(device), logger, directory / "checkpoints",
                                   config.EARLY_STOPPING_PATIENCE,
                                   gradient_clip_value=config.GRADIENT_CLIP_VALUE, num_epochs=config.NUM_EPOCHS)
        history = trainer.train()
        if not all(torch.equal(value, model.frozen.state_dict()[key]) for key, value in frozen_before.items()):
            raise AssertionError("Frozen backbone parameters or buffers changed.")
        (directory / "history.json").write_text(json.dumps(history, indent=2))
        checkpoint = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=True)
        epoch = checkpoint["epoch"]
        if epoch == -1:
            metrics = trainer.initial_metrics
        else:
            metrics = {key: values[epoch] for key, values in history.items() if key != "train_loss"}
            metrics["mse"] = metrics.pop("val_loss")
            metrics["epoch"] = epoch
        results = directory / "results"
        results.mkdir()
        (results / "initial_validation_metrics.json").write_text(json.dumps(trainer.initial_metrics, indent=2))
        (results / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
        settings["training_complete"] = True
        settings["best_checkpoint_sha256"] = file_hash(trainer.checkpoint_path)
        settings_path.write_text(json.dumps(settings, indent=2))
        return metrics
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def comparison_summary(metrics):
    """Report effect sizes only; architecture decisions remain manual."""
    if "A" not in metrics:
        return {}
    base = metrics["A"]
    summary = {}
    for arm in ("B", "C"):
        if arm in metrics:
            result = metrics[arm]
            summary[arm] = {
                "rmse_improvement_percent": 100 * (base["rmse"] - result["rmse"]) / base["rmse"] if base["rmse"] else None,
                "mae_change_percent": 100 * (result["mae"] - base["mae"]) / base["mae"] if base["mae"] else None,
            }
    if "B" in summary and summary["B"]["mae_change_percent"] is not None and summary["B"]["mae_change_percent"] > 2:
        print("WARNING: Arm B validation MAE is more than 2% worse than A. Inspect calibration/training; no automatic rerun.")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(42, 47), default=config.BRANCH_QFA_SCREENING_SEEDS)
    parser.add_argument("--allow-multiple-seeds", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("A", "B", "C"), default=("A", "B", "C"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--build-cache-only", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) > 1 and not args.allow_multiple_seeds:
        parser.error("Multiple seeds require --allow-multiple-seeds; initial screening defaults to seed 42.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error("Seeds and arms must be unique.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    return args


def print_run_summary(args, datasets, baseline, cache_dir):
    runs = 0 if args.build_cache_only else len(args.arms) * len(args.seeds)
    status = "cached (pending verification)" if cache_dir.exists() else "being generated"
    print(f"Arms: {', '.join(args.arms)}\nSeeds: {', '.join(map(str, args.seeds))}\n"
          f"Total training runs: {runs} ({len(args.arms)} arms × {len(args.seeds)} seeds; cache-only={args.build_cache_only})\n"
          f"Backbone representations: {status}\nCheckpoint: {baseline.path} ({baseline.sha256})\n"
          f"Device: {args.device} (cache extraction: cpu)\nTrain samples: {len(datasets['train'])}\n"
          f"Validation samples: {len(datasets['validation'])}\nCache path: {cache_dir}", flush=True)


def main():
    args = parse_args()
    if config.ACTIVE_HORIZON != "15" or config.LOOKBACK != 24:
        raise ValueError("Requires the finalized 15-minute, 24-step configuration.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    if config.NUM_EPOCHS < 1:
        raise ValueError("Training budget must be positive.")
    torch.set_num_threads(args.threads)
    if not args.build_cache_only:
        for seed in args.seeds:
            for arm in args.arms:
                if experiment_directory(arm, seed).exists():
                    raise FileExistsError(f"Refusing to overwrite {experiment_directory(arm, seed)}")
    print(f"Planned training runs: {0 if args.build_cache_only else len(args.seeds) * len(args.arms)}; validation only.", flush=True)
    datasets, _, _, provenance = reconstruct(args.processed_csv)
    report = {}
    for seed in args.seeds:
        baseline = FrozenBaseline(seed - 41)
        cache_dir = config.ARTIFACT_DIR / "proposed_branch_qfa" / "cache" / baseline.sha256
        print_run_summary(args, datasets, baseline, cache_dir)
        cached, identity = prepare_cache(cache_dir, baseline, datasets, provenance, config.BATCH_SIZE)
        try:
            gate = validation_gate(CachedBaseline(baseline), cached["validation"], seed - 41)
            if not gate["passed"]:
                raise ValueError("Validation reconstruction does not reproduce pretrained checkpoint: " + json.dumps(gate))
            print("Backbone representations: cached and verified", flush=True)
            if args.build_cache_only:
                continue
            metrics = {}
            for arm in args.arms:
                metrics[arm] = train_arm(baseline, cached, arm, seed, identity, gate, args.device)
                print(f"seed={seed} arm={arm}: " + json.dumps(metrics[arm]), flush=True)
            report[str(seed)] = {"validation_metrics": metrics, "relative_to_A": comparison_summary(metrics)}
        finally:
            for split in cached.values():
                split.close()
    if report:
        report_path = config.EXPERIMENTS_DIR / "proposed_branch_qfa" / ("screening_" + "_".join(map(str, args.seeds)) + "_" + "".join(args.arms) + ".json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
