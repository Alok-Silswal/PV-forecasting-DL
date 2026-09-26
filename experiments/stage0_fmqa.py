"""Stage 0 fidelity study for FMQA-based HPO of the Proposed Model.

This module deliberately stops before FM, QUBO, and simulated annealing.
It measures whether 25- and 50-epoch training rank configurations similarly
to the existing 100-epoch/15-patience training protocol.
"""

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs import config


CONFIGURATION_FILE = Path(__file__).with_name("stage0_configurations.json")
PARAMETER_NAMES = (
    "DCNN_FILTERS",
    "BILSTM_HIDDEN_SIZE",
    "DCNN_DROPOUT_RATE",
    "BILSTM_DROPOUT_RATE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "MLP_DROPOUT_RATE",
)
FIDELITIES = (25, 50, 100)
TOP_K_VALUES = (1, 3, 5)

SEARCH_LEVELS = {
    "DCNN_FILTERS": [32, 64, 96, 128],
    "BILSTM_HIDDEN_SIZE": [64, 128, 192, 256],
    "DCNN_DROPOUT_RATE": [0.0, 0.1, 0.2, 0.3, 0.5],
    "BILSTM_DROPOUT_RATE": [0.0, 0.1, 0.2, 0.3, 0.5],
    "LEARNING_RATE": [1e-4, 3e-4, 5e-4, 1e-3],
    "WEIGHT_DECAY": [0.0, 1e-6, 1e-5, 1e-4],
    "MLP_DROPOUT_RATE": [0.0, 0.1, 0.2, 0.3, 0.5],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        default="15",
        choices=tuple(config.HORIZON_TO_OUTPUT_DIM),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the Stage 0 integration and instantiate one model without training.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_configurations() -> list[dict]:
    with CONFIGURATION_FILE.open("r", encoding="utf-8") as configuration_file:
        configurations = json.load(configuration_file)

    if len(configurations) != 12:
        raise ValueError("Stage 0 must contain exactly 12 configurations.")

    expected_ids = [f"config_{index:03d}" for index in range(1, 13)]
    actual_ids = [item.get("configuration_id") for item in configurations]
    if actual_ids != expected_ids:
        raise ValueError("Stage 0 configuration IDs are invalid or unordered.")

    for item in configurations:
        missing = [name for name in PARAMETER_NAMES if name not in item]
        if missing:
            raise KeyError(f"{item['configuration_id']} is missing {missing}.")
        for name in PARAMETER_NAMES:
            if item[name] not in SEARCH_LEVELS[name]:
                raise ValueError(
                    f"{item['configuration_id']} has invalid {name}: {item[name]}"
                )

    return configurations


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)


def _build_loaders(horizon: str) -> tuple[DataLoader, DataLoader]:
    from main import _load_tensor_dataset, _resolve_dataset_paths

    config.ACTIVE_HORIZON = horizon
    train_path, val_path = _resolve_dataset_paths(horizon)
    train_dataset = _load_tensor_dataset(train_path)
    val_dataset = _load_tensor_dataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_TRAIN,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader


def _train_configuration(
    configuration: dict,
    requested_epochs: int,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> dict:
    from models.proposed_model import ProposedModel
    from training.loss import get_loss_function
    from training.trainer import Trainer
    from utils.logger import get_logger

    configuration_id = configuration["configuration_id"]
    fidelity_name = f"fidelity_{requested_epochs}"
    fidelity_dir = output_dir / configuration_id / fidelity_name
    checkpoint_dir = fidelity_dir / "checkpoints"
    fidelity_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)
    model = ProposedModel(
        dcnn_filters=configuration["DCNN_FILTERS"],
        dcnn_dropout_rate=configuration["DCNN_DROPOUT_RATE"],
        bilstm_hidden_size=configuration["BILSTM_HIDDEN_SIZE"],
        bilstm_dropout_rate=configuration["BILSTM_DROPOUT_RATE"],
        mlp_hidden_dim=config.MLP_HIDDEN_DIM,
        mlp_dropout_rate=configuration["MLP_DROPOUT_RATE"],
        use_quantum_branch=False,
    ).to(device)

    criterion = get_loss_function()
    optimizer = Adam(
        model.parameters(),
        lr=configuration["LEARNING_RATE"],
        weight_decay=configuration["WEIGHT_DECAY"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config.SCHEDULER_MODE,
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )
    logger = get_logger(fidelity_dir, log_file="training.log")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        gradient_clip_value=config.GRADIENT_CLIP_VALUE,
        num_epochs=requested_epochs,
    )

    start_time = time.perf_counter()
    history = trainer.train()
    elapsed_seconds = time.perf_counter() - start_time
    if not history["val_loss"]:
        raise RuntimeError(f"No validation losses recorded for {configuration_id}.")

    history_path = fidelity_dir / "history.json"
    _write_json(history_path, history)
    best_index = int(np.argmin(history["val_loss"]))
    return {
        "configuration_id": configuration_id,
        **{name: configuration[name] for name in PARAMETER_NAMES},
        "seed": seed,
        "requested_budget": requested_epochs,
        "actual_epochs_trained": len(history["val_loss"]),
        "best_validation_loss": float(history["val_loss"][best_index]),
        "final_validation_loss": float(history["val_loss"][-1]),
        "wall_clock_training_seconds": elapsed_seconds,
        "checkpoint_path": str(trainer.checkpoint_path),
        "history_path": str(history_path),
    }


def _rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def _spearman(first: list[float], second: list[float]) -> float:
    first_ranks = _rankdata(first)
    second_ranks = _rankdata(second)
    first_centered = first_ranks - np.mean(first_ranks)
    second_centered = second_ranks - np.mean(second_ranks)
    denominator = np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(first_centered * second_centered) / denominator)


def _top_k_preservation(
    cheap: dict[str, float], full: dict[str, float], k: int
) -> float:
    cheap_top = {
        item[0]
        for item in sorted(cheap.items(), key=lambda item: item[1])[:k]
    }
    full_top = {
        item[0]
        for item in sorted(full.items(), key=lambda item: item[1])[:k]
    }
    return len(cheap_top & full_top) / k


def _analyze(results: list[dict]) -> dict:
    by_fidelity = {
        budget: {
            item["configuration_id"]: item["best_validation_loss"]
            for item in results
            if item["requested_budget"] == budget
        }
        for budget in FIDELITIES
    }
    full = by_fidelity[100]
    correlations = {
        f"spearman_{budget}_vs_full": _spearman(
            [by_fidelity[budget][key] for key in full],
            [full[key] for key in full],
        )
        for budget in (25, 50)
    }
    top_k = {
        str(budget): {
            str(k): _top_k_preservation(by_fidelity[budget], full, k)
            for k in TOP_K_VALUES
        }
        for budget in (25, 50)
    }
    timing = {
        str(budget): {
            "total_seconds": float(
                sum(item["wall_clock_training_seconds"] for item in results
                    if item["requested_budget"] == budget)
            ),
            "mean_seconds": float(np.mean([
                item["wall_clock_training_seconds"] for item in results
                if item["requested_budget"] == budget
            ])),
        }
        for budget in FIDELITIES
    }
    return {
        "configuration_count": len(full),
        "fidelity_budgets": list(FIDELITIES),
        "spearman_rank_correlations": correlations,
        "top_k_preservation": top_k,
        "training_time_seconds": timing,
    }


def _write_results_csv(path: Path, results: list[dict]) -> None:
    fieldnames = list(results[0])
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _dry_run(horizon: str) -> None:
    """Validate Stage 0 wiring without loading test data or creating checkpoints."""

    configurations = _load_configurations()
    config.ACTIVE_HORIZON = horizon

    try:
        from main import _resolve_dataset_paths
        from models.proposed_model import ProposedModel
        from training.loss import get_loss_function
    except ModuleNotFoundError as error:
        if error.name == "pennylane":
            print(
                "Stage 0 dry-run could not instantiate ProposedModel: "
                "PennyLane is not installed in the active environment."
            )
            print(
                "Configuration validation passed; no training, test-set "
                "loading, or checkpoint creation was attempted."
            )
            return
        raise

    train_path, val_path = _resolve_dataset_paths(horizon)
    configuration = configurations[0]
    model = ProposedModel(
        dcnn_filters=configuration["DCNN_FILTERS"],
        dcnn_dropout_rate=configuration["DCNN_DROPOUT_RATE"],
        bilstm_hidden_size=configuration["BILSTM_HIDDEN_SIZE"],
        bilstm_dropout_rate=configuration["BILSTM_DROPOUT_RATE"],
        mlp_hidden_dim=config.MLP_HIDDEN_DIM,
        mlp_dropout_rate=configuration["MLP_DROPOUT_RATE"],
        use_quantum_branch=False,
    )
    criterion = get_loss_function()
    optimizer = Adam(
        model.parameters(),
        lr=configuration["LEARNING_RATE"],
        weight_decay=configuration["WEIGHT_DECAY"],
    )
    ReduceLROnPlateau(
        optimizer,
        mode=config.SCHEDULER_MODE,
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )

    print(f"Stage 0 dry-run passed for {configuration['configuration_id']}.")
    print(f"Train artifact: {train_path}")
    print(f"Validation artifact: {val_path}")
    print(f"Model class: {type(model).__name__}")
    print(f"use_quantum_branch: {model.use_quantum_branch}")
    print(f"Loss: {type(criterion).__name__}")
    print("No training, test-set loading, or checkpoint creation was performed.")


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or (
        config.EXPERIMENTS_DIR
        / "proposed"
        / f"horizon_{args.horizon}"
        / "stage0_fmqa"
    )
    configurations = _load_configurations()

    if args.dry_run:
        _dry_run(args.horizon)
        return

    if args.analyze_only:
        results_path = output_dir / "results.json"
        with results_path.open("r", encoding="utf-8") as results_file:
            results = json.load(results_file)["measurements"]
    else:
        from main import _select_device

        train_loader, val_loader = _build_loaders(args.horizon)
        device = _select_device()
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "configurations.json", configurations)
        results = []
        for configuration_index, configuration in enumerate(configurations):
            seed = config.RANDOM_SEED + configuration_index
            for requested_epochs in FIDELITIES:
                results.append(
                    _train_configuration(
                        configuration,
                        requested_epochs,
                        seed,
                        train_loader,
                        val_loader,
                        device,
                        output_dir,
                    )
                )
        _write_results_csv(output_dir / "results.csv", results)
        _write_json(output_dir / "results.json", {"measurements": results})

    analysis = _analyze(results)
    _write_json(output_dir / "rank_correlations.json", analysis)
    logging.getLogger(__name__).info("Stage 0 analysis written to %s", output_dir)


if __name__ == "__main__":
    main()