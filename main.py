"""
main.py

Entry point for the Short-Term Photovoltaic (PV) Power Forecasting project.

This script assembles all experiment dependencies (data, model, loss,
optimizer, scheduler, logger, trainer) via dependency injection and starts
training. It contains no architecture, preprocessing, or evaluation logic;
those responsibilities belong to their respective modules.
"""

import argparse
import json
import importlib
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from configs import config
from models.model_factory import get_model
from training.loss import get_loss_function
from training.trainer import Trainer
from utils.logger import get_logger
from experiments.baseline_experiment import run_baseline_analysis


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments. ``args.model`` is ``None`` when ``--model`` is
        omitted, in which case ``config.MODEL_NAME`` is used unchanged.
    """

    parser = argparse.ArgumentParser(
        description="Train a PV power forecasting model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Name of the model to train (overrides config.MODEL_NAME for "
            "this run only). Supported: cnn, lstm, cnn_lstm, "
            "dcnn_rbilstm, proposed, proposed_no_ta."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default=None,
        choices=config.HORIZON_TO_OUTPUT_DIM.keys(),
        help=(
            "Forecast horizon to train for (overrides "
            "config.ACTIVE_HORIZON for this run only)."
        ),
    )
    return parser.parse_args()


def _apply_runtime_overrides(model_name: str, horizon: str, run_number: int) -> None:
    """
    Rebuild config.MODEL_NAME, config.ACTIVE_HORIZON, config.RUN_NUMBER,
    and every path derived from them for the current execution only.

    Always re-derives the full
    Model -> Forecast Horizon -> Run -> Artifacts
    hierarchy from config.MODEL_NAME, config.ACTIVE_HORIZON, and
    config.RUN_NUMBER, so that no stale path from a previous run or
    horizon survives into the current one. The horizon segment always
    comes from config.ACTIVE_HORIZON and nowhere else (in particular,
    never from config.IS_KAGGLE, which is reserved exclusively for
    dataset paths).

    Parameters
    ----------
    model_name : str
        Model name supplied via --model.
    horizon : str
        Forecast horizon supplied via --horizon.
    run_number : int
        Run index in [1, config.NUM_RUNS] to configure paths for.
    """

    config.MODEL_NAME = model_name
    config.ACTIVE_HORIZON = horizon
    config.RUN_NUMBER = run_number

    config.HORIZON_DIR_NAME = f"horizon_{config.ACTIVE_HORIZON}"
    config.RUN_NAME = f"run_{config.RUN_NUMBER}"

    config.MODEL_EXPERIMENT_DIR = (
        config.EXPERIMENTS_DIR / config.MODEL_NAME / config.HORIZON_DIR_NAME
    )
    config.MODEL_EVALUATION_DIR = (
        config.EVALUATION_DIR / config.MODEL_NAME / config.HORIZON_DIR_NAME
    )

    config.RUN_EXPERIMENT_DIR = config.MODEL_EXPERIMENT_DIR / config.RUN_NAME
    config.RUN_EVALUATION_DIR = config.MODEL_EVALUATION_DIR / config.RUN_NAME

    config.AVERAGE_EXPERIMENT_DIR = config.MODEL_EXPERIMENT_DIR / "average"
    config.AVERAGE_EVALUATION_DIR = config.MODEL_EVALUATION_DIR / "average"

    config.ALL_RUN_EXPERIMENT_DIRS = [
        config.MODEL_EXPERIMENT_DIR / f"run_{i}"
        for i in range(1, config.NUM_RUNS + 1)
    ]
    config.ALL_RUN_EVALUATION_DIRS = [
        config.MODEL_EVALUATION_DIR / f"run_{i}"
        for i in range(1, config.NUM_RUNS + 1)
    ]

    config.CHECKPOINT_DIR = config.RUN_EXPERIMENT_DIR / "checkpoints"
    config.BEST_CHECKPOINT_PATH = config.CHECKPOINT_DIR / "best_checkpoint.pt"

    config.HISTORY_FILE = config.RUN_EXPERIMENT_DIR / "history.json"

    config.EVALUATION_RESULTS_DIR = config.RUN_EVALUATION_DIR / "results"
    config.EVALUATION_PLOTS_DIR = config.RUN_EVALUATION_DIR / "plots"

    config.PREDICTIONS_FILE = config.EVALUATION_RESULTS_DIR / "predictions.csv"
    config.EVALUATION_METRICS_FILE = (
        config.EVALUATION_RESULTS_DIR / "evaluation_metrics.json"
    )


def _set_seed(seed: int) -> None:
    """
    Set the random seed across all relevant libraries for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value, read from ``configs.config.RANDOM_SEED``.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # No-ops on CPU-only execution; kept for portability, matching the
    # project's CUDA-aware device selection.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_device() -> torch.device:
    """
    Select the execution device.

    Returns
    -------
    torch.device
        TPU (XLA) if available, otherwise CUDA, otherwise CPU.
    """

    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
        return xm.xla_device()
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _resolve_dataset_paths(horizon: str) -> Tuple[Path, Path]:
    """
    Resolve the train/validation artifact paths for the configured
    forecast horizon.

    Parameters
    ----------
    horizon : str
        Forecast horizon identifier, e.g. ``"15"`` or ``"60"``, matching
        the ``TRAIN_{horizon}_FILE`` / ``VAL_{horizon}_FILE`` constants in
        ``configs.config``.

    Returns
    -------
    tuple[Path, Path]
        Paths to the training and validation ``.pt`` artifacts.

    Raises
    ------
    ValueError
        If no dataset constants exist for the given horizon.
    FileNotFoundError
        If the resolved artifact paths do not exist on disk.
    """

    train_attr = f"TRAIN_{horizon}_FILE"
    val_attr = f"VAL_{horizon}_FILE"

    if not hasattr(config, train_attr) or not hasattr(config, val_attr):
        raise ValueError(
            f"Invalid forecast horizon '{horizon}'. No '{train_attr}' / "
            f"'{val_attr}' constants are defined in configs/config.py."
        )

    train_path: Path = getattr(config, train_attr)
    val_path: Path = getattr(config, val_attr)

    for dataset_path in (train_path, val_path):
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Required dataset artifact not found: {dataset_path}. "
                f"Run the preprocessing pipeline before starting training."
            )

    return train_path, val_path

def _load_tensor_dataset(path: Path) -> TensorDataset:
    """
    Load a preprocessed ``.pt`` artifact into a ``TensorDataset``.

    Parameters
    ----------
    path : Path
        Path to a ``.pt`` file containing a ``(inputs, targets)`` tensor
        pair, as produced by the preprocessing pipeline.

    Returns
    -------
    TensorDataset
        Dataset wrapping the loaded input and target tensors.
    """

    loaded = torch.load(path, map_location="cpu")

    if isinstance(loaded, dict):
        try:
            inputs = loaded["X"]
            targets = loaded["y"]
        except KeyError as exc:
            raise KeyError(
                f"Unsupported dataset artifact format in {path}. "
                "Expected keys 'X' and 'y'."
            ) from exc
    else:
        inputs, targets = loaded

    return TensorDataset(inputs, targets)

def _train_single_run() -> None:
    """
    Train and save history for the currently configured run.

    Assumes config paths have already been set for this run via
    _apply_runtime_overrides(). Resumes from config.CHECKPOINT_DIR /
    'best_checkpoint.pt' only if that exact run's checkpoint exists;
    never reuses another run's checkpoint, since CHECKPOINT_DIR is
    unique per run.
    """

    device = _select_device()

    train_path, val_path = _resolve_dataset_paths(config.ACTIVE_HORIZON)

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

    model = get_model(config.MODEL_NAME).to(device)

    criterion = get_loss_function()

    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config.SCHEDULER_MODE,
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )

    logger = get_logger(config.RUN_EXPERIMENT_DIR)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        checkpoint_dir=config.CHECKPOINT_DIR,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        gradient_clip_value=config.GRADIENT_CLIP_VALUE,
        num_epochs=config.NUM_EPOCHS,
    )

    if trainer.checkpoint_path.exists():
        trainer.load_checkpoint(trainer.checkpoint_path)

        if trainer.start_epoch >= config.NUM_EPOCHS:
            logger.info(
                "Checkpoint already reached configured num_epochs (%d). "
                "Skipping training.",
                config.NUM_EPOCHS,
            )
            return

    history = trainer.train()

    config.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(config.HISTORY_FILE, "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    logger.info("Training history saved to %s", config.HISTORY_FILE)

    run_baseline_analysis()

    logger.info(
        "Baseline experiment analysis complete for %s.",
        config.RUN_EXPERIMENT_DIR,
    )

def _is_run_complete() -> bool:
    """
    Check whether the currently configured run has already finished
    both training and evaluation.
    """

    # Legacy compatibility for existing run_1 results
    if config.RUN_NUMBER == 1:
        return (
            config.HISTORY_FILE.exists()
            and config.EVALUATION_METRICS_FILE.exists()
        )

    # Modern runs
    return (
        config.HISTORY_FILE.exists()
        and config.BEST_CHECKPOINT_PATH.exists()
        and config.EVALUATION_METRICS_FILE.exists()
    )

def _create_run_folders() -> None:
    """
    Ensure all directories for the currently configured run exist.

    Must be called after _apply_runtime_overrides() has set
    config.RUN_EXPERIMENT_DIR / config.RUN_EVALUATION_DIR (and
    everything derived from them) for the run being prepared.
    """

    config.RUN_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUN_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EVALUATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _compute_average_metrics() -> None:
    """
    Aggregate evaluation_metrics.json across all completed runs and
    write the averaged result to config.AVERAGE_EVALUATION_DIR.

    Reads config.ALL_RUN_EVALUATION_DIRS (already rebuilt for the
    current model/horizon by the final _apply_runtime_overrides()
    call), collects each run's 'evaluation_metrics.json', averages
    matching numeric keys, and writes 'average_metrics.json' to
    config.AVERAGE_EVALUATION_DIR.
    """

    metrics_list = []

    for run_eval_dir in config.ALL_RUN_EVALUATION_DIRS:
        metrics_path = run_eval_dir / "results" / "evaluation_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_list.append(json.load(f))

    if not metrics_list:
        return

    keys = metrics_list[0].keys()
    averaged = {
        key: float(np.mean([m[key] for m in metrics_list if key in m]))
        for key in keys
    }

    config.AVERAGE_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    average_path = config.AVERAGE_EVALUATION_DIR / "average_metrics.json"

    with open(average_path, "w", encoding="utf-8") as f:
        json.dump(averaged, f, indent=2)

def main() -> None:
    """
    Assemble the experiment and run training/evaluation across all
    configured runs, then compute averaged metrics.

    For each run in [1, config.NUM_RUNS]: rebuilds all run-scoped
    config paths, creates required folders, and trains + evaluates
    unless that run has already completed (history.json and
    evaluation_metrics.json both already exist). After all runs,
    aggregates metrics into config.AVERAGE_EVALUATION_DIR.
    """

    args = _parse_args()

    model_name = args.model if args.model is not None else config.MODEL_NAME
    horizon = args.horizon if args.horizon is not None else config.ACTIVE_HORIZON

    for run_number in range(1, config.NUM_RUNS + 1):
        _apply_runtime_overrides(model_name, horizon, run_number)

        if _is_run_complete():
            print(
                f"[SKIP] {config.MODEL_NAME}/{config.HORIZON_DIR_NAME}/"
                f"{config.RUN_NAME} already complete "
                f"(history, checkpoint, and metrics all found). Skipping."
            )
            continue

        run_seed = config.RANDOM_SEED + (run_number - 1)
        _set_seed(run_seed)
        _create_run_folders()
        _train_single_run()

    # Rebuild paths once more (run_number irrelevant here) so that
    # config.ALL_RUN_EVALUATION_DIRS / config.AVERAGE_EVALUATION_DIR
    # reflect the final model/horizon, not a stale value from the loop.
    _apply_runtime_overrides(model_name, horizon, config.NUM_RUNS)
    _compute_average_metrics()


if __name__ == "__main__":
    main()