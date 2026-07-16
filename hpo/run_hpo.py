"""
run_hpo.py

Entry point for the hyperparameter optimization (HPO) pipeline of the
short-term photovoltaic (PV) power forecasting project. Analogous to
``main.py`` for training and ``evaluate.py`` for evaluation, this
script only orchestrates existing components: it loads configuration
and artifacts, delegates the actual search to
``hpo.hpo_runner.run_hpo``, and persists the resulting trial history.

This script performs no optimization, training, or evaluation logic of
its own. Hyperparameter optimization is performed only for the
proposed model; there is no ``--model`` argument.
"""

import argparse
import importlib
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs import config
from hpo.hpo_runner import run_hpo
from utils.logger import get_logger


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments. ``args.optimizer`` selects which HPO
        algorithm to run. ``args.num_trials``, ``args.num_iterations``,
        and ``args.population_size`` are optional overrides for the
        selected optimizer's own defaults; each is ``None`` unless
        explicitly supplied, and only the ones relevant to the chosen
        optimizer are ever forwarded (see ``_build_optimizer_kwargs``).
    """

    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for the proposed "
        "PV power forecasting model."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=["random_search", "bayesian_optimization", "qs_bat", "qubo_sa"],
        help="Name of the HPO optimizer to run.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Number of trials to evaluate. Only used by random_search; "
        "if omitted, random_search's own default is used.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Number of search iterations. Used by "
        "bayesian_optimization, qs_bat, and qubo_sa; if omitted, the "
        "selected optimizer's own default is used.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Bat population size. Only used by qs_bat; if omitted, "
        "qs_bat's own default is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Name of the model to run HPO for (overrides "
            "config.MODEL_NAME for this run only). HPO currently "
            "targets the proposed model only; this option exists for "
            "CLI consistency with main.py and evaluate.py."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default=None,
        choices=config.HORIZON_TO_OUTPUT_DIM.keys(),
        help=(
            "Forecast horizon to run HPO for (overrides "
            "config.ACTIVE_HORIZON for this run only)."
        ),
    )
    return parser.parse_args()

def _apply_runtime_overrides(model_name: str, horizon: str) -> None:
    """
    Rebuild ``config.MODEL_NAME``, ``config.ACTIVE_HORIZON``, and every
    path derived from them for the current execution only.

    Always re-derives the full Model -> Forecast Horizon -> Artifacts
    hierarchy from ``config.MODEL_NAME`` and ``config.ACTIVE_HORIZON``,
    so that HPO artifacts for different forecast horizons never
    collide. The horizon segment always comes from
    ``config.ACTIVE_HORIZON`` and nowhere else (in particular, never
    from ``config.IS_KAGGLE``, which is reserved exclusively for
    dataset paths).

    Parameters
    ----------
    model_name : str
        Model name supplied via ``--model``.

    horizon : str
        Forecast horizon supplied via ``--horizon``.
    """

    config.MODEL_NAME = model_name
    config.ACTIVE_HORIZON = horizon

    config.HORIZON_DIR_NAME = f"horizon_{config.ACTIVE_HORIZON}"

    config.MODEL_EXPERIMENT_DIR = (
        config.EXPERIMENTS_DIR / config.MODEL_NAME / config.HORIZON_DIR_NAME
    )
    config.MODEL_EVALUATION_DIR = (
        config.EVALUATION_DIR / config.MODEL_NAME / config.HORIZON_DIR_NAME
    )

    config.CHECKPOINT_DIR = config.MODEL_EXPERIMENT_DIR / "checkpoints"
    config.BEST_CHECKPOINT_PATH = config.CHECKPOINT_DIR / "best_checkpoint.pt"

    config.HISTORY_FILE = config.MODEL_EXPERIMENT_DIR / "history.json"

    config.EVALUATION_RESULTS_DIR = config.MODEL_EVALUATION_DIR / "results"
    config.EVALUATION_PLOTS_DIR = config.MODEL_EVALUATION_DIR / "plots"

    config.PREDICTIONS_FILE = config.EVALUATION_RESULTS_DIR / "predictions.csv"
    config.EVALUATION_METRICS_FILE = (
        config.EVALUATION_RESULTS_DIR / "evaluation_metrics.json"
    )

def _build_optimizer_kwargs(args: argparse.Namespace) -> dict:
    """
    Build the optimizer-specific keyword arguments forwarded to
    ``hpo.hpo_runner.run_hpo`` as ``**optimizer_kwargs``.

    Every argument that was not explicitly supplied on the command line
    is omitted entirely, so the selected optimizer's own default value
    (defined in its own signature) applies unchanged. Only the
    argument(s) relevant to the selected optimizer are included, since
    each optimizer accepts a different set of keyword arguments (for
    example, ``random_search`` has no ``population_size`` parameter).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Keyword arguments to forward to the selected optimizer. May be
        empty.
    """

    optimizer_kwargs: dict = {}

    if args.optimizer == "random_search":
        if args.num_trials is not None:
            optimizer_kwargs["num_trials"] = args.num_trials

    elif args.optimizer in ("bayesian_optimization", "qubo_sa"):
        if args.num_iterations is not None:
            optimizer_kwargs["num_iterations"] = args.num_iterations

    elif args.optimizer == "qs_bat":
        if args.num_iterations is not None:
            optimizer_kwargs["num_iterations"] = args.num_iterations
        if args.population_size is not None:
            optimizer_kwargs["population_size"] = args.population_size

    return optimizer_kwargs


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


def _build_trainer_config(hpo_checkpoint_dir: Path) -> dict:
    """
    Build the non-tunable trainer settings shared by every HPO trial.

    Parameters
    ----------
    hpo_checkpoint_dir : Path
        Checkpoint directory passed to ``Trainer`` for every trial.
        Checkpointing itself is disabled inside
        ``hpo.objective.evaluate_hyperparameters``, so this directory is
        never actually written to; it is required only because
        ``Trainer`` expects a ``checkpoint_dir`` argument.

    Returns
    -------
    dict
        Dictionary with the keys required by
        ``hpo.objective.evaluate_hyperparameters``: ``"num_epochs"``,
        ``"early_stopping_patience"``, ``"gradient_clip_value"``, and
        ``"checkpoint_dir"``.
    """

    return {
        "num_epochs": config.NUM_EPOCHS,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "gradient_clip_value": config.GRADIENT_CLIP_VALUE,
        "checkpoint_dir": hpo_checkpoint_dir,
    }


def _json_default(value: object) -> object:
    """
    Fallback serializer for ``json.dump``, converting objects that are
    not natively JSON serializable but may appear inside the results
    dictionary returned by ``hpo.hpo_runner.run_hpo`` (for example,
    numpy scalar types or tensors nested inside trial metrics computed
    upstream by ``Trainer.train()``).

    Parameters
    ----------
    value : object
        The non-serializable object encountered by ``json.dump``.

    Returns
    -------
    object
        A JSON-serializable equivalent.

    Raises
    ------
    TypeError
        If ``value`` is of a type this function does not know how to
        convert, re-raised so ``json.dump`` reports the failure clearly
        instead of silently producing incorrect output.
    """

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


def _print_summary(optimizer_name: str, results: dict) -> None:
    """
    Print a concise HPO summary to the console.

    Parameters
    ----------
    optimizer_name : str
        Name of the optimizer that was run.

    results : dict
        Dictionary returned by ``hpo.hpo_runner.run_hpo``.
    """

    best_result = results["best_result"]
    best_hyperparameters = results["best_hyperparameters"]

    title = optimizer_name.replace("_", " ").title()

    print("=" * 40)
    print(f"{title} Completed")
    print("=" * 40)
    print(f"Best Validation Loss : {best_result['val_loss']:.6f}")
    print(f"Best Epoch           : {best_result['best_epoch']}")
    print(f"Elapsed Time         : {results['elapsed_time_seconds']:.2f} seconds")
    print()
    print("Best Hyperparameters")
    print("-" * 40)

    for name, value in best_hyperparameters.items():
        print(f"{name:<20}  : {value}")


def main() -> None:
    """
    Assemble the HPO experiment and run the requested optimizer.

    Loads configuration, sets the random seed, selects the device,
    builds the DataLoaders, builds the trainer configuration,
    instantiates the logger, dispatches to
    ``hpo.hpo_runner.run_hpo``, saves the resulting trial history to
    disk, and prints a concise summary.
    """

    args = _parse_args()

    _apply_runtime_overrides(
        args.model if args.model is not None else config.MODEL_NAME,
        args.horizon if args.horizon is not None else config.ACTIVE_HORIZON,
    )

    _set_seed(config.RANDOM_SEED)

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

    hpo_dir = config.MODEL_EXPERIMENT_DIR / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = _build_trainer_config(hpo_dir / "checkpoints")

    logger = get_logger(hpo_dir, log_file="hpo.log")

    optimizer_kwargs = _build_optimizer_kwargs(args)

    logger.info("Starting HPO with optimizer: %s", args.optimizer)

    results = run_hpo(
        optimizer_name=args.optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        trainer_config=trainer_config,
        logger=logger,
        **optimizer_kwargs,
    )

    results_path = hpo_dir / f"results_{args.optimizer}.json"

    if results_path.exists():
        print("Existing HPO results found.")
        print(f"Overwriting: {results_path}")

    logger.info("Saving HPO results to %s", results_path)

    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=4, default=_json_default)

    logger.info("HPO run complete: %s", args.optimizer)

    _print_summary(args.optimizer, results)


if __name__ == "__main__":
    main()