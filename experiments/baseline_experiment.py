"""
baseline_experiment.py

Baseline experiment analysis for short-term photovoltaic (PV) power
forecasting.

This script consumes ``experiments/history.json``, produced by
``main.py`` via ``training.trainer.Trainer.train()``, and produces:

* a learning-curve figure (training loss vs. validation loss)
* a baseline metrics summary extracted from the final epoch
* a concise console summary

It performs no training, no model or dataset instantiation, no checkpoint
loading, no metric computation, and no test-set evaluation. It only
analyzes a training history that has already been produced.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from configs import config

REQUIRED_KEYS = (
    "train_loss",
    "val_loss",
    "rmse",
    "mae",
    "mape",
    "r2",
    "nrmse",
)


def _load_history(history_path: Path) -> Dict[str, List[float]]:
    """
    Load the training history JSON file.

    Parameters
    ----------
    history_path : Path
        Path to ``history.json``, as produced by ``Trainer.train()``.

    Returns
    -------
    dict[str, list[float]]
        The parsed training history.

    Raises
    ------
    FileNotFoundError
        If ``history_path`` does not exist.
    """

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}. "
            f"Run main.py to produce it before running this script."
        )

    with open(history_path, "r", encoding="utf-8") as history_file:
        history: Dict[str, List[float]] = json.load(history_file)

    return history


def _validate_history(history: Dict[str, List[float]]) -> None:
    """
    Validate the structure and contents of the training history.

    Parameters
    ----------
    history : dict[str, list[float]]
        The parsed training history.

    Raises
    ------
    ValueError
        If the history is not a dictionary, required keys are missing,
        required values are not lists, list lengths are inconsistent, or
        the history is empty.
    """

    if not isinstance(history, dict):
        raise ValueError(
            f"Invalid history structure: expected a JSON object, "
            f"got {type(history).__name__}."
        )

    missing_keys = [key for key in REQUIRED_KEYS if key not in history]

    if missing_keys:
        raise ValueError(
            f"History file is missing required key(s): {missing_keys}."
        )

    non_list_keys = [
        key for key in REQUIRED_KEYS if not isinstance(history[key], list)
    ]

    if non_list_keys:
        raise ValueError(
            f"History key(s) must map to lists, but do not: {non_list_keys}."
        )

    lengths = {key: len(history[key]) for key in REQUIRED_KEYS}

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"History lists have inconsistent lengths: {lengths}."
        )

    if next(iter(lengths.values())) == 0:
        raise ValueError("History file is empty; no epochs were recorded.")


def _plot_learning_curve(
    history: Dict[str, List[float]],
    output_path: Path,
) -> None:
    """
    Plot training and validation loss curves and save the figure.

    Parameters
    ----------
    history : dict[str, list[float]]
        The parsed and validated training history.

    output_path : Path
        Destination path for the saved PNG figure.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
        color="#1f77b4",
        linewidth=1.8,
    )
    axis.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        color="#d62728",
        linewidth=1.8,
    )

    axis.set_title("Training and Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss (MSE)")
    axis.legend(loc="upper right")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _save_baseline_metrics(
    history: Dict[str, List[float]],
    output_path: Path,
) -> Dict[str, float]:
    """
    Extract final-epoch metrics and save them to disk.

    Parameters
    ----------
    history : dict[str, list[float]]
        The parsed and validated training history.

    output_path : Path
        Destination path for the saved metrics JSON file.

    Returns
    -------
    dict[str, float]
        The extracted final-epoch metrics.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_metrics: Dict[str, float] = {
        key: history[key][-1] for key in REQUIRED_KEYS
    }

    with open(output_path, "w", encoding="utf-8") as metrics_file:
        json.dump(final_metrics, metrics_file, indent=2)

    return final_metrics


def _print_summary(
    final_metrics: Dict[str, float],
    plot_path: Path,
    metrics_path: Path,
) -> None:
    """
    Print a concise console summary of the baseline experiment.

    Parameters
    ----------
    final_metrics : dict[str, float]
        Final-epoch metrics, as returned by ``_save_baseline_metrics``.

    plot_path : Path
        Path where the loss-curve figure was saved.

    metrics_path : Path
        Path where the baseline metrics JSON was saved.
    """

    print("Baseline Experiment Summary")
    print("---------------------------")
    print(f"Final Train Loss      : {final_metrics['train_loss']:.6f}")
    print(f"Final Validation Loss : {final_metrics['val_loss']:.6f}")
    print(f"RMSE                  : {final_metrics['rmse']:.6f}")
    print(f"MAE                   : {final_metrics['mae']:.6f}")
    print(f"MAPE                  : {final_metrics['mape']:.4f}%")
    print(f"R\u00b2                     : {final_metrics['r2']:.6f}")
    print(f"nRMSE                 : {final_metrics['nrmse']:.6f}")
    print()
    print(f"Loss curve saved to {plot_path}")
    print(f"Metrics saved to {metrics_path}")


def main() -> None:
    """
    Run the baseline experiment analysis.

    Loads and validates ``experiments/history.json``, generates the
    learning-curve figure, saves final-epoch baseline metrics, and prints
    a concise summary.
    """

    history = _load_history(config.HISTORY_FILE)

    _validate_history(history)

    plot_path = config.HISTORY_FILE.parent / "plots" / "loss_curve.png"
    metrics_path = (
        config.HISTORY_FILE.parent / "results" / "baseline_metrics.json"
    )

    _plot_learning_curve(history, plot_path)

    final_metrics = _save_baseline_metrics(history, metrics_path)

    _print_summary(final_metrics, plot_path, metrics_path)


if __name__ == "__main__":
    main()