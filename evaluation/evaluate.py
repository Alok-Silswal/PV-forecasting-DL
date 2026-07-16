"""
evaluate.py

Entry point for the evaluation pipeline of the short-term photovoltaic
(PV) power forecasting project. Analogous to ``main.py`` for training,
this script only orchestrates existing components: it loads
configuration and artifacts, delegates inference and metric computation
to ``evaluation.evaluator.Evaluator``, delegates figure generation to
``evaluation.plots.EvaluationPlotter``, and persists the resulting
predictions and metrics.

This script performs no inference, metric computation, or plotting of
its own.
"""

import csv
import importlib
import json
import pickle
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from configs import config
from evaluation.evaluator import Evaluator
from evaluation.plots import EvaluationPlotter
from models.model_factory import get_model


def _select_device() -> torch.device:
    """
    Select the device used for evaluation.

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


def _resolve_test_dataset_path() -> Path:
    """
    Resolve the test dataset artifact matching the configured forecast
    horizon.

    Returns
    -------
    Path
        Path to ``config.TEST_15_FILE`` or ``config.TEST_60_FILE``,
        depending on ``config.ACTIVE_HORIZON``.

    Raises
    ------
    ValueError
        If ``config.ACTIVE_HORIZON`` is not a recognized horizon.
    """

    horizon_to_file = {
        "15": config.TEST_15_FILE,
        "60": config.TEST_60_FILE,
    }

    if config.ACTIVE_HORIZON not in horizon_to_file:
        raise ValueError(
            f"Invalid forecast horizon: {config.ACTIVE_HORIZON!r}. "
            f"Expected one of {sorted(horizon_to_file)}."
        )

    return horizon_to_file[config.ACTIVE_HORIZON]


def _load_test_dataset(test_dataset_path: Path) -> Dataset:
    """
    Load the test dataset artifact from disk.

    Accepts a saved ``Dataset`` instance, a ``(inputs, targets)``
    tuple/list of tensors, or a dictionary with ``"inputs"`` and
    ``"targets"`` keys, and normalizes any of these to a
    ``torch.utils.data.Dataset``.

    Parameters
    ----------
    test_dataset_path : Path
        Path to the ``.pt`` file produced during preprocessing.

    Returns
    -------
    torch.utils.data.Dataset
        The test dataset.

    Raises
    ------
    FileNotFoundError
        If ``test_dataset_path`` does not exist.
    ValueError
        If the loaded object is not a recognized dataset format.
    """

    if not test_dataset_path.exists():
        raise FileNotFoundError(
            f"Test dataset file not found: {test_dataset_path}."
        )

    loaded = torch.load(test_dataset_path, weights_only=False)

    if isinstance(loaded, Dataset):
        return loaded

    if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
        inputs, targets = loaded
        return TensorDataset(inputs, targets)

    if isinstance(loaded, dict) and "inputs" in loaded and "targets" in loaded:
        return TensorDataset(loaded["inputs"], loaded["targets"])

    raise ValueError(
        f"Unrecognized test dataset format in {test_dataset_path}: "
        f"expected a Dataset, a (inputs, targets) tuple, or a dict with "
        f"'inputs'/'targets' keys; got {type(loaded).__name__}."
    )


def _load_target_scaler(target_scaler_path: Path):
    """
    Load the fitted target scaler used to inverse-transform predictions.

    Parameters
    ----------
    target_scaler_path : Path
        Path to ``target_scaler.pkl``.

    Returns
    -------
    Any
        The deserialized scaler object.

    Raises
    ------
    FileNotFoundError
        If ``target_scaler_path`` does not exist.
    """

    if not target_scaler_path.exists():
        raise FileNotFoundError(
            f"Target scaler file not found: {target_scaler_path}."
        )

    with open(target_scaler_path, "rb") as scaler_file:
        return pickle.load(scaler_file)


def _save_predictions(
    predictions_path: Path,
    predictions,
    targets,
) -> None:
    """
    Save actual and predicted values to a CSV file.

    Parameters
    ----------
    predictions_path : Path
        Destination path for ``predictions.csv``.

    predictions : np.ndarray
        Inverse-transformed model predictions.

    targets : np.ndarray
        Inverse-transformed ground-truth targets.
    """

    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    with open(predictions_path, "w", newline="",encoding="utf-8") as predictions_file:
        writer = csv.writer(predictions_file)
        writer.writerow(["Actual", "Predicted"])
        writer.writerows(zip(targets.tolist(), predictions.tolist()))


def _save_metrics(metrics_path: Path, metrics: Dict[str, float]) -> None:
    """
    Save evaluation metrics to an indented JSON file.

    Parameters
    ----------
    metrics_path : Path
        Destination path for ``evaluation_metrics.json``.

    metrics : dict[str, float]
        Evaluation metrics returned by the evaluator.
    """

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w",encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)


def _print_summary(
    metrics: Dict[str, float],
    predictions_path: Path,
    metrics_path: Path,
    plots_directory: Path,
) -> None:
    """
    Print a concise evaluation summary to the console.

    Parameters
    ----------
    metrics : dict[str, float]
        Evaluation metrics returned by the evaluator.

    predictions_path : Path
        Path where predictions were saved.

    metrics_path : Path
        Path where metrics were saved.

    plots_directory : Path
        Directory where figures were saved.
    """

    print("Evaluation Summary")
    print("------------------")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name} : {metric_value:.4f}")

    print()
    print(f"Predictions saved to {predictions_path}")
    print()
    print(f"Metrics saved to {metrics_path}")
    print()
    print(f"Plots saved to {plots_directory}")


def main() -> None:
    """Run the full evaluation pipeline end to end."""

    device = _select_device()

    test_dataset_path = _resolve_test_dataset_path()
    test_dataset = _load_test_dataset(test_dataset_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    target_scaler_path = Path(config.TARGET_SCALER_FILE)
    target_scaler = _load_target_scaler(target_scaler_path)

    checkpoint_path = Path(config.BEST_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}."
        )

    model = get_model(config.MODEL_NAME)

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        checkpoint_path=checkpoint_path,
        target_scaler=target_scaler,
        device=device,
    )

    results = evaluator.evaluate()

    plotter = EvaluationPlotter(
        output_directory=config.EVALUATION_PLOTS_DIR,
        max_plot_samples=config.MAX_PLOT_SAMPLES,
    )
    plotter.plot_predictions(results["predictions"], results["targets"])
    plotter.plot_residuals(results["predictions"], results["targets"])
    plotter.plot_prediction_scatter(results["predictions"], results["targets"])

    predictions_path = Path(config.PREDICTIONS_FILE)
    _save_predictions(predictions_path, results["predictions"], results["targets"])

    metrics_path = Path(config.EVALUATION_METRICS_FILE)
    _save_metrics(metrics_path, results["metrics"])

    _print_summary(
        metrics=results["metrics"],
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        plots_directory=Path(config.EVALUATION_PLOTS_DIR),
    )


if __name__ == "__main__":
    main()