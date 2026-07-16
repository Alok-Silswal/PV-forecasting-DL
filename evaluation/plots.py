"""
plots.py

Publication-quality visualization utilities for short-term photovoltaic
(PV) power forecasting evaluation results.

This module implements the ``EvaluationPlotter`` class, which consumes
already inverse-transformed predictions and targets (as produced by
``evaluation.evaluator.Evaluator``) and generates publication-quality
figures. It performs no inference, metric computation, checkpoint
loading, dataset loading, or hyperparameter optimization; its only
responsibility is visualization.
"""

from pathlib import Path
from typing import Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


class EvaluationPlotter:
    """
    Generates publication-quality evaluation plots for PV power
    forecasting results.

    Parameters
    ----------
    output_directory : str or Path
        Directory in which generated figures are saved. Created
        automatically if it does not already exist.

    max_plot_samples : int
        Maximum number of consecutive samples to plot. If the number of
        samples exceeds this value, only the first ``max_plot_samples``
        consecutive samples are plotted. A continuous window (rather
        than random sampling) preserves the temporal evolution of the
        PV generation curve. Must be a positive integer.

    Raises
    ------
    ValueError
        If ``max_plot_samples`` is not a positive integer.
    """

    _DPI = 300

    def __init__(
        self,
        output_directory: Union[str, Path],
        max_plot_samples: int,
    ) -> None:

        if max_plot_samples <= 0:
            raise ValueError(
                "max_plot_samples must be a positive integer; got "
                f"{max_plot_samples}."
            )

        self.output_directory = Path(output_directory)
        self.max_plot_samples = max_plot_samples

        self.output_directory.mkdir(parents=True, exist_ok=True)

    def plot_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Path:
        """
        Plot predicted vs. actual PV power as a time-series line plot.

        Parameters
        ----------
        predictions : np.ndarray
            One-dimensional array of inverse-transformed model
            predictions.

        targets : np.ndarray
            One-dimensional array of inverse-transformed ground-truth
            targets.

        Returns
        -------
        Path
            Path to the saved figure.
        """

        self._validate_inputs(predictions, targets)

        plot_predictions, plot_targets = self._downsample(
            predictions, targets
        )
        x_axis = np.arange(plot_predictions.shape[0])

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(x_axis, plot_targets, label="Actual", linewidth=1.0)
        ax.plot(x_axis, plot_predictions, label="Predicted", linewidth=1.0)

        ax.set_title("Predicted vs. Actual PV Power")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("PV Power")
        ax.legend()
        ax.grid(True,alpha=0.3,)

        return self._save_figure(fig, "prediction_vs_actual.png")

    def plot_residuals(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Path:
        """
        Plot residuals (Actual - Predicted) over time as a scatter plot.

        Parameters
        ----------
        predictions : np.ndarray
            One-dimensional array of inverse-transformed model
            predictions.

        targets : np.ndarray
            One-dimensional array of inverse-transformed ground-truth
            targets.

        Returns
        -------
        Path
            Path to the saved figure.
        """

        self._validate_inputs(predictions, targets)

        plot_predictions, plot_targets = self._downsample(
            predictions, targets
        )
        residuals = plot_targets - plot_predictions
        x_axis = np.arange(residuals.shape[0])

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.scatter(x_axis, residuals, s=8, alpha=0.6)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

        ax.set_title("Residuals (Actual - Predicted) Over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Residual")
        ax.grid(True)

        return self._save_figure(fig, "residual_plot.png")

    def plot_prediction_scatter(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Path:
        """
        Plot predicted vs. actual PV power as a scatter plot, with the
        ideal ``y = x`` reference line overlaid.

        Parameters
        ----------
        predictions : np.ndarray
            One-dimensional array of inverse-transformed model
            predictions.

        targets : np.ndarray
            One-dimensional array of inverse-transformed ground-truth
            targets.

        Returns
        -------
        Path
            Path to the saved figure.
        """

        self._validate_inputs(predictions, targets)

        plot_predictions, plot_targets = self._downsample(
            predictions, targets
        )

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.scatter(
            plot_targets,
            plot_predictions,
            s=8,
            alpha=0.6,
            label="Predictions",
        )

        lower_bound = min(plot_targets.min(), plot_predictions.min())
        upper_bound = max(plot_targets.max(), plot_predictions.max())
        ideal_line = np.array([lower_bound, upper_bound])
        ax.plot(
            ideal_line,
            ideal_line,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="y = x",
        )

        ax.set_title("Predicted vs. Actual PV Power (Scatter)")
        ax.set_xlabel("Actual PV Power")
        ax.set_ylabel("Predicted PV Power")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        ax.grid(True)

        return self._save_figure(fig, "prediction_scatter.png")

    def _validate_inputs(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """
        Validate that predictions and targets are well-formed.

        Parameters
        ----------
        predictions : np.ndarray
            Array of model predictions.

        targets : np.ndarray
            Array of ground-truth targets.

        Raises
        ------
        ValueError
            If the arrays are not one-dimensional, have mismatched
            lengths, or are empty.
        """

        if predictions.ndim != 1 or targets.ndim != 1:
            raise ValueError(
                "predictions and targets must be one-dimensional; got "
                f"shapes {predictions.shape} and {targets.shape}."
            )

        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(
                "predictions and targets must have identical lengths; "
                f"got {predictions.shape[0]} and {targets.shape[0]}."
            )

        if predictions.shape[0] == 0:
            raise ValueError("predictions and targets must not be empty.")

    def _downsample(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restrict predictions and targets to at most ``max_plot_samples``
        consecutive samples.

        Parameters
        ----------
        predictions : np.ndarray
            One-dimensional array of predictions.

        targets : np.ndarray
            One-dimensional array of targets.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The predictions and targets, truncated to the first
            ``max_plot_samples`` consecutive samples if the original
            arrays exceed that length; otherwise returned unchanged.
        """

        number_of_samples = predictions.shape[0]

        if number_of_samples <= self.max_plot_samples:
            return predictions, targets

        return (
            predictions[: self.max_plot_samples],
            targets[: self.max_plot_samples],
        )

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """
        Save a figure to the output directory at 300 DPI with a tight
        layout, then close it to free resources.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.

        filename : str
            Name of the output file, relative to ``output_directory``.

        Returns
        -------
        Path
            Path to the saved figure.
        """

        output_path = self.output_directory / filename

        fig.tight_layout()
        fig.savefig(output_path, dpi=self._DPI,bbox_inches="tight")
        plt.close(fig)

        return output_path