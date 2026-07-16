"""
evaluator.py

Core inference engine for short-term photovoltaic (PV) power forecasting.

This module implements the ``Evaluator`` class, which loads a trained
model's best checkpoint, runs inference over a test DataLoader, inverse
transforms predictions and targets back to their original scale, and
computes evaluation metrics. It performs no training, plotting, or file
output; it only produces results for a downstream evaluation pipeline to
consume.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import compute_metrics


class Evaluator:
    """
    Runs inference and computes evaluation metrics for a trained PV
    forecasting model.

    The evaluator restores model weights from a checkpoint, performs
    inference over a test DataLoader, inverse-transforms predictions and
    targets to their original scale, and computes evaluation metrics. It
    does not train the model, generate plots, or write any files.

    Parameters
    ----------
    model : nn.Module
        The forecasting model to evaluate. Its architecture must match
        the weights stored in ``checkpoint_path``.

    test_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for the test
        set.

    checkpoint_path : str or Path
        Path to the checkpoint file saved by ``training.trainer.Trainer``
        (e.g., ``best_checkpoint.pt``).

    target_scaler : Any
        Fitted scaler used during preprocessing to normalize the target
        variable (e.g., a scikit-learn ``StandardScaler`` loaded from
        ``target_scaler.pkl``). Must expose an ``inverse_transform``
        method.

    device : torch.device
        Device on which inference is performed.

    Raises
    ------
    ValueError
        If ``target_scaler`` does not expose ``inverse_transform``, or if
        ``test_loader`` has an empty underlying dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        checkpoint_path: Union[str, Path],
        target_scaler: Any,
        device: torch.device,
    ) -> None:

        if not hasattr(target_scaler, "inverse_transform"):
            raise ValueError(
                "target_scaler must expose an 'inverse_transform' method; "
                f"got object of type {type(target_scaler).__name__}."
            )

        if len(test_loader.dataset) == 0:
            raise ValueError(
                "test_loader has an empty dataset; there is nothing to "
                "evaluate."
            )

        self.model = model.to(device)
        self.test_loader = test_loader
        self.checkpoint_path = Path(checkpoint_path)
        self.target_scaler = target_scaler
        self.device = device

    def evaluate(self) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """
        Run the full evaluation workflow.

        Loads the checkpoint, restores model weights, runs inference over
        the test DataLoader, inverse-transforms predictions and targets,
        and computes evaluation metrics.

        Returns
        -------
        dict[str, np.ndarray or dict[str, float]]
            Dictionary with keys:

            - ``"predictions"``: inverse-transformed model predictions.
            - ``"targets"``: inverse-transformed ground-truth targets.
            - ``"metrics"``: dictionary returned by
              ``training.metrics.compute_metrics``.
        """

        self._load_checkpoint()

        self.model.eval()

        raw_predictions, raw_targets = self._run_inference()

        predictions = self._inverse_transform(raw_predictions)
        targets = self._inverse_transform(raw_targets)

        metrics = compute_metrics(predictions, targets)

        return {
            "predictions": predictions,
            "targets": targets,
            "metrics": metrics,
        }

    def _load_checkpoint(self) -> None:
        """
        Load the checkpoint file and restore model weights.

        Raises
        ------
        FileNotFoundError
            If ``self.checkpoint_path`` does not exist.
        RuntimeError
            If the checkpoint file cannot be deserialized.
        ValueError
            If the deserialized checkpoint does not contain a
            ``"model_state_dict"`` key.
        """

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}."
            )

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint from {self.checkpoint_path}: "
                f"{exc}"
            ) from exc

        if (
            not isinstance(checkpoint, dict)
            or "model_state_dict" not in checkpoint
        ):
            raise ValueError(
                f"Invalid checkpoint at {self.checkpoint_path}: expected "
                f"a dictionary containing a 'model_state_dict' key."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _run_inference(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference over the test DataLoader.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Concatenated raw (still normalized) predictions and targets,
            on CPU.
        """

        all_predictions = []
        all_targets = []

        progress_bar = tqdm(
            self.test_loader,
            desc="Evaluating",
            leave=False,
        )

        with torch.no_grad():

            for inputs, targets in progress_bar:

                inputs = inputs.to(self.device)

                predictions = self.model(inputs)

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        return predictions, targets

    def _inverse_transform(self, values: torch.Tensor) -> np.ndarray:
        """
        Inverse-transform normalized values back to their original scale.

        Parameters
        ----------
        values : torch.Tensor
            Normalized values, of shape ``(n_samples,)`` or
            ``(n_samples, 1)``.

        Returns
        -------
        np.ndarray
            Inverse-transformed values, of shape ``(n_samples,)``.
        """

        values_array = values.numpy().reshape(-1, 1)

        original_scale = self.target_scaler.inverse_transform(values_array)

        return original_scale.reshape(-1)