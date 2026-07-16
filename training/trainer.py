"""
Training orchestration for short-term photovoltaic (PV) power forecasting.

This module implements the ``Trainer`` class, which orchestrates the full
training loop: forward/backward passes, validation, metric computation,
learning-rate scheduling, gradient clipping, early stopping, checkpointing,
and epoch-level logging.

The trainer has no knowledge of the model architecture, the data
preprocessing pipeline, or the hyperparameter-optimization strategy used to
select its dependencies. All dependencies (model, data loaders, criterion,
optimizer, scheduler, device, logger) are injected through the constructor,
which keeps this module free to be reused unchanged by downstream
optimization routines such as Bayesian Optimization, QS-BAT, or
QUBO-inspired Simulated Annealing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # PyTorch < 2.0 exposes only the private base class
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from training.metrics import compute_metrics


class Trainer:
    """
    Orchestrates training and validation for a PV power forecasting model.

    The trainer performs model training, validation, metric computation,
    early stopping, checkpointing, and checkpoint resumption. It does not
    perform data preprocessing, plotting, test-set evaluation, or
    hyperparameter optimization.

    Parameters
    ----------
    model : nn.Module
        The forecasting model to train. Must already be moved to the
        target device by the caller, or will be moved via ``.to(device)``
        during construction.

    train_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for training.

    val_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for validation.

    criterion : nn.Module
        Loss function used for both training and validation
        (e.g., the output of ``training.loss.get_loss_function()``).

    optimizer : Optimizer
        Optimizer instance already bound to ``model.parameters()``.

    scheduler : LRScheduler or ReduceLROnPlateau, optional
        Learning-rate scheduler instance already bound to ``optimizer``.
        Stepped once per epoch, after validation. Standard schedulers
        (e.g., ``StepLR``, ``CosineAnnealingLR``) are stepped with no
        arguments; ``ReduceLROnPlateau`` is stepped with the epoch's
        validation loss, since it does not share the standard scheduler
        interface. Pass ``None`` to disable learning-rate scheduling.

    device : torch.device
        Device on which tensors and the model reside.

    logger : logging.Logger
        Logger instance used for epoch-level logging
        (e.g., the output of ``utils.logger.get_logger()``).

    checkpoint_dir : str or Path
        Directory in which the best checkpoint is saved.

    early_stopping_patience : int
        Number of consecutive non-improving epochs allowed before training
        is stopped.

    gradient_clip_value : float, optional
        Maximum gradient norm for gradient clipping. If ``None``, gradient
        clipping is not applied.

    num_epochs : int, optional
        Maximum number of epochs to train for. Training may stop earlier
        due to early stopping. Default is 100.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]],
        device: torch.device,
        logger: logging.Logger,
        checkpoint_dir: Union[str, Path],
        early_stopping_patience: int,
        gradient_clip_value: Optional[float] = None,
        num_epochs: int = 100,
        enable_checkpointing: bool = True,
    ) -> None:

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_value = gradient_clip_value
        self.num_epochs = num_epochs
        self.enable_checkpointing = enable_checkpointing

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"

        # Early stopping state
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Resume state (updated by load_checkpoint)
        self.start_epoch = 0

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop with validation and early stopping.

        Returns
        -------
        dict[str, list[float]]
            History dictionary containing per-epoch training loss,
            validation loss, and validation metrics, suitable for
            downstream plotting.
        """

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "rmse": [],
            "mae": [],
            "mape": [],
            "r2": [],
            "nrmse": [],
        }

        for epoch in range(self.start_epoch, self.num_epochs):

            train_loss = self._train_one_epoch(epoch)

            val_loss, val_metrics = self._validate_one_epoch(epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau does not share the standard
                    # scheduler interface and requires the monitored
                    # metric to be passed explicitly.
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["rmse"].append(val_metrics["rmse"])
            history["mae"].append(val_metrics["mae"])
            history["mape"].append(val_metrics["mape"])
            history["r2"].append(val_metrics["r2"])
            history["nrmse"].append(val_metrics["nrmse"])

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                "Epoch %d/%d | train_loss: %.6f | val_loss: %.6f | "
                "rmse: %.6f | mae: %.6f | mape: %.4f%% | r2: %.6f | "
                "nrmse: %.6f | lr: %.8f",
                epoch + 1,
                self.num_epochs,
                train_loss,
                val_loss,
                val_metrics["rmse"],
                val_metrics["mae"],
                val_metrics["mape"],
                val_metrics["r2"],
                val_metrics["nrmse"],
                current_lr,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.enable_checkpointing:
                    self._save_checkpoint(epoch)
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(
                        "Early stopping triggered at epoch %d "
                        "(no improvement for %d epochs).",
                        epoch + 1,
                        self.early_stopping_patience,
                    )
                    break

        return history

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """
        Restore model, optimizer, scheduler, and training state from a
        checkpoint file.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file previously saved by
            ``_save_checkpoint``.

        Returns
        -------
        int
            The epoch at which training should resume.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint["best_val_loss"]
        self.start_epoch = checkpoint["epoch"] + 1

        self.logger.info(
            "Resumed from checkpoint '%s' at epoch %d (best_val_loss: %.6f).",
            path,
            self.start_epoch + 1,
            self.best_val_loss,
        )

        return self.start_epoch

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Run a single training epoch over ``self.train_loader``.

        Parameters
        ----------
        epoch : int
            Zero-indexed current epoch, used only for the progress-bar
            description.

        Returns
        -------
        float
            Average training loss over the epoch.
        """

        self.model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs} [train]",
            leave=False,
        )

        for inputs, targets in progress_bar:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(inputs)

            print("Prediction:", predictions.shape)
            print("Target    :", targets.shape)

            loss = self.criterion(predictions, targets)

            loss.backward()

            if self.gradient_clip_value is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value,
                )

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            progress_bar.set_postfix(loss=loss.item())

        average_loss = running_loss / len(self.train_loader.dataset)

        return average_loss

    def _validate_one_epoch(
        self,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run a single validation epoch over ``self.val_loader``.

        Parameters
        ----------
        epoch : int
            Zero-indexed current epoch, used only for the progress-bar
            description.

        Returns
        -------
        tuple[float, dict[str, float]]
            Average validation loss and the corresponding metrics
            dictionary computed by ``training.metrics.compute_metrics``.
        """

        self.model.eval()

        running_loss = 0.0

        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs} [val]",
            leave=False,
        )

        with torch.no_grad():

            for inputs, targets in progress_bar:

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(inputs)

                print("Prediction:", predictions.shape)
                print("Target    :", targets.shape)

                loss = self.criterion(predictions, targets)

                running_loss += loss.item() * inputs.size(0)

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

                progress_bar.set_postfix(loss=loss.item())

        average_loss = running_loss / len(self.val_loader.dataset)

        predictions_array = torch.cat(all_predictions).numpy()
        targets_array = torch.cat(all_targets).numpy()

        metrics = compute_metrics(
            predictions_array,
            targets_array,
        )

        return average_loss, metrics

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save the current model as the best checkpoint.

        Parameters
        ----------
        epoch : int
            Zero-indexed epoch at which the checkpoint is being saved.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, self.checkpoint_path)

        self.logger.info(
            "Saved new best checkpoint at epoch %d (val_loss: %.6f) -> %s",
            epoch + 1,
            self.best_val_loss,
            self.checkpoint_path,
        )