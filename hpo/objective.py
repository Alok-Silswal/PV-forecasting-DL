"""
objective.py

Optimizer-independent objective function for hyperparameter optimization
of the short-term photovoltaic (PV) power forecasting model.

Given a candidate hyperparameter configuration, this module instantiates
the model with those hyperparameters, trains it using the existing
``Trainer`` pipeline, and returns the best validation loss achieved
during training together with the corresponding evaluation metrics. It
contains no search-algorithm logic and is shared unchanged by Random
Search, Bayesian Optimization, QS-BAT, and QUBO-inspired Simulated
Annealing.

Fixed (non-tuned) architecture hyperparameters
------------------------------------------------
``DCNN_KERNEL_SIZE``, ``DCNN_DILATION_RATE``, ``MLP_HIDDEN_DIM``, and
``MLP_DROPOUT_RATE`` are intentionally excluded from
``hpo.search_space`` (a research decision to shrink the search space
and reduce runtime) and are therefore never present in ``hyperparameters``.
``ProposedModel`` is instantiated using the project's existing default
values for these four architecture settings, taken unchanged from
``config.py``, while the remaining six hyperparameters come from the
candidate configuration supplied by the optimizer.
"""

import logging

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs import config
from models.proposed_model import ProposedModel
from training.loss import get_loss_function
from training.trainer import Trainer

_REQUIRED_HYPERPARAMETER_KEYS = (
    "DCNN_FILTERS",
    "DCNN_DROPOUT_RATE",
    "BILSTM_HIDDEN_SIZE",
    "BILSTM_DROPOUT_RATE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
)

_REQUIRED_TRAINER_CONFIG_KEYS = (
    "num_epochs",
    "early_stopping_patience",
    "gradient_clip_value",
    "checkpoint_dir",
)


def _validate_hyperparameters(hyperparameters: dict) -> None:
    """
    Validate that all required hyperparameter keys are present.

    Parameters
    ----------
    hyperparameters : dict
        Candidate hyperparameter configuration.

    Raises
    ------
    KeyError
        If any required hyperparameter key is missing.
    """

    missing_keys = [
        key for key in _REQUIRED_HYPERPARAMETER_KEYS if key not in hyperparameters
    ]

    if missing_keys:
        raise KeyError(
            f"hyperparameters is missing required key(s): {missing_keys}."
        )


def _validate_trainer_config(trainer_config: dict) -> None:
    """
    Validate that all required trainer configuration keys are present.

    Parameters
    ----------
    trainer_config : dict
        Non-tunable trainer settings.

    Raises
    ------
    KeyError
        If any required trainer configuration key is missing.
    """

    missing_keys = [
        key for key in _REQUIRED_TRAINER_CONFIG_KEYS if key not in trainer_config
    ]

    if missing_keys:
        raise KeyError(
            f"trainer_config is missing required key(s): {missing_keys}."
        )


def _resolve_logger(logger: logging.Logger | None) -> logging.Logger:
    """
    Resolve the logger to use, falling back to a silent logger.

    Parameters
    ----------
    logger : logging.Logger or None
        Logger supplied by the caller, or ``None``.

    Returns
    -------
    logging.Logger
        The supplied logger, or a silent logger if none was supplied.
    """

    if logger is not None:
        return logger

    silent_logger = logging.getLogger(f"{__name__}.evaluate_hyperparameters")
    silent_logger.addHandler(logging.NullHandler())
    silent_logger.propagate = False

    return silent_logger


def evaluate_hyperparameters(
    hyperparameters: dict[str, int | float],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    trainer_config: dict[str, int | float | str],
    logger: logging.Logger | None = None,
) -> dict:
    """
    Train and evaluate a candidate hyperparameter configuration.

    Instantiates ``ProposedModel`` with the supplied architecture
    hyperparameters, trains it with the supplied optimizer
    hyperparameters using the existing ``Trainer`` pipeline (with
    checkpoint saving disabled), and returns the best validation loss
    achieved during training together with its corresponding metrics.

    ``DCNN_KERNEL_SIZE``, ``DCNN_DILATION_RATE``, ``MLP_HIDDEN_DIM``,
    and ``MLP_DROPOUT_RATE`` are not part of the HPO search space and
    are therefore not read from ``hyperparameters``; they are supplied
    to ``ProposedModel`` using the project's fixed default values from
    ``config.py``.

    Parameters
    ----------
    hyperparameters : dict
        Candidate hyperparameter configuration. Must contain the keys
        ``"DCNN_FILTERS"``, ``"DCNN_DROPOUT_RATE"``,
        ``"BILSTM_HIDDEN_SIZE"``, ``"BILSTM_DROPOUT_RATE"``,
        ``"LEARNING_RATE"``, and ``"WEIGHT_DECAY"``.

    train_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for training.

    val_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for validation.

    device : torch.device
        Device on which training is performed.

    trainer_config : dict
        Non-tunable trainer settings. Must contain the keys
        ``"num_epochs"``, ``"early_stopping_patience"``,
        ``"gradient_clip_value"``, and ``"checkpoint_dir"``.

    logger : logging.Logger, optional
        Logger for epoch-level training logs. If ``None``, a silent
        logger is used.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"objective"``: best validation loss achieved (float).
        - ``"val_loss"``: best validation loss achieved (float).
        - ``"rmse"``, ``"mae"``, ``"mape"``, ``"r2"``, ``"nrmse"``:
          metrics at the best epoch (float).
        - ``"best_epoch"``: 1-indexed epoch at which the best
          validation loss occurred (int).

    Raises
    ------
    KeyError
        If ``hyperparameters`` or ``trainer_config`` is missing a
        required key.
    RuntimeError
        If training produces no validation loss values.
    """

    _validate_hyperparameters(hyperparameters)
    _validate_trainer_config(trainer_config)

    resolved_logger = _resolve_logger(logger)

    model = ProposedModel(
        dcnn_filters=int(hyperparameters["DCNN_FILTERS"]),
        dcnn_kernel_size=int(config.DCNN_KERNEL_SIZE),
        dcnn_dilation_rate=int(config.DCNN_DILATION_RATE),
        dcnn_dropout_rate=float(hyperparameters["DCNN_DROPOUT_RATE"]),
        bilstm_hidden_size=int(hyperparameters["BILSTM_HIDDEN_SIZE"]),
        bilstm_dropout_rate=float(hyperparameters["BILSTM_DROPOUT_RATE"]),
        mlp_hidden_dim=int(config.MLP_HIDDEN_DIM),
        mlp_dropout_rate=float(config.MLP_DROPOUT_RATE),
    ).to(device)

    criterion = get_loss_function()

    optimizer = Adam(
        model.parameters(),
        lr=float(hyperparameters["LEARNING_RATE"]),
        weight_decay=float(hyperparameters["WEIGHT_DECAY"]),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        logger=resolved_logger,
        checkpoint_dir=trainer_config["checkpoint_dir"],
        early_stopping_patience=int(trainer_config["early_stopping_patience"]),
        gradient_clip_value=trainer_config["gradient_clip_value"],
        num_epochs=int(trainer_config["num_epochs"]),
        enable_checkpointing=False,
    )

    history = trainer.train()

    if not history["val_loss"]:
        raise RuntimeError(
            "Training produced no validation loss values; check "
            "'num_epochs' and 'early_stopping_patience' in trainer_config."
        )

    best_epoch_index = int(np.argmin(history["val_loss"]))

    return {
        "objective": history["val_loss"][best_epoch_index],
        "val_loss": history["val_loss"][best_epoch_index],
        "rmse": history["rmse"][best_epoch_index],
        "mae": history["mae"][best_epoch_index],
        "mape": history["mape"][best_epoch_index],
        "r2": history["r2"][best_epoch_index],
        "nrmse": history["nrmse"][best_epoch_index],
        "best_epoch": best_epoch_index + 1,
    }