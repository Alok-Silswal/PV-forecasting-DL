"""
hpo_runner.py

Single entry point for running any hyperparameter optimization (HPO)
algorithm for the short-term photovoltaic (PV) power forecasting model.

This module contains no optimization, training, or evaluation logic of
its own. It only validates the requested optimizer name, dispatches to
the corresponding ``run_*`` function with all arguments passed through
unchanged, emits high-level start/finish logging, and returns that
function's result unmodified. Iteration-level logging already exists
inside every optimizer and is not duplicated here.

Supported optimizers are ``"random_search"``, ``"bayesian_optimization"``,
``"qs_bat"``, and ``"qubo_sa"``. ``grid_search.py`` exists in this
package for completeness but is intentionally excluded from dispatch:
its exhaustive search space produces an impractically large number of
trials for this project and it is not part of the HPO pipeline that is
actually run.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from hpo.bayesian_optimization import run_bayesian_optimization
from hpo.qs_bat import run_qs_bat
from hpo.qubo_sa import run_qubo_sa
from hpo.random_search import run_random_search

OPTIMIZERS: dict[str, Callable[..., dict]] = {
    "random_search": run_random_search,
    "bayesian_optimization": run_bayesian_optimization,
    "qs_bat": run_qs_bat,
    "qubo_sa": run_qubo_sa,
}

_DISPLAY_NAMES: dict[str, str] = {
    "random_search": "Random Search",
    "bayesian_optimization": "Bayesian Optimization",
    "qs_bat": "QS-BAT",
    "qubo_sa": "QUBO-SA",
}


def _validate_optimizer_name(optimizer_name: str) -> None:
    """
    Validate that the requested optimizer is supported.

    Parameters
    ----------
    optimizer_name : str
        Name of the requested optimizer.

    Raises
    ------
    ValueError
        If ``optimizer_name`` is not a key of ``OPTIMIZERS``.
    """

    if optimizer_name not in OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Supported "
            f"optimizers are: {sorted(OPTIMIZERS)}."
        )


def run_hpo(
    optimizer_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    trainer_config: dict,
    logger: logging.Logger | None = None,
    **optimizer_kwargs: Any,
) -> dict:
    """
    Dispatch to the requested HPO optimizer and return its result.

    Parameters
    ----------
    optimizer_name : str
        Name of the optimizer to run. Must be one of ``"random_search"``,
        ``"bayesian_optimization"``, ``"qs_bat"``, or ``"qubo_sa"``.

    train_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for training.
        Passed unchanged to the selected optimizer.

    val_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for validation.
        Passed unchanged to the selected optimizer.

    device : torch.device
        Device on which every trial is trained. Passed unchanged to the
        selected optimizer.

    trainer_config : dict
        Non-tunable trainer settings. Passed unchanged to the selected
        optimizer.

    logger : logging.Logger, optional
        Logger used for this module's own high-level start/finish
        messages, and passed unchanged to the selected optimizer for its
        own iteration-level logging. If ``None``, a silent logger is
        used (see ``hpo.objective``), matching the fallback already used
        throughout the project.

    **optimizer_kwargs : Any
        Optimizer-specific keyword arguments (for example
        ``num_trials``, ``population_size``, ``num_iterations``,
        ``random_seed``), forwarded unchanged to the selected optimizer.

    Returns
    -------
    dict
        The selected optimizer's result, returned exactly as received:
        a dictionary containing ``"best_result"``,
        ``"best_hyperparameters"``, ``"trial_results"``,
        ``"number_of_trials"``, and ``"elapsed_time_seconds"``.

    Raises
    ------
    ValueError
        If ``optimizer_name`` is not a supported optimizer.
    """

    _validate_optimizer_name(optimizer_name)

    resolved_logger = (
        logger
        if logger is not None
        else logging.getLogger(f"{__name__}.run_hpo")
    )

    if logger is None:
        resolved_logger.addHandler(logging.NullHandler())
        resolved_logger.propagate = False

    display_name = _DISPLAY_NAMES[optimizer_name]
    optimizer_function = OPTIMIZERS[optimizer_name]

    resolved_logger.info("Starting %s...", display_name)

    result = optimizer_function(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        trainer_config=trainer_config,
        logger=resolved_logger,
        **optimizer_kwargs,
    )

    resolved_logger.info("Finished %s.", display_name)
    resolved_logger.info(
        "Best objective: %.6f", result["best_result"]["objective"]
    )
    resolved_logger.info(
        "Elapsed time: %.2f seconds", result["elapsed_time_seconds"]
    )

    return result