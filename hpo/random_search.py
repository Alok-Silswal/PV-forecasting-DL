"""
random_search.py

Random Search optimizer for hyperparameter optimization of the
short-term photovoltaic (PV) power forecasting model.

This module independently samples ``num_trials`` hyperparameter
configurations from ``hpo.search_space.SEARCH_SPACE`` using a dedicated,
seeded random number generator, evaluates each via
``hpo.objective.evaluate_hyperparameters``, and returns the best
configuration found. It contains no training or evaluation logic of its
own and no other search-algorithm logic.
"""

import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from hpo.objective import evaluate_hyperparameters
from hpo.utils import sample_hyperparameters


def run_random_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    trainer_config: dict,
    num_trials: int = 5,
    random_seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Run Random Search over the hyperparameter search space.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for training.

    val_loader : DataLoader
        DataLoader yielding ``(inputs, targets)`` batches for validation.

    device : torch.device
        Device on which each trial is trained.

    trainer_config : dict
        Non-tunable trainer settings, passed unchanged to
        ``evaluate_hyperparameters`` for every trial.

    num_trials : int, optional
        Number of independently sampled hyperparameter configurations to
        evaluate. Default is 5 (reduced HPO budget, reflecting the
        smaller six-hyperparameter search space).

    random_seed : int, optional
        Seed for the dedicated random number generator used for
        sampling, ensuring reproducibility. Default is 42.

    logger : logging.Logger, optional
        Logger passed unchanged to ``evaluate_hyperparameters`` for
        every trial. If ``None``, a silent logger is used (see
        ``hpo.objective``).

    Returns
    -------
    dict
        Dictionary containing:

        - ``"best_result"``: trial record (hyperparameters and metrics)
          for the trial with the lowest objective.
        - ``"best_hyperparameters"``: hyperparameter dict for the best
          trial.
        - ``"trial_results"``: list of trial records, one per sampled
          configuration, in evaluation order.
        - ``"number_of_trials"``: total number of configurations
          evaluated.
        - ``"elapsed_time_seconds"``: total wall-clock time spent
          evaluating all configurations.

    Raises
    ------
    ValueError
        If ``num_trials`` is less than 1.
    """

    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}.")

    rng = np.random.default_rng(random_seed)

    trial_results = []
    best_result = None

    start_time = time.perf_counter()

    for _ in range(num_trials):

        hyperparameters = sample_hyperparameters(rng)

        result = evaluate_hyperparameters(
            hyperparameters=hyperparameters,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            trainer_config=trainer_config,
            logger=logger,
        )

        trial_result = {"hyperparameters": hyperparameters, **result}
        trial_results.append(trial_result)

        if best_result is None or trial_result["objective"] < best_result["objective"]:
            best_result = trial_result

    elapsed_time_seconds = time.perf_counter() - start_time

    return {
        "best_result": best_result,
        "best_hyperparameters": best_result["hyperparameters"],
        "trial_results": trial_results,
        "number_of_trials": len(trial_results),
        "elapsed_time_seconds": elapsed_time_seconds,
    }