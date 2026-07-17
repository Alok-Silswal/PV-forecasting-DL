"""
bayesian_optimization.py

Bayesian Optimization for hyperparameter optimization of the short-term
photovoltaic (PV) power forecasting model.

This module implements only Bayesian Optimization. It draws an initial
random design and every subsequent candidate pool using
``hpo.utils.sample_hyperparameters``, evaluates configurations via
``hpo.objective.evaluate_hyperparameters``, and proposes each further
configuration by fitting a Gaussian process surrogate over
``hpo.search_space.get_search_space`` and maximizing Expected
Improvement over a random candidate pool. It contains no training or
evaluation logic of its own and no other search-algorithm logic. Its
return structure matches ``hpo.random_search.run_random_search``
exactly.
"""

import logging
import time

import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Kernel,
    Matern,
    WhiteKernel,
)
from torch.utils.data import DataLoader

from hpo.objective import evaluate_hyperparameters
from hpo.search_space import Hyperparameter, get_search_space
from hpo.utils import sample_hyperparameters

_CANDIDATE_POOL_SIZE = 1000
_GP_N_RESTARTS = 5


def _build_kernel() -> Kernel:
    """
    Build the Gaussian process kernel used by the surrogate model.

    Returns
    -------
    Kernel
        A constant-scaled Matern kernel (nu=2.5) plus a white-noise
        kernel. Matern is used in preference to the squared-exponential
        (RBF) kernel because it assumes a less smooth underlying
        function, which is generally more appropriate for hyperparameter
        response surfaces. The kernel operates on the min-max-normalized,
        one-hot-encoded representation built by ``_encode_configurations``,
        so a unit initial length scale is an appropriate starting point.
    """

    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    )


def _build_encoding_spec(
    search_space: dict[str, Hyperparameter],
) -> list[dict]:
    """
    Build per-dimension encoding metadata for the surrogate model.

    Integer and float hyperparameters are min-max normalized to
    ``[0, 1]``; categorical hyperparameters are one-hot encoded. This
    encoding is used only to build numeric feature vectors for the
    Gaussian process in this module; it does not alter how candidates
    are sampled from ``hpo.search_space``.

    Parameters
    ----------
    search_space : dict[str, Hyperparameter]
        Hyperparameter search space, as returned by
        ``hpo.search_space.get_search_space``.

    Returns
    -------
    list[dict]
        One entry per hyperparameter, in the same (insertion) order as
        ``search_space``, describing how to encode that dimension.
    """

    encoding_spec = []

    for name, hyperparameter in search_space.items():
        if hyperparameter.param_type == "categorical":
            encoding_spec.append(
                {
                    "name": name,
                    "kind": "categorical",
                    "choices": hyperparameter.choices,
                }
            )
        else:
            encoding_spec.append(
                {
                    "name": name,
                    "kind": "numeric",
                    "lower": hyperparameter.lower,
                    "upper": hyperparameter.upper,
                }
            )

    return encoding_spec


def _encode_configurations(
    configurations: list[dict],
    encoding_spec: list[dict],
) -> np.ndarray:
    """
    Encode hyperparameter configurations into numeric feature vectors.

    Parameters
    ----------
    configurations : list[dict]
        Hyperparameter configurations to encode.

    encoding_spec : list[dict]
        Per-dimension encoding metadata from ``_build_encoding_spec``.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(configurations), n_encoded_features)``.
    """

    encoded_rows = []

    for configuration in configurations:

        features = []

        for dimension in encoding_spec:
            value = configuration[dimension["name"]]

            if dimension["kind"] == "categorical":
                features.extend(
                    1.0 if choice == value else 0.0
                    for choice in dimension["choices"]
                )
            else:
                span = dimension["upper"] - dimension["lower"]
                normalized = (value - dimension["lower"]) / span
                features.append(normalized)

        encoded_rows.append(features)

    return np.array(encoded_rows, dtype=float)


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
) -> np.ndarray:
    """
    Compute Expected Improvement for a minimization objective.

    Parameters
    ----------
    mean : np.ndarray
        Gaussian process posterior mean at each candidate.

    std : np.ndarray
        Gaussian process posterior standard deviation at each
        candidate.

    best_observed : float
        Best (lowest) objective value observed so far.

    Returns
    -------
    np.ndarray
        Expected Improvement at each candidate. Where ``std`` is zero,
        Expected Improvement is evaluated at its analytic limit,
        ``max(best_observed - mean, 0)``, rather than being set to
        zero unconditionally.
    """

    improvement = best_observed - mean
    positive_std = std > 0

    z_score = np.divide(
        improvement,
        std,
        out=np.zeros_like(improvement),
        where=positive_std,
    )

    return np.where(
        positive_std,
        improvement * norm.cdf(z_score) + std * norm.pdf(z_score),
        np.maximum(improvement, 0.0),
    )


def run_bayesian_optimization(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    trainer_config: dict,
    num_initial_points: int = 3,
    num_iterations: int = 5,
    random_seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Run Bayesian Optimization over the hyperparameter search space.

    An initial design of ``num_initial_points`` configurations is drawn
    uniformly at random and evaluated. A Gaussian process surrogate is
    then fit to all observations so far and, for ``num_iterations``
    further evaluations, is used to score a freshly sampled random
    candidate pool by Expected Improvement; the highest-scoring
    candidate is evaluated and the surrogate is refit on the updated
    observations before the next iteration.

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

    num_initial_points : int, optional
        Number of randomly sampled configurations evaluated before
        Bayesian Optimization begins. Default is 3 (reduced HPO
        budget, reflecting the smaller six-hyperparameter search
        space).

    num_iterations : int, optional
        Number of surrogate-guided evaluations performed after the
        initial design. Default is 5 (reduced HPO budget, reflecting
        the smaller six-hyperparameter search space).

    random_seed : int, optional
        Seed for the dedicated random number generator used for
        sampling and for the Gaussian process's internal kernel-
        hyperparameter restarts, ensuring reproducibility. Default is
        42.

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
        - ``"trial_results"``: list of trial records, one per evaluated
          configuration, in evaluation order (initial design first).
        - ``"number_of_trials"``: total number of configurations
          evaluated, equal to ``num_initial_points + num_iterations``.
        - ``"elapsed_time_seconds"``: total wall-clock time spent
          evaluating all configurations.

    Raises
    ------
    ValueError
        If ``num_initial_points`` or ``num_iterations`` is less than 1.
    """

    if num_initial_points < 1:
        raise ValueError(
            f"num_initial_points must be >= 1, got {num_initial_points}."
        )

    if num_iterations < 1:
        raise ValueError(
            f"num_iterations must be >= 1, got {num_iterations}."
        )

    rng = np.random.default_rng(random_seed)
    search_space = get_search_space()
    encoding_spec = _build_encoding_spec(search_space)

    trial_results = []
    best_result = None

    observed_configurations = []
    observed_objectives = []

    start_time = time.perf_counter()

    for _ in range(num_initial_points):

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

        observed_configurations.append(hyperparameters)
        observed_objectives.append(trial_result["objective"])

        if (
            best_result is None
            or trial_result["objective"] < best_result["objective"]
        ):
            best_result = trial_result

    gaussian_process = GaussianProcessRegressor(
        kernel=_build_kernel(),
        normalize_y=True,
        n_restarts_optimizer=_GP_N_RESTARTS,
        random_state=random_seed,
    )

    for _ in range(num_iterations):

        features_observed = _encode_configurations(
            observed_configurations,
            encoding_spec,
        )
        objectives_observed = np.array(observed_objectives)

        gaussian_process.fit(features_observed, objectives_observed)

        candidate_pool = [
            sample_hyperparameters(rng)
            for _ in range(_CANDIDATE_POOL_SIZE)
        ]
        features_candidates = _encode_configurations(
            candidate_pool,
            encoding_spec,
        )

        mean, std = gaussian_process.predict(
            features_candidates,
            return_std=True,
        )

        best_observed = float(np.min(objectives_observed))
        acquisition_scores = _expected_improvement(
            mean,
            std,
            best_observed,
        )

        best_candidate_index = int(np.argmax(acquisition_scores))
        hyperparameters = candidate_pool[best_candidate_index]

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

        observed_configurations.append(hyperparameters)
        observed_objectives.append(trial_result["objective"])

        if trial_result["objective"] < best_result["objective"]:
            best_result = trial_result

    elapsed_time_seconds = time.perf_counter() - start_time

    return {
        "best_result": best_result,
        "best_hyperparameters": best_result["hyperparameters"],
        "trial_results": trial_results,
        "number_of_trials": len(trial_results),
        "elapsed_time_seconds": elapsed_time_seconds,
    }