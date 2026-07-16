"""
utils.py

Shared hyperparameter sampling utilities for the short-term
photovoltaic (PV) power forecasting HPO suite.

This module provides generic, optimizer-independent sampling of a
single hyperparameter and of a complete hyperparameter configuration
from ``hpo.search_space.SEARCH_SPACE``. It contains no optimization,
training, or evaluation logic of its own; it is consumed by search
algorithms such as Random Search, Bayesian Optimization, QS-BAT, and
QUBO-inspired Simulated Annealing wherever they need to draw a random
candidate configuration.
"""

import numpy as np

from hpo.search_space import Hyperparameter, get_search_space


def sample_parameter_value(
    hyperparameter: Hyperparameter,
    rng: np.random.Generator,
) -> object:
    """
    Draw one random candidate value for a single hyperparameter.

    Parameters
    ----------
    hyperparameter : Hyperparameter
        Hyperparameter definition from the search space.

    rng : numpy.random.Generator
        Dedicated, seeded random number generator.

    Returns
    -------
    object
        A sampled ``int``, ``float``, or categorical choice.

    Raises
    ------
    ValueError
        If ``hyperparameter.type`` is not one of ``"integer"``,
        ``"float"``, or ``"categorical"``.
    """

    if hyperparameter.param_type == "integer":
        return int(rng.integers(hyperparameter.lower, hyperparameter.upper + 1))

    if hyperparameter.param_type == "float":
        return float(rng.uniform(hyperparameter.lower, hyperparameter.upper))

    if hyperparameter.param_type == "categorical":
        choice_index = int(rng.integers(len(hyperparameter.choices)))
        return hyperparameter.choices[choice_index]

    raise ValueError(
        f"Unsupported hyperparameter type '{hyperparameter.param_type}' "
        f"for parameter '{hyperparameter.name}'."
    )


def sample_hyperparameters(rng: np.random.Generator) -> dict:
    """
    Draw one complete random hyperparameter configuration.

    Parameters
    ----------
    rng : numpy.random.Generator
        Dedicated, seeded random number generator.

    Returns
    -------
    dict
        Mapping from hyperparameter name to its sampled value.
    """

    search_space = get_search_space()

    return {
        name: sample_parameter_value(hyperparameter, rng)
        for name, hyperparameter in search_space.items()
    }