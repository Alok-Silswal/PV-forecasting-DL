"""
search_space.py

Defines the tunable hyperparameter search space for short-term
photovoltaic (PV) power forecasting.

This module contains no optimization logic. It is a single,
optimizer-independent source of truth describing which
hyperparameters may be tuned and their valid ranges/choices.
It is shared unchanged by Random Search, Bayesian Optimization,
QS-BAT, and QUBO-inspired Simulated Annealing.

Search space scope
-------------------
Only six hyperparameters are exposed for HPO: ``DCNN_FILTERS``,
``DCNN_DROPOUT_RATE``, ``BILSTM_HIDDEN_SIZE``, ``BILSTM_DROPOUT_RATE``,
``LEARNING_RATE``, and ``WEIGHT_DECAY``. ``DCNN_KERNEL_SIZE``,
``DCNN_DILATION_RATE``, ``MLP_HIDDEN_DIM``, and ``MLP_DROPOUT_RATE``
were intentionally removed from HPO (an explicit research decision to
reduce runtime, since they contributed little practical improvement)
and are fixed to their existing default architecture values in
``config.py``. They remain part of the model architecture itself and
are still accepted by ``ProposedModel``; they are simply no longer
varied by any optimizer. ``hpo.objective.evaluate_hyperparameters``
supplies them as fixed defaults.
"""

from dataclasses import dataclass


@dataclass
class Hyperparameter:
    """
    Describes a single tunable hyperparameter.

    Parameters
    ----------
    name : str
        Name of the hyperparameter.

    param_type : str
        One of "integer", "float", or "categorical".

    lower : int | float | None
        Lower bound for numeric hyperparameters.

    upper : int | float | None
        Upper bound for numeric hyperparameters.

    choices : list[int | float | str] | None
        Candidate values for categorical hyperparameters.

    description : str
        Human-readable description.
    """

    name: str
    param_type: str
    lower: int | float | None
    upper: int | float | None
    choices: list[int | float | str] | None
    description: str


SEARCH_SPACE: dict[str, Hyperparameter] = {
    "DCNN_FILTERS": Hyperparameter(
        name="DCNN_FILTERS",
        param_type="integer",
        lower=32,
        upper=128,
        choices=None,
        description="Number of convolutional filters in the DCNN branch.",
    ),
    "DCNN_DROPOUT_RATE": Hyperparameter(
        name="DCNN_DROPOUT_RATE",
        param_type="float",
        lower=0.0,
        upper=0.5,
        choices=None,
        description="Dropout rate applied within the DCNN branch.",
    ),
    "BILSTM_HIDDEN_SIZE": Hyperparameter(
        name="BILSTM_HIDDEN_SIZE",
        param_type="integer",
        lower=64,
        upper=256,
        choices=None,
        description="Hidden size of the Residual BiLSTM branch.",
    ),
    "BILSTM_DROPOUT_RATE": Hyperparameter(
        name="BILSTM_DROPOUT_RATE",
        param_type="float",
        lower=0.0,
        upper=0.5,
        choices=None,
        description="Dropout rate applied within the Residual BiLSTM branch.",
    ),
    "LEARNING_RATE": Hyperparameter(
        name="LEARNING_RATE",
        param_type="float",
        lower=1e-4,
        upper=5e-3,
        choices=None,
        description="Optimizer learning rate.",
    ),
    "WEIGHT_DECAY": Hyperparameter(
        name="WEIGHT_DECAY",
        param_type="float",
        lower=1e-6,
        upper=1e-3,
        choices=None,
        description="Optimizer weight decay (L2 regularization).",
    ),
}


def get_search_space() -> dict[str, Hyperparameter]:
    """
    Return the hyperparameter search space.

    Returns
    -------
    dict[str, Hyperparameter]
        Mapping from hyperparameter names to their definitions.
    """

    return SEARCH_SPACE