"""
qubo_sa.py

QUBO-inspired Simulated Annealing (QUBO-SA) for hyperparameter
optimization of the short-term photovoltaic (PV) power forecasting
model.

This module implements only QUBO-SA. It initializes a single candidate
configuration via ``hpo.utils.sample_hyperparameters``, evaluates it via
``hpo.objective.evaluate_hyperparameters``, and then repeatedly proposes
a neighbouring configuration, accepts or rejects it under the Boltzmann
acceptance criterion of Algorithm 2 in the reference paper, and cools an
exponentially decaying temperature. It contains no training or
evaluation logic of its own and no other search-algorithm logic. Its
return structure matches ``hpo.random_search.run_random_search``
exactly.

Note on naming
---------------
The reference paper ("Quantum inspired hyperparameter optimization for
enhanced deep learning based intrusion detection in wireless sensor
networks", Vinotha & Eswaran, 2026) titles its outer algorithm "QS-BAT"
and describes Algorithm 2 as a "quantum exponential search" step nested
inside a bat-algorithm swarm. Despite the "QUBO" naming convention used
for this module (matching the terminology requested for this specific
optimizer), the paper never formulates a QUBO matrix, binary decision
variables, or a quadratic objective; Algorithm 2, in isolation, is a
Simulated Annealing acceptance rule (Boltzmann criterion over an
exponential/temperature-based cooling schedule) inspired by
quantum-annealing terminology rather than an actual QUBO/Ising
formulation. This module therefore implements a faithful, standalone
Simulated Annealing optimizer built around that acceptance rule, with
no invented QUBO matrix or binary encoding, consistent with what the
paper actually specifies.
"""

import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from hpo.objective import evaluate_hyperparameters
from hpo.search_space import Hyperparameter, get_search_space
from hpo.utils import sample_hyperparameters

# Smallest temperature used in the Boltzmann acceptance probability,
# avoiding a division producing +inf/nan once the exponential cooling
# schedule has decayed the temperature to (near) zero. Mirrors the
# numerical-safety pattern used for _MIN_QUANTUM_UNIFORM in qs_bat.py.
_MIN_TEMPERATURE = 1e-12


def _validate_parameters(
    num_iterations: int,
    initial_temperature: float,
    alpha: float,
    perturbation_fraction: float,
) -> None:
    """
    Validate QUBO-SA configuration parameters.

    Parameters
    ----------
    num_iterations : int
        Number of annealing iterations after the initial evaluation.

    initial_temperature : float
        Starting temperature of the cooling schedule.

    alpha : float
        Exponential cooling factor.

    perturbation_fraction : float
        Neighbourhood radius, as a fraction of each numeric
        hyperparameter's range.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.
    """

    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}.")

    if initial_temperature <= 0.0:
        raise ValueError(
            f"initial_temperature must be > 0, got {initial_temperature}."
        )

    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")

    if not 0.0 < perturbation_fraction <= 1.0:
        raise ValueError(
            f"perturbation_fraction must be in (0, 1], got "
            f"{perturbation_fraction}."
        )


def _generate_neighbour(
    current_hyperparameters: dict,
    search_space: dict,
    dimension_names: list,
    perturbation_fraction: float,
    rng: np.random.Generator,
) -> dict:
    """
    Generate a neighbouring hyperparameter configuration.

    Every dimension is perturbed by a small, type-appropriate amount so
    the neighbour stays close to ``current_hyperparameters`` rather than
    becoming an independent random sample:

    - Integer dimensions take a random integer step within
      ``+/- max(1, round(perturbation_fraction * range))``, clipped to
      bounds.
    - Float dimensions take a Gaussian step with standard deviation
      ``perturbation_fraction * range``, clipped to bounds.
    - Categorical dimensions switch to a different valid choice, drawn
      uniformly from the choices other than the current one.

    Parameters
    ----------
    current_hyperparameters : dict
        The current hyperparameter configuration.

    search_space : dict
        Mapping from hyperparameter name to ``Hyperparameter``.

    dimension_names : list
        Fixed, ordered list of hyperparameter names.

    perturbation_fraction : float
        Neighbourhood radius, as a fraction of each numeric
        hyperparameter's range.

    rng : numpy.random.Generator
        Dedicated, seeded random number generator.

    Returns
    -------
    dict
        A valid neighbouring hyperparameter configuration.
    """

    neighbour = {}

    for name in dimension_names:
        hyperparameter = search_space[name]
        current_value = current_hyperparameters[name]

        if hyperparameter.param_type == "categorical":
            other_choices = [
                choice for choice in hyperparameter.choices if choice != current_value
            ]
            if other_choices:
                choice_index = int(rng.integers(len(other_choices)))
                neighbour[name] = other_choices[choice_index]
            else:
                neighbour[name] = current_value

        elif hyperparameter.param_type == "integer":
            value_range = hyperparameter.upper - hyperparameter.lower
            step_radius = max(1, round(perturbation_fraction * value_range))
            delta = int(rng.integers(-step_radius, step_radius + 1))
            neighbour[name] = int(
                np.clip(
                    current_value + delta,
                    hyperparameter.lower,
                    hyperparameter.upper,
                )
            )

        else:
            value_range = hyperparameter.upper - hyperparameter.lower
            step_size = perturbation_fraction * value_range
            delta = rng.normal(0.0, step_size) if step_size > 0.0 else 0.0
            neighbour[name] = float(
                np.clip(
                    current_value + delta,
                    hyperparameter.lower,
                    hyperparameter.upper,
                )
            )

    return neighbour


def _boltzmann_accept(
    delta: float,
    temperature: float,
    rng: np.random.Generator,
) -> bool:
    """
    Decide whether to accept a candidate solution under the Boltzmann
    acceptance criterion of Algorithm 2.

    An improving candidate (``delta < 0``) is always accepted. A
    non-improving candidate is accepted with probability
    ``exp(-delta / temperature)``.

    Parameters
    ----------
    delta : float
        ``new_objective - current_objective``. Negative values indicate
        an improving candidate, since the objective is minimized.

    temperature : float
        Current annealing temperature. Clipped to ``_MIN_TEMPERATURE``
        before use, to keep the acceptance probability well defined as
        the temperature decays towards zero.

    rng : numpy.random.Generator
        Dedicated, seeded random number generator.

    Returns
    -------
    bool
        ``True`` if the candidate is accepted, ``False`` otherwise.
    """

    if delta < 0.0:
        return True

    safe_temperature = max(temperature, _MIN_TEMPERATURE)
    acceptance_probability = float(np.exp(-delta / safe_temperature))

    return bool(rng.uniform() < acceptance_probability)


def run_qubo_sa(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    trainer_config: dict,
    num_iterations: int = 12,
    initial_temperature: float = 10.0,
    alpha: float = 0.95,
    perturbation_fraction: float = 0.1,
    random_seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Run QUBO-SA (QUBO-inspired Simulated Annealing) over the
    hyperparameter search space.

    A single configuration is drawn by random initialization and
    evaluated, seeding both the current solution and the best solution.
    For ``num_iterations`` further iterations, a neighbouring
    configuration is generated, evaluated, and accepted or rejected
    under the Boltzmann acceptance criterion; the current solution is
    updated only on acceptance, while the best solution is updated
    whenever a strictly better configuration is evaluated, regardless of
    acceptance. The temperature is cooled exponentially
    (``temperature *= alpha``) once per iteration.

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

    num_iterations : int, optional
        Number of annealing iterations performed after the initial
        evaluation. Default is 12 (reduced HPO budget, reflecting the
        smaller six-hyperparameter search space).

    initial_temperature : float, optional
        Starting temperature of the exponential cooling schedule. Must
        be > 0. Default is 10.0.

    alpha : float, optional
        Exponential cooling factor, applied once per iteration as
        ``temperature *= alpha``. Must be in ``(0, 1]``. Default is
        0.95.

    perturbation_fraction : float, optional
        Neighbourhood radius used by ``_generate_neighbour``, as a
        fraction of each numeric hyperparameter's range. Must be in
        ``(0, 1]``. Default is 0.1.

    random_seed : int, optional
        Seed for the dedicated random number generator used throughout,
        ensuring reproducibility. Default is 42.

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
        - ``"trial_results"``: list of trial records, one per
          evaluation, in evaluation order (initial configuration
          first).
        - ``"number_of_trials"``: total number of configurations
          evaluated, equal to ``1 + num_iterations``.
        - ``"elapsed_time_seconds"``: total wall-clock time spent
          evaluating all configurations.

    Raises
    ------
    ValueError
        If any configuration parameter is outside its valid range.
    """

    _validate_parameters(
        num_iterations, initial_temperature, alpha, perturbation_fraction
    )

    rng = np.random.default_rng(random_seed)

    resolved_logger = (
        logger
        if logger is not None
        else logging.getLogger(f"{__name__}.run_qubo_sa")
    )

    if logger is None:
        resolved_logger.addHandler(logging.NullHandler())
        resolved_logger.propagate = False

    search_space = get_search_space()
    dimension_names = list(search_space.keys())

    trial_results = []

    def _evaluate(hyperparameters: dict) -> dict:
        result = evaluate_hyperparameters(
            hyperparameters=hyperparameters,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            trainer_config=trainer_config,
            logger=resolved_logger,
        )
        trial_result = {"hyperparameters": hyperparameters, **result}
        trial_results.append(trial_result)
        return trial_result

    start_time = time.perf_counter()

    # --- Random initialization and evaluation ----------------------------
    current_hyperparameters = sample_hyperparameters(rng)
    current_result = _evaluate(current_hyperparameters)
    best_result = current_result

    temperature = initial_temperature

    # --- Main annealing loop ----------------------------------------------
    for iteration in range(1, num_iterations + 1):

        neighbour_hyperparameters = _generate_neighbour(
            current_hyperparameters,
            search_space,
            dimension_names,
            perturbation_fraction,
            rng,
        )
        neighbour_result = _evaluate(neighbour_hyperparameters)

        delta = neighbour_result["objective"] - current_result["objective"]

        if _boltzmann_accept(delta, temperature, rng):
            current_hyperparameters = neighbour_hyperparameters
            current_result = neighbour_result

        if neighbour_result["objective"] < best_result["objective"]:
            best_result = neighbour_result

        temperature = max(temperature * alpha, _MIN_TEMPERATURE)

        resolved_logger.info(
            "QUBO-SA iteration %d/%d | temperature: %.6f | "
            "current objective: %.6f | best objective: %.6f",
            iteration,
            num_iterations,
            temperature,
            current_result["objective"],
            best_result["objective"],
        )

    elapsed_time_seconds = time.perf_counter() - start_time

    return {
        "best_result": best_result,
        "best_hyperparameters": best_result["hyperparameters"],
        "trial_results": trial_results,
        "number_of_trials": len(trial_results),
        "elapsed_time_seconds": elapsed_time_seconds,
    }