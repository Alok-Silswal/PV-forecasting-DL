"""
qs_bat.py

Quantum-behaved Search Bat Algorithm (QS-BAT) for hyperparameter
optimization of the short-term photovoltaic (PV) power forecasting
model.

This module implements only QS-BAT. It initializes a population of
bats via ``hpo.utils.sample_hyperparameters``, evaluates every
configuration via ``hpo.objective.evaluate_hyperparameters``, and
maintains a swarm of bats whose positions evolve every iteration via
the classical Bat Algorithm's frequency/velocity/position equations
(Yang, 2010), substituting a quantum-behaved exponential jump (in the
style of Quantum Particle Swarm Optimization) for the local search step
whenever a bat's pulse condition triggers. Categorical dimensions are
handled with an explicit loudness-gated keep/resample rule during the
quantum jump, since velocity-based movement has no natural meaning for
unordered categories. A separately maintained elite archive is
refreshed every iteration and its members are mutated per-dimension to
promote diversity. It contains no training or evaluation logic of its
own and no other search-algorithm logic. Its return structure matches
``hpo.random_search.run_random_search`` exactly.
"""

import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from hpo.objective import evaluate_hyperparameters
from hpo.search_space import Hyperparameter, get_search_space
from hpo.utils import sample_hyperparameters, sample_parameter_value

# Smallest value used in place of exactly 0.0 when drawing the uniform
# variate consumed by ln(1/u) in the quantum jump, avoiding a division
# producing +inf.
_MIN_QUANTUM_UNIFORM = 1e-12


def _validate_parameters(
    population_size: int,
    num_iterations: int,
    elite_archive_size: int,
    mutation_rate: float,
    frequency_min: float,
    frequency_max: float,
    initial_loudness: float,
    initial_pulse_rate: float,
    alpha: float,
    gamma: float,
) -> None:
    """
    Validate QS-BAT configuration parameters.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.
    """

    if population_size < 1:
        raise ValueError(f"population_size must be >= 1, got {population_size}.")

    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}.")

    if elite_archive_size < 1:
        raise ValueError(
            f"elite_archive_size must be >= 1, got {elite_archive_size}."
        )

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}.")

    if frequency_min > frequency_max:
        raise ValueError(
            f"frequency_min ({frequency_min}) must be <= frequency_max "
            f"({frequency_max})."
        )

    if not 0.0 <= initial_loudness <= 1.0:
        # Loudness doubles as the quantum-jump scale, the acceptance
        # probability (rand < A_i), and the categorical resample
        # probability (1 - A_i). All three uses require A_i in [0, 1];
        # since loudness only ever decays (A_i *= alpha, alpha <= 1),
        # constraining the initial value keeps it valid for the whole run.
        raise ValueError(
            f"initial_loudness must be in [0, 1], got {initial_loudness}."
        )

    if not 0.0 <= initial_pulse_rate <= 1.0:
        raise ValueError(
            f"initial_pulse_rate must be in [0, 1], got {initial_pulse_rate}."
        )

    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")

    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}.")


def _encode(
    hyperparameters: dict,
    search_space: dict,
    dimension_names: list,
) -> np.ndarray:
    """
    Encode a hyperparameter configuration into a normalized position
    vector for the continuous BAT mechanics.

    Integer and float dimensions are min-max normalized to ``[0, 1]``.

    Categorical dimensions are represented as their choice index,
    normalized to ``[0, 1]`` over ``[0, len(choices) - 1]``. This is an
    engineering approximation, not a claim about the variable's true
    structure: it imposes an artificial ordinal distance between
    choices (e.g., it treats kernel_size 3 as "closer" to 5 than to 7)
    that a categorical variable does not actually possess. The
    approximation exists solely so the classical BAT frequency/
    velocity/position equations -- which require a single continuous
    scalar per dimension -- can be applied uniformly across every
    dimension without a second, structurally different position
    representation for categoricals. The correct fix would be one-hot
    encoding (as used in ``bayesian_optimization.py``), but that
    requires each categorical to occupy a constrained multi-dimensional
    block with its own decode rule (e.g., argmax), which BAT's
    single-scalar velocity mechanics do not accommodate without a
    deeper redesign. This encoding has no bearing on how candidates are
    sampled (`hpo.utils.sample_hyperparameters` samples categoricals
    uniformly over their actual choices, not over this ordinal
    representation) or on the quantum-jump branch, which overrides
    categorical dimensions with the explicit keep/resample rule in
    ``_quantum_jump_candidate`` rather than using this encoding at all.

    Parameters
    ----------
    hyperparameters : dict
        A complete hyperparameter configuration.

    search_space : dict
        Mapping from hyperparameter name to ``Hyperparameter``.

    dimension_names : list
        Fixed, ordered list of hyperparameter names.

    Returns
    -------
    np.ndarray
        1-D normalized position vector, one entry per dimension.
    """

    position = np.empty(len(dimension_names), dtype=float)

    for index, name in enumerate(dimension_names):
        hyperparameter = search_space[name]
        value = hyperparameters[name]

        if hyperparameter.param_type == "categorical":
            choice_index = hyperparameter.choices.index(value)
            span = max(len(hyperparameter.choices) - 1, 1)
            position[index] = choice_index / span
        else:
            span = hyperparameter.upper - hyperparameter.lower
            if span == 0:
                position[index] = 0.0
            else:
                position[index] = (value - hyperparameter.lower) / span 

    return position


def _decode(
    position: np.ndarray,
    search_space: dict,
    dimension_names: list,
) -> dict:
    """
    Decode a normalized position vector back into a valid hyperparameter
    configuration.

    The position is clipped to ``[0, 1]`` before decoding. Integer
    dimensions are rounded to the nearest valid integer; categorical
    dimensions are rounded to the nearest valid choice index.

    Parameters
    ----------
    position : np.ndarray
        1-D normalized position vector, one entry per dimension.

    search_space : dict
        Mapping from hyperparameter name to ``Hyperparameter``.

    dimension_names : list
        Fixed, ordered list of hyperparameter names.

    Returns
    -------
    dict
        A valid hyperparameter configuration.
    """

    clipped_position = np.clip(position, 0.0, 1.0)

    hyperparameters = {}

    for index, name in enumerate(dimension_names):
        hyperparameter = search_space[name]
        value = clipped_position[index]

        if hyperparameter.param_type == "categorical":
            span = max(len(hyperparameter.choices) - 1, 1)
            choice_index = int(round(value * span))
            choice_index = min(max(choice_index, 0), len(hyperparameter.choices) - 1)
            hyperparameters[name] = hyperparameter.choices[choice_index]
        elif hyperparameter.param_type == "integer":
            span = hyperparameter.upper - hyperparameter.lower
            raw_value = hyperparameter.lower if span == 0 else hyperparameter.lower + value * span
            hyperparameters[name] = int(
                min(max(round(raw_value), hyperparameter.lower), hyperparameter.upper)
            )
        else:
            span = hyperparameter.upper - hyperparameter.lower
            raw_value = hyperparameter.lower if span == 0 else hyperparameter.lower + value * span
            hyperparameters[name] = min(
                max(raw_value, hyperparameter.lower), hyperparameter.upper
            )

    return hyperparameters


def _quantum_jump_candidate(
    bat_position: np.ndarray,
    bat_hyperparameters: dict,
    global_best_position: np.ndarray,
    loudness: float,
    search_space: dict,
    dimension_names: list,
    rng: np.random.Generator,
) -> dict:
    """
    Generate a candidate configuration via a quantum-behaved exponential
    jump around the global best.

    Numeric (integer and float) dimensions use the standard QPSO
    exponential-jump formulation, with the bat's own loudness used as
    the jump scale in place of a separate tunable coefficient::

        x_new = x_best +/- A_i * |x_i - x_best| * ln(1 / u),  u ~ U(0, 1)

    This adapts classical QPSO, which centers the jump on ``mBest``
    (the mean of all particles' personal-best positions), rather than
    on the single global best used here. That substitution is not
    merely a simplification of convenience: the QS-BAT reference paper
    does not specify the quantum search equation, and this BAT design
    (per finalized decision 6/7) keeps no per-bat personal-best
    history -- every bat moves unconditionally every iteration, with
    acceptance gating only global state -- so there is no ``pbest_i``
    population to average into an ``mBest`` in the first place.
    ``global_best_position`` is therefore the only population-level
    reference point this design actually maintains.

    Categorical dimensions have no continuous jump equivalent, so the
    current category is kept with probability ``1 - A_i``; otherwise a
    new category is resampled uniformly via
    ``hpo.utils.sample_parameter_value``.

    Parameters
    ----------
    bat_position : np.ndarray
        The bat's current normalized position (used only for the
        numeric dimensions' jump magnitude).

    bat_hyperparameters : dict
        The bat's current (decoded) hyperparameter configuration (used
        only for the categorical dimensions' "keep current" branch).

    global_best_position : np.ndarray
        Current global best normalized position, the jump's center.

    loudness : float
        This bat's current loudness, ``A_i in [0, 1]``.

    search_space : dict
        Mapping from hyperparameter name to ``Hyperparameter``.

    dimension_names : list
        Fixed, ordered list of hyperparameter names.

    rng : numpy.random.Generator
        Dedicated, seeded random number generator.

    Returns
    -------
    dict
        A valid hyperparameter configuration produced by the jump.
    """

    jumped_position = np.empty_like(bat_position)

    for index in range(len(dimension_names)):
        uniform_draw = rng.uniform(_MIN_QUANTUM_UNIFORM, 1.0)
        sign = rng.choice((-1.0, 1.0))
        magnitude = loudness * abs(bat_position[index] - global_best_position[index])
        jumped_position[index] = global_best_position[index] + sign * magnitude * np.log(
            1.0 / uniform_draw
        )

    jumped_position = np.clip(jumped_position, 0.0, 1.0)

    numeric_decoded = _decode(jumped_position, search_space, dimension_names)

    candidate = {}

    for name in dimension_names:
        hyperparameter = search_space[name]

        if hyperparameter.param_type == "categorical":
            if rng.uniform() < (1.0 - loudness):
                candidate[name] = bat_hyperparameters[name]
            else:
                candidate[name] = sample_parameter_value(hyperparameter, rng)
        else:
            candidate[name] = numeric_decoded[name]

    return candidate


def _insert_into_archive(
    archive: list,
    trial_result: dict,
    max_size: int,
) -> None:
    """
    Insert a trial record into a sorted elite archive, in place.

    Parameters
    ----------
    archive : list
        Current elite archive, sorted ascending by ``"objective"``.
        Modified in place.

    trial_result : dict
        Trial record to insert.

    max_size : int
        Maximum number of entries the archive may hold.
    """

    archive.append(trial_result)
    archive.sort(key=lambda trial: trial["objective"])

    del archive[max_size:]


def run_qs_bat(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    trainer_config: dict,
    population_size: int = 4,
    num_iterations: int = 3,
    elite_archive_size: int = 5,
    mutation_rate: float = 0.1,
    frequency_min: float = 0.0,
    frequency_max: float = 2.0,
    initial_loudness: float = 0.5,
    initial_pulse_rate: float = 0.5,
    alpha: float = 0.9,
    gamma: float = 0.9,
    random_seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Run QS-BAT (Quantum-behaved Search Bat Algorithm) over the
    hyperparameter search space.

    A population of ``population_size`` bats is initialized by random
    sampling and evaluated, seeding both the global best and a sorted
    elite archive. For ``num_iterations`` further iterations, every bat
    updates its frequency, velocity, and position according to the
    classical Bat Algorithm equations; whenever that bat's pulse
    condition triggers, a quantum-behaved exponential jump around the
    global best is used instead. Every bat's resulting candidate is
    always evaluated and the bat always moves to it, regardless of
    fitness. A loudness-gated acceptance rule separately determines
    whether that candidate is allowed to update the global best and
    elite archive, and, when accepted, that bat's own loudness decays
    and pulse rate rises. After all bats have moved, every current
    elite archive member is mutated per-dimension (with probability
    ``mutation_rate`` per dimension) and the mutated variant is
    evaluated and considered for archive/global-best insertion using
    plain fitness-based elitism.

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

    population_size : int, optional
        Number of bats in the population. Default is 4 (reduced HPO
        budget, reflecting the smaller six-hyperparameter search
        space).

    num_iterations : int, optional
        Number of iterations the swarm evolves for, after the initial
        population evaluation. Default is 3 (reduced HPO budget,
        reflecting the smaller six-hyperparameter search space).

    elite_archive_size : int, optional
        Maximum number of configurations retained in the elite archive.
        Default is 5.

    mutation_rate : float, optional
        Per-dimension mutation probability applied to elite archive
        members every iteration. Must be in ``[0, 1]``. Default is 0.1.

    frequency_min : float, optional
        Minimum bat pulse frequency. Default is 0.0.

    frequency_max : float, optional
        Maximum bat pulse frequency. Default is 2.0.

    initial_loudness : float, optional
        Initial loudness ``A_i`` for every bat. Must be in ``[0, 1]``,
        since loudness also serves as the acceptance probability and
        the categorical resample probability. Default is 0.5.

    initial_pulse_rate : float, optional
        Initial pulse emission rate ``r_i`` for every bat. Must be in
        ``[0, 1]``. Default is 0.5.

    alpha : float, optional
        Loudness decay factor, applied as ``A_i *= alpha`` whenever a
        bat's candidate is accepted. Must be in ``(0, 1]``. Default is
        0.9.

    gamma : float, optional
        Pulse rate growth-rate factor, applied as
        ``r_i = r_i_0 * (1 - exp(-gamma * t))`` whenever a bat's
        candidate is accepted. Must be > 0. Default is 0.9.

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
          evaluation, in evaluation order.
        - ``"number_of_trials"``: total number of configurations
          evaluated.
        - ``"elapsed_time_seconds"``: total wall-clock time spent
          evaluating all configurations.

    Raises
    ------
    ValueError
        If any configuration parameter is outside its valid range.
    """

    _validate_parameters(
        population_size, num_iterations, elite_archive_size, mutation_rate,
        frequency_min, frequency_max, initial_loudness, initial_pulse_rate,
        alpha, gamma,
    )

    rng = np.random.default_rng(random_seed)
    
    resolved_logger = (
    logger
    if logger is not None
    else logging.getLogger(f"{__name__}.run_qs_bat")
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

    # --- Initialize and evaluate the bat population ---------------------
    bat_hyperparameters = [sample_hyperparameters(rng) for _ in range(population_size)]
    bat_positions = [
        _encode(hp, search_space, dimension_names) for hp in bat_hyperparameters
    ]
    bat_velocities = [np.zeros(len(dimension_names)) for _ in range(population_size)]
    bat_loudness = [initial_loudness] * population_size
    bat_pulse_rate = [initial_pulse_rate] * population_size

    bat_trials = [_evaluate(hp) for hp in bat_hyperparameters]

    global_best_result = min(bat_trials, key=lambda trial: trial["objective"])
    global_best_position = _encode(
        global_best_result["hyperparameters"], search_space, dimension_names
    )

    elite_archive = sorted(bat_trials, key=lambda trial: trial["objective"])
    elite_archive = elite_archive[:elite_archive_size]

    # --- Main QS-BAT loop -------------------------------------------------
    for iteration in range(1, num_iterations + 1):

        for bat_index in range(population_size):

            beta = rng.uniform()
            current_frequency = frequency_min + (frequency_max - frequency_min) * beta

            bat_velocities[bat_index] = bat_velocities[bat_index] + (
                bat_positions[bat_index] - global_best_position
            ) * current_frequency

            flight_position = np.clip(
                bat_positions[bat_index] + bat_velocities[bat_index], 0.0, 1.0
            )

            if rng.uniform() > bat_pulse_rate[bat_index]:
                candidate_hyperparameters = _quantum_jump_candidate(
                    bat_positions[bat_index],
                    bat_hyperparameters[bat_index],
                    global_best_position,
                    bat_loudness[bat_index],
                    search_space,
                    dimension_names,
                    rng,
                )
                new_position = _encode(
                    candidate_hyperparameters, search_space, dimension_names
                )
            else:
                candidate_hyperparameters = _decode(
                    flight_position, search_space, dimension_names
                )
                new_position = flight_position

            # Every bat always moves to its newly computed position,
            # regardless of the acceptance rule below (design decision:
            # acceptance gates global-best/archive updates, not movement).
            bat_positions[bat_index] = new_position
            bat_hyperparameters[bat_index] = candidate_hyperparameters

            trial_result = _evaluate(candidate_hyperparameters)
            new_objective = trial_result["objective"]

            accept = (
                rng.uniform() < bat_loudness[bat_index]
                and new_objective < global_best_result["objective"]
            )

            if accept:
                global_best_result = trial_result
                global_best_position = new_position.copy()

                _insert_into_archive(elite_archive, trial_result, elite_archive_size)

                bat_loudness[bat_index] *= alpha
                bat_pulse_rate[bat_index] = initial_pulse_rate * (
                    1.0 - np.exp(-gamma * iteration)
                )

        # --- Elite archive mutation (per-dimension, deterministic elitism) --
        for elite_trial in list(elite_archive):

            elite_hyperparameters = elite_trial["hyperparameters"]
            mutated_hyperparameters = dict(elite_hyperparameters)

            for name in dimension_names:
                if rng.uniform() < mutation_rate:
                    mutated_hyperparameters[name] = sample_parameter_value(
                        search_space[name], rng
                    )

            if mutated_hyperparameters == elite_hyperparameters:
                # No dimension actually changed; skip the wasted
                # evaluation of an exact duplicate.
                continue

            mutated_trial = _evaluate(mutated_hyperparameters)

            if mutated_trial["objective"] < global_best_result["objective"]:
                global_best_result = mutated_trial
                global_best_position = _encode(
                    mutated_hyperparameters, search_space, dimension_names
                )

            _insert_into_archive(elite_archive, mutated_trial, elite_archive_size)

        resolved_logger.info(
            "QS-BAT iteration %d/%d | best_objective: %.6f | "
            "archive_size: %d | mean_loudness: %.4f",
            iteration,
            num_iterations,
            global_best_result["objective"],
            len(elite_archive),
            float(np.mean(bat_loudness)),
        )        

    elapsed_time_seconds = time.perf_counter() - start_time

    return {
        "best_result": global_best_result,
        "best_hyperparameters": global_best_result["hyperparameters"],
        "trial_results": trial_results,
        "number_of_trials": len(trial_results),
        "elapsed_time_seconds": elapsed_time_seconds,
    }