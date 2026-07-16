"""
finalize_hpo.py

Finalization step for the hyperparameter optimization (HPO) pipeline of
the short-term photovoltaic (PV) power forecasting project.

This module's only responsibility is to compare the independently
produced results of every HPO technique (Random Search, Bayesian
Optimization, QS-BAT, QUBO-SA) already saved under a given HPO
directory, select the single global winner by minimum
``best_result["objective"]`` (validation loss; this selection criterion
is unchanged from what each optimizer already uses internally), and
persist that winner as ``best_hpo.json``.

It performs no optimization, training, or evaluation logic of its own,
and it never runs automatically as a side effect of any optimizer.
Finalization is a separate, explicit step, invoked via
``python run_hpo.py --finalize``.
"""

import json
from pathlib import Path
from typing import Dict, List

_OPTIMIZER_NAMES: List[str] = [
    "random_search",
    "bayesian_optimization",
    "qs_bat",
    "qubo_sa",
]


def _load_results(hpo_dir: Path) -> Dict[str, dict]:
    """
    Load every available HPO result file from ``hpo_dir``.

    Parameters
    ----------
    hpo_dir : Path
        Directory containing ``results_<optimizer_name>.json`` files,
        as produced by ``run_hpo.py`` for each optimizer.

    Returns
    -------
    dict[str, dict]
        Mapping from optimizer name to its parsed result dictionary,
        containing only the optimizers whose result file was found.

    Raises
    ------
    FileNotFoundError
        If no HPO result files are found in ``hpo_dir`` at all.
    """

    results: Dict[str, dict] = {}

    for optimizer_name in _OPTIMIZER_NAMES:
        result_path = hpo_dir / f"results_{optimizer_name}.json"

        if not result_path.exists():
            continue

        with open(result_path, "r", encoding="utf-8") as result_file:
            results[optimizer_name] = json.load(result_file)

    if not results:
        raise FileNotFoundError(
            f"No HPO result files found in {hpo_dir}. Run "
            f"'python run_hpo.py --optimizer <name>' for at least one "
            f"optimizer before finalizing."
        )

    return results


def _select_winner(results: Dict[str, dict]) -> tuple:
    """
    Select the optimizer with the lowest ``best_result["objective"]``.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping from optimizer name to its parsed result dictionary, as
        returned by ``_load_results``.

    Returns
    -------
    tuple[str, dict]
        The winning optimizer's name and its result dictionary.
    """

    winning_optimizer_name = min(
        results,
        key=lambda name: results[name]["best_result"]["objective"],
    )

    return winning_optimizer_name, results[winning_optimizer_name]


def finalize_hpo(hpo_dir: Path) -> dict:
    """
    Compare every available HPO result in ``hpo_dir`` and save the
    global winner as ``best_hpo.json``.

    Parameters
    ----------
    hpo_dir : Path
        Directory containing ``results_<optimizer_name>.json`` files
        and where ``best_hpo.json`` will be written.

    Returns
    -------
    dict
        The saved contents of ``best_hpo.json``: ``"winning_optimizer"``,
        ``"objective"``, and ``"best_hyperparameters"``.

    Raises
    ------
    FileNotFoundError
        If no HPO result files are found in ``hpo_dir``.
    """

    results = _load_results(hpo_dir)

    winning_optimizer_name, winning_result = _select_winner(results)

    best_hpo = {
        "winning_optimizer": winning_optimizer_name,
        "objective": winning_result["best_result"]["objective"],
        "best_hyperparameters": winning_result["best_hyperparameters"],
    }

    best_hpo_path = hpo_dir / "best_hpo.json"

    with open(best_hpo_path, "w", encoding="utf-8") as best_hpo_file:
        json.dump(best_hpo, best_hpo_file, indent=4)

    return best_hpo