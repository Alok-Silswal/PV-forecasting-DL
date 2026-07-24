"""
analysis/significance_tests.py

Inferential statistics for comparing PV power forecasting models across
run-wise metrics: the Friedman omnibus test, run independently for
every (horizon, metric) cell discovered in the loader's output.

This module performs NO I/O and NO descriptive statistics. It consumes
the in-memory structure returned by `load_all_metrics()` (defined
elsewhere in the project) and returns structured results only.


Consumes `analysis.load_metrics.load_all_metrics` output directly:

    dict[model_name, dict[horizon_name, ModelMetrics]]

where `ModelMetrics.metrics` is `dict[metric_name, list[float]]`
(per-run values, in run order).

This is model-first / horizon-second, the opposite axis order needed
for per-(horizon, metric) comparisons across models. A private
reshaping helper (`_pivot_to_horizon_metric_model`) performs this pivot
in-memory; it duplicates no I/O or validation logic from the loader,
it only re-indexes the already-loaded `ModelMetrics` objects.

"""

from __future__ import annotations
from analysis.load_metrics import ModelMetrics

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import friedmanchisquare

__all__ = [
    "FriedmanResult",
    "MetricSignificanceResult",
    "run_significance_analysis",
]

ALPHA_DEFAULT = 0.05

_PROPOSED_KEY = "proposed"


# ======================================================================
# Dataclasses
# ======================================================================
#
# DECISION: ACCEPT dedicated dataclasses.
# JUSTIFICATION: The result of this pipeline is consumed downstream by
# reporting/table-generation code (a different module, per the brief).
# Plain dicts would work but push every consumer to remember string keys
# ("p_value" vs "pvalue" vs "p"), with no static checking and easy typos
# across module boundaries. `frozen=True` dataclasses give (a) an explicit,
# self-documenting contract, (b) immutability appropriate for a result
# that should never be mutated after computation, (c) IDE/type-checker
# support for downstream code. The extra boilerplate is justified because
# these objects cross a module boundary into code this module doesn't
# control.


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    significant: bool
    n_models: int
    n_runs: int


@dataclass(frozen=True)
class MetricSignificanceResult:
    """Full result for one (horizon, metric) cell."""
    horizon: str
    metric: str
    friedman: FriedmanResult


# ======================================================================
# Return structure
# ======================================================================
#
# DECISION: REJECT the prompt's suggested nested-dict-of-dicts return
# ({horizon: {metric: {friedman:...}}}).
# JUSTIFICATION: A flat List[MetricSignificanceResult] is used instead.
# Reasons:
#   1. Every consumer of this data (table generators, plotting code) will
#      need to iterate over "all (horizon, metric) results" -- a flat list
#      of typed records is directly iterable/filterable
#      (e.g. `[r for r in results if r.metric == "RMSE"]`) without nested
#      `.items()` loops and without ever risking a KeyError on a missing
#      horizon/metric combination.
#   2. `horizon` and `metric` are attributes of the result itself
#      (MetricSignificanceResult), not implied by dict nesting -- so the
#      identifying information travels with the record even after
#      filtering, sorting, or exporting.
#   3. Nested dicts of dataclasses is a redundant hybrid: you still need
#      the dataclass for the leaf, but now also need to know the outer
#      dict-key convention. A flat list removes that ambiguity.
# A convenience `group_by_horizon` helper is intentionally NOT added here,
# since grouping/reshaping for presentation is a downstream concern
# (explicitly out of scope: "generate publication tables").


SignificanceResults = Dict[str, Dict[str, "MetricSignificanceResult"]]


def run_significance_analysis(
    all_metrics: Dict[str, Dict[str, ModelMetrics]],
    alpha: float = ALPHA_DEFAULT,
    proposed_model: str = _PROPOSED_KEY,
) -> SignificanceResults:
    """
    Run the full significance pipeline over every (horizon, metric) pair
    discovered in the loader's output.

    Parameters
    ----------
    all_metrics : output of `analysis.load_metrics.load_all_metrics`,
        shape {model_name: {horizon_name: ModelMetrics}}.
    alpha : significance threshold for the Friedman decision.
    proposed_model : name of the model required to be present for
        every (horizon, metric) cell. Defaults to "proposed" (this
        project's convention), but is not hardcoded since
        load_metrics.py discovers model names from the filesystem
        rather than declaring them. Retained for validation purposes
        (see `_validate_or_raise`) even though pairwise comparisons
        against it have been removed.

    Returns
    -------
    dict[horizon][metric] -> MetricSignificanceResult, mirroring the
    horizon->metric nesting used elsewhere in this package.
    """
    pivoted = _pivot_to_horizon_metric_model(all_metrics)

    results: SignificanceResults = {}
    for horizon, metric_block in pivoted.items():
        results[horizon] = {}
        for metric_name, model_runs in metric_block.items():
            results[horizon][metric_name] = _analyze_one(
                horizon, metric_name, model_runs, alpha, proposed_model
            )
    return results


def _analyze_one(
    horizon: str,
    metric: str,
    model_runs: Dict[str, List[float]],
    alpha: float,
    proposed_model: str,
) -> MetricSignificanceResult:
    _validate_or_raise(horizon, metric, model_runs, proposed_model)

    model_names = list(model_runs.keys())
    samples = [np.asarray(model_runs[m], dtype=float) for m in model_names]

    friedman = _run_friedman(samples, model_names, alpha)

    return MetricSignificanceResult(
        horizon=horizon,
        metric=metric,
        friedman=friedman,
    )


# ======================================================================
# Validation
# ======================================================================
#
# DECISION on raise conditions, with justification per check:
#
#   RAISES (ValueError) -- these indicate the input is unusable for a
#   statistically valid Friedman test, so silently continuing would
#   produce a meaningless or crashing result further down the pipeline:
#     - Proposed model missing entirely: `proposed_model` is still a
#       required, explicitly-named entrant in every (horizon, metric)
#       cell by this project's convention, even though it is no longer
#       singled out for pairwise comparison -- its absence signals an
#       incomplete evaluation run and is treated the same as any other
#       missing model would be.
#     - Fewer than 3 models for a metric: Friedman's test is undefined
#       below 3 groups (it degenerates to a paired test at best).
#     - Unequal run counts across models: Friedman requires paired/
#       aligned observations (same run index = same random seed /
#       condition across models). Silently truncating or padding would
#       fabricate a comparison that never happened.
#     - Fewer than 2 runs for any model: the omnibus test is undefined
#       on a single observation.

def _pivot_to_horizon_metric_model(
    all_metrics: Dict[str, Dict[str, ModelMetrics]],
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Reshape loader output from model->horizon->ModelMetrics into
    horizon->metric->model->[runs], the axis order this module's
    per-cell comparisons need.

    Does not duplicate any loading/validation logic: it only re-indexes
    ModelMetrics objects that load_metrics.py already validated (equal
    required_metrics per model/horizon, numeric values, etc.).
    """
    pivoted: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for model_name, per_horizon in all_metrics.items():
        for horizon_name, model_metrics in per_horizon.items():
            horizon_bucket = pivoted.setdefault(horizon_name, {})
            for metric_name, values in model_metrics.metrics.items():
                horizon_bucket.setdefault(metric_name, {})[model_name] = values
    return pivoted

def _validate_or_raise(
    horizon: str,
    metric: str,
    model_runs: Dict[str, List[float]],
    proposed_model: str,
) -> None:
    ctx = f"(horizon='{horizon}', metric='{metric}')"

    if not model_runs:
        raise ValueError(f"No models found for {ctx}.")

    if proposed_model not in model_runs:
        raise ValueError(
            f"Required model '{proposed_model}' missing for {ctx}. This "
            f"means '{proposed_model}' has no data for this horizon in "
            f"the loader output (load_metrics.py discovers horizons "
            f"per-model, so a model can legitimately be missing one)."
        )

    run_counts = {m: len(v) for m, v in model_runs.items()}
    unique_counts = set(run_counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            f"Unequal run counts across models for {ctx}: {run_counts}. "
            f"Friedman requires paired (equal-length) samples."
        )

    n_runs = unique_counts.pop()
    if n_runs < 2:
        raise ValueError(
            f"At least 2 runs are required per model for {ctx}, got "
            f"{n_runs}."
        )


# ======================================================================
# Friedman test
# ======================================================================


def _run_friedman(
    samples: List[np.ndarray], model_names: List[str], alpha: float = ALPHA_DEFAULT
) -> FriedmanResult:
    statistic, p_value = friedmanchisquare(*samples)
    return FriedmanResult(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        n_models=len(samples),
        n_runs=len(samples[0]),
    )