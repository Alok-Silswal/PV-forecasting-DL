"""
analysis/descriptive_statistics.py

Descriptive statistics for the PV power forecasting statistical
analysis package.

This module computes per-model, per-horizon, per-metric descriptive
statistics (mean, sample standard deviation, run count) from the
clean objects produced by `analysis.load_metrics.load_all_metrics`.

It performs NO filesystem access, NO significance testing (Friedman,
Wilcoxon, Holm correction, effect sizes), and NO export (CSV, Excel,
plots). Those responsibilities belong to later modules in this
package.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from analysis.load_metrics import ModelMetrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Relative tolerance used when comparing computed means against the
#: values found in average_metrics.json (loaded via ModelMetrics).
AVERAGE_MISMATCH_RELATIVE_TOLERANCE: float = 1e-3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricStatistics:
    """Descriptive statistics for a single metric's runs.

    Attributes:
        mean: Arithmetic mean of the observed values across runs.
        std: Sample standard deviation (ddof=1) across runs, or None
            if fewer than 2 runs are available (a sample standard
            deviation is undefined for a single observation). None is
            used instead of NaN so that downstream code is forced to
            handle the missing-variance case explicitly rather than
            silently propagating NaN through further arithmetic.
        n: Number of runs the statistics were computed from.
    """

    mean: float
    std: float | None
    n: int

#: Nested statistics structure: model -> horizon -> metric -> MetricStatistics.
StatisticsResults = dict[str, dict[str, dict[str, MetricStatistics]]]
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_metric_statistics(
    values: list[float],
    *,
    model_name: str,
    horizon_name: str,
    metric_name: str,
    warnings: list[str],
) -> MetricStatistics:
    """Compute mean, sample std, and run count for one metric's values.

    Args:
        values: Per-run observed values for this metric.
        model_name: Name of the model, used in warning messages.
        horizon_name: Name of the horizon, used in warning messages.
        metric_name: Name of the metric, used in warning messages.
        warnings: List to which non-fatal warning messages are
            appended.

    Returns:
        A `MetricStatistics` instance summarizing `values`.
    """

    assert values, (
        f"Model '{model_name}' ({horizon_name}, {metric_name}): received "
        "an empty values list; this violates the invariant guaranteed by "
        "load_metrics.load_all_metrics (every ModelMetrics.metrics entry "
        "must have at least one run)."
    )

    n = len(values)
    mean = statistics.fmean(values)

    if n < 2:
        warnings.append(
            f"Model '{model_name}' ({horizon_name}, {metric_name}): only "
            f"{n} run available; sample standard deviation is undefined "
            "and reported as None."
        )
        return MetricStatistics(mean=mean, std=None, n=n)

    std = statistics.stdev(values)  # ddof=1 (sample standard deviation)
    return MetricStatistics(mean=mean, std=std, n=n)


def _check_average_against_computed(
    model_name: str,
    horizon_name: str,
    average_metrics: dict[str, float],
    computed_stats: dict[str, MetricStatistics],
    *,
    warnings: list[str],
) -> None:
    """Warn if declared averages differ from the computed means.

    Args:
        model_name: Name of the model, used in warning messages.
        horizon_name: Name of the horizon, used in warning messages.
        average_metrics: Metrics read from average_metrics.json (may
            be empty if that file was missing).
        computed_stats: Metric name -> computed `MetricStatistics` for
            this model/horizon.
        warnings: List to which warning messages are appended.
    """
    if not average_metrics:
        return

    for metric_name, stats in computed_stats.items():
        declared_value = average_metrics.get(metric_name)
        if declared_value is None:
            continue

        tolerance = AVERAGE_MISMATCH_RELATIVE_TOLERANCE * max(
            abs(stats.mean), abs(declared_value), 1e-12
        )

        if abs(declared_value - stats.mean) > tolerance:
            warnings.append(
                f"Model '{model_name}' ({horizon_name}): declared average "
                f"'{metric_name}' ({declared_value}) differs from computed "
                f"mean of runs ({stats.mean})."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_descriptive_statistics(
    all_metrics: dict[str, dict[str, ModelMetrics]],
) -> tuple[StatisticsResults, list[str]]:
    """Compute descriptive statistics for every model, horizon, and metric.

    Args:
        all_metrics: Output of `analysis.load_metrics.load_all_metrics`,
            i.e. a mapping of model name -> horizon name ->
            `ModelMetrics`.

    Returns:
        A tuple of:
            - Nested mapping model -> horizon -> metric ->
              `MetricStatistics`, mirroring the structure of
              `all_metrics` so callers can iterate both consistently.
            - A list of collected warning messages (e.g. N=1 cases,
              average/mean mismatches). No exceptions are raised for
              these conditions; execution always continues.
    """
    warnings: list[str] = []
    results: StatisticsResults = {}

    for model_name, per_horizon in all_metrics.items():
        results[model_name] = {}

        for horizon_name, model_metrics in per_horizon.items():
            per_metric_stats: dict[str, MetricStatistics] = {}

            for metric_name, values in model_metrics.metrics.items():
                per_metric_stats[metric_name] = _compute_metric_statistics(
                    values,
                    model_name=model_name,
                    horizon_name=horizon_name,
                    metric_name=metric_name,
                    warnings=warnings,
                )

            _check_average_against_computed(
                model_name=model_name,
                horizon_name=horizon_name,
                average_metrics=model_metrics.average_metrics,
                computed_stats=per_metric_stats,
                warnings=warnings,
            )

            results[model_name][horizon_name] = per_metric_stats

    return results, warnings