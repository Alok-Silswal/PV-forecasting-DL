"""
analysis/generate_tables.py

Converts already-computed statistical results into publication-ready
tables.

This module performs NO filesystem loading, NO descriptive-statistics
computation, and NO significance testing (Friedman). It consumes the
outputs of

    analysis.descriptive_statistics.compute_descriptive_statistics
    analysis.significance_tests.run_significance_analysis

exactly as those modules define them, and reshapes the results into
flat tables suitable for a paper or report. The average-rank table
additionally consumes the loader's raw `all_metrics` structure
directly (see Table 3 below), since per-run values -- not just the
Friedman summary -- are needed to compute ranks.

====================================================================
EXPORT FORMAT: DECISION
====================================================================
DECISION: ACCEPT pandas DataFrames as the return type; REJECT writing
CSV/Excel directly from this module.

JUSTIFICATION:
  - DataFrames give the caller (a notebook, a script, a test) a single
    well-known object with `.to_csv`, `.to_excel`, `.to_latex`,
    `.style`, etc. already implemented correctly (quoting, encoding,
    float formatting). Reimplementing any of that by hand here would
    be strictly worse and untested.
  - Writing files is an I/O side effect. The module docstring for this
    package draws a hard line between "compute/shape" modules (no I/O)
    and "load" modules (I/O only). Table *generation* is a shaping
    step, not an export step; forcing a file write here would couple
    table construction to a choice of output path/format the caller
    may not want (some callers want to render the table inline, e.g.
    in a notebook, not write it to disk at all).
  - This module therefore returns DataFrames only. A caller that wants
    CSV/Excel calls `df.to_csv(...)` / `df.to_excel(...)` themselves.
    This is a thin, composable surface rather than a hidden opinion
    about file paths, sheet names, or formats.

====================================================================
IS PANDAS THE RIGHT ABSTRACTION?
====================================================================
DECISION: ACCEPT pandas for these three tables specifically.

JUSTIFICATION:
  - All three tables are exactly the shape pandas is for: a flat list
    of homogeneous records with a handful of named fields, destined
    for either (a) display as a table, (b) export as CSV/Excel/LaTeX,
    or (c) further slicing/filtering/sorting by the caller (e.g.
    "show me only the significant rows", "sort by adjusted p-value").
    Plain lists of dataclasses would force every caller to re-derive
    this filtering/sorting/exporting logic themselves.
  - REJECTED alternative: returning the raw dataclasses/lists
    unchanged. That is literally the input this module receives, so
    it would make this module a no-op; it does not satisfy "generate
    publication tables".
  - REJECTED alternative: hand-rolled nested dicts formatted as
    strings ("12.34 ± 0.56"). Useful as a *display* convenience but
    actively harmful as the primary return value, because it destroys
    numeric type information a downstream consumer might need (e.g.
    to sort by mean, or re-check a threshold). This module returns
    numeric columns and provides formatted string columns
    *additionally* (see `mean_std` in Table 1), never instead of the
    numeric ones.
  - No other structure (numpy structured arrays, arrow tables, raw
    dict-of-dicts) offers a comparable ubiquity/tooling benefit for
    something whose end destination is "goes in a paper", so pandas
    is not a default reached for blindly here -- it is the structure
    that matches both the shape of the data and its downstream uses.

====================================================================
VALIDATION
====================================================================
Earlier modules (`compute_descriptive_statistics`,
`run_significance_analysis`) are the source of statistical truth and
have already validated their inputs (e.g. non-empty run lists, minimum
run counts for Friedman). This module does not re-derive or re-check
statistical correctness. It performs only lightweight structural
validation appropriate to a formatting layer:
  - guard against a completely empty input (produces an empty,
    correctly-columned DataFrame rather than raising or returning
    `None`, so callers can always call `.empty`/`.to_csv` uniformly);
  - no silent coercion of missing values -- `std=None` (N=1 case) is
    preserved as pandas' `NA`/NaN, not silently dropped or fabricated
    as 0.0, so a reader of the table can still tell N=1 apart from a
    genuinely zero variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analysis.descriptive_statistics import StatisticsResults
from analysis.load_metrics import ModelMetrics
from analysis.significance_tests import SignificanceResults

PublicationTables = dict[str, pd.DataFrame]

__all__ = [
    "generate_descriptive_table",
    "generate_friedman_table",
    "generate_average_rank_table",
    "generate_all_tables",
    "PublicationTables"
]


# ---------------------------------------------------------------------------
# Table 1: Descriptive statistics
# ---------------------------------------------------------------------------

def generate_descriptive_table(stats: StatisticsResults) -> pd.DataFrame:
    """Build the descriptive-statistics table.

    One row per (model, horizon, metric). Columns: Model, Horizon,
    Metric, Mean, SD, N, and a formatted `Mean ± SD` convenience
    column for direct use in a manuscript.

    Args:
        stats: Output of
            `analysis.descriptive_statistics.compute_descriptive_statistics`
            (the results mapping only; its accompanying warnings list
            is not this module's concern).

    Returns:
        A DataFrame sorted by Model, Horizon, Metric.
    """
    columns = ["Model", "Horizon", "Metric", "Mean", "SD", "N", "Mean_SD"]
    rows: list[dict[str, object]] = []

    for model_name, per_horizon in stats.items():
        for horizon_name, per_metric in per_horizon.items():
            for metric_name, metric_stats in per_metric.items():
                mean = metric_stats.mean
                std = metric_stats.std
                n = metric_stats.n

                formatted = (
                    f"{mean:.4g} ± {std:.4g}"
                    if std is not None
                    else f"{mean:.4g} (N=1)"
                )

                rows.append(
                    {
                        "Model": model_name,
                        "Horizon": horizon_name,
                        "Metric": metric_name,
                        "Mean": mean,
                        "SD": std,
                        "N": n,
                        "Mean_SD": formatted,
                    }
                )

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    return df.sort_values(["Horizon", "Metric", "Model"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 2: Friedman test summary
# ---------------------------------------------------------------------------

def generate_friedman_table(significance: SignificanceResults) -> pd.DataFrame:
    """Build the Friedman omnibus-test summary table.

    One row per (horizon, metric). Columns: Horizon, Metric,
    Chi-square, p-value, Significant, N models, N runs.

    Args:
        significance: Output of
            `analysis.significance_tests.run_significance_analysis`,
            shape {horizon: {metric: MetricSignificanceResult}}.

    Returns:
        A DataFrame sorted by Horizon, Metric.
    """
    columns = [
        "Horizon",
        "Metric",
        "Chi-square",
        "p-value",
        "Significant",
        "N models",
        "N runs",
    ]
    rows: list[dict[str, object]] = []

    for horizon_name, per_metric in significance.items():
        for metric_name, result in per_metric.items():
            friedman = result.friedman
            rows.append(
                {
                    "Horizon": horizon_name,
                    "Metric": metric_name,
                    "Chi-square": friedman.statistic,
                    "p-value": friedman.p_value,
                    "Significant": friedman.significant,
                    "N models": friedman.n_models,
                    "N runs": friedman.n_runs,
                }
            )

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    return df.sort_values(["Horizon", "Metric"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 3: Average ranks
# ---------------------------------------------------------------------------
#
# DECISION: ACCEPT scipy.stats.rankdata(method="average") for ranking;
# REJECT a hand-rolled sort-based rank assignment.
#
# JUSTIFICATION:
#   - `rankdata` is the established, tested implementation for this
#     exact operation and handles tied values by splitting the tied
#     rank positions evenly (e.g. two models tied for best both get
#     rank 1.5), which a manual `sorted()`-based assignment would get
#     wrong (it would silently break ties by insertion/comparison
#     order instead of reporting the tie honestly).
#   - Ranking direction (lower-is-better vs. higher-is-better) is
#     handled by ranking the negated values for higher-is-better
#     metrics, rather than writing a second, direction-aware ranking
#     routine -- `rankdata` itself stays the single source of ranking
#     logic for both directions.

#: Metrics for which a LOWER observed value is better (rank 1 = lowest).
_LOWER_IS_BETTER_METRICS = {"rmse", "mae", "nrmse", "mape"}
#: Metrics for which a HIGHER observed value is better (rank 1 = highest).
_HIGHER_IS_BETTER_METRICS = {"r2"}


def generate_average_rank_table(
    all_metrics: dict[str, dict[str, ModelMetrics]],
) -> pd.DataFrame:
    """Build the average-rank table.

    For every (horizon, metric) cell, models are ranked against each
    other independently for each run (rank 1 = best, per the metric's
    known directionality; ties split evenly via
    `scipy.stats.rankdata(method="average")`). Ranks are then averaged
    across runs to give one Average Rank per (horizon, metric, model).

    Args:
        all_metrics: Output of `analysis.load_metrics.load_all_metrics`,
            shape {model_name: {horizon_name: ModelMetrics}}. Raw
            per-run values are required here (not the Friedman
            summary), since ranks must be computed run-by-run.

    Returns:
        A DataFrame with columns Horizon, Metric, Model, Average Rank,
        sorted by Horizon, Metric, Average Rank.
    """
    columns = ["Horizon", "Metric", "Model", "Average Rank"]

    # Reshape model->horizon->ModelMetrics into horizon->metric->model->
    # [runs], the axis order per-cell ranking needs. Mirrors the pivot
    # performed privately in significance_tests.py, but is kept local to
    # this module rather than importing that module's private helper.
    pivoted: dict[str, dict[str, dict[str, list[float]]]] = {}
    for model_name, per_horizon in all_metrics.items():
        for horizon_name, model_metrics in per_horizon.items():
            horizon_bucket = pivoted.setdefault(horizon_name, {})
            for metric_name, values in model_metrics.metrics.items():
                horizon_bucket.setdefault(metric_name, {})[model_name] = values

    rows: list[dict[str, object]] = []

    for horizon_name, per_metric in pivoted.items():
        for metric_name, model_runs in per_metric.items():
            higher_is_better = metric_name.lower() in _HIGHER_IS_BETTER_METRICS

            model_names = list(model_runs.keys())
            # shape (n_models, n_runs)
            values = np.array([model_runs[m] for m in model_names], dtype=float)

            # Rank each run (column) independently across models, then
            # average the per-run ranks for each model across runs.
            per_run_ranks = np.array(
                [
                    rankdata(
                        -values[:, run_idx] if higher_is_better else values[:, run_idx],
                        method="average",
                    )
                    for run_idx in range(values.shape[1])
                ]
            )  # shape (n_runs, n_models)
            average_ranks = per_run_ranks.mean(axis=0)

            for model_name, average_rank in zip(model_names, average_ranks):
                rows.append(
                    {
                        "Horizon": horizon_name,
                        "Metric": metric_name,
                        "Model": model_name,
                        "Average Rank": float(average_rank),
                    }
                )

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    return df.sort_values(
        ["Horizon", "Metric", "Average Rank"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: build all three tables at once
# ---------------------------------------------------------------------------

def generate_all_tables(
    stats: StatisticsResults,
    significance: SignificanceResults,
    all_metrics: dict[str, dict[str, ModelMetrics]],
) -> PublicationTables:
    """Build all three publication tables in one call.

    Args:
        stats: Output of `compute_descriptive_statistics`.
        significance: Output of `run_significance_analysis`.
        all_metrics: Output of `load_all_metrics`, needed by the
            average-rank table (see `generate_average_rank_table`).

    Returns:
        Dict with keys "descriptive", "friedman", "average_ranks"
        mapping to the corresponding DataFrame.
    """
    return {
        "descriptive": generate_descriptive_table(stats),
        "friedman": generate_friedman_table(significance),
        "average_ranks": generate_average_rank_table(all_metrics),
    }