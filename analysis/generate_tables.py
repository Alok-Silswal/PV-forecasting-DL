"""
analysis/generate_tables.py

Converts already-computed statistical results into publication-ready
tables.

This module performs NO filesystem loading, NO descriptive-statistics
computation, and NO significance testing (Friedman, Wilcoxon, Holm
correction, effect sizes). It consumes the outputs of

    analysis.descriptive_statistics.compute_descriptive_statistics
    analysis.significance_tests.run_significance_analysis

exactly as those modules define them, and reshapes the results into
flat tables suitable for a paper or report.

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
run counts for Wilcoxon). This module does not re-derive or re-check
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

import pandas as pd

from analysis.descriptive_statistics import StatisticsResults
from analysis.significance_tests import SignificanceResults

PublicationTables = dict[str, pd.DataFrame]

__all__ = [
    "generate_descriptive_table",
    "generate_friedman_table",
    "generate_pairwise_table",
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
# Table 3: Pairwise comparisons
# ---------------------------------------------------------------------------

def generate_pairwise_table(significance: SignificanceResults) -> pd.DataFrame:
    """Build the pairwise (Proposed vs. baseline) comparison table.

    One row per (horizon, metric, baseline). Columns: Horizon, Metric,
    Baseline, Raw p-value, Holm-adjusted p-value, Significant, Effect
    size, Effect size method, N effective.

    Rows for a (horizon, metric) cell with no baselines compared (e.g.
    only one model present) are simply absent -- there is nothing to
    tabulate, and fabricating a placeholder row would misrepresent the
    underlying analysis.

    Args:
        significance: Output of
            `analysis.significance_tests.run_significance_analysis`.

    Returns:
        A DataFrame sorted by Horizon, Metric, Holm-adjusted p-value.
    """
    columns = [
        "Horizon",
        "Metric",
        "Baseline",
        "Raw p-value",
        "Holm-adjusted p-value",
        "Significant",
        "Effect size",
        "Effect size method",
        "N effective",
    ]
    rows: list[dict[str, object]] = []

    for horizon_name, per_metric in significance.items():
        for metric_name, result in per_metric.items():
            for pairwise in result.pairwise:
                rows.append(
                    {
                        "Horizon": horizon_name,
                        "Metric": metric_name,
                        "Baseline": pairwise.baseline,
                        "Raw p-value": pairwise.raw_p,
                        "Holm-adjusted p-value": pairwise.adjusted_p,
                        "Significant": pairwise.significant,
                        "Effect size": pairwise.effect_size,
                        "Effect size method": pairwise.effect_size_method,
                        "N effective": pairwise.n_effective,
                    }
                )

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    return df.sort_values(
        ["Horizon", "Metric", "Holm-adjusted p-value"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: build all three tables at once
# ---------------------------------------------------------------------------

def generate_all_tables(
    stats: StatisticsResults,
    significance: SignificanceResults,
) -> PublicationTables:
    """Build all three publication tables in one call.

    Args:
        stats: Output of `compute_descriptive_statistics`.
        significance: Output of `run_significance_analysis`.

    Returns:
        Dict with keys "descriptive", "friedman", "pairwise" mapping
        to the corresponding DataFrame.
    """
    return {
        "descriptive": generate_descriptive_table(stats),
        "friedman": generate_friedman_table(significance),
        "pairwise": generate_pairwise_table(significance),
    }