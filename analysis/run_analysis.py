"""
analysis/run_analysis.py

Top-level orchestration for the PV power forecasting statistical
analysis pipeline.

This module coordinates, in order:

    1. analysis.load_metrics.load_all_metrics
    2. analysis.descriptive_statistics.compute_descriptive_statistics
    3. analysis.significance_tests.run_significance_analysis
    4. analysis.generate_tables.generate_all_tables

It performs NO statistics, NO table reshaping, NO filesystem
traversal, and NO re-validation of data already validated by those
modules. Its only job is to call them in the right order, thread
results between them, and hand everything back to the caller in one
object.

====================================================================
RETURN TYPE: OPTION A / B / C
====================================================================
DECISION: ACCEPT Option C -- a dedicated `AnalysisResults` dataclass.

JUSTIFICATION:
  - REJECTED Option A (tables only): this is a real loss of
    information, not a simplification. `MetricStatistics.n` and
    `.std is None` (the N=1 case) do not survive the trip through
    `generate_all_tables` unchanged in a form a caller could still act
    on programmatically (the descriptive table does carry N and
    SD=NaN, but that is the only place this distinction is visible;
    a caller working from tables alone still cannot recover the raw
    `MetricStatistics` objects).
  - REJECTED Option B (a bare tuple/dict of the four pieces): callers
    would need to memorize positional order or a set of dict keys
    with no static-typing help (`results["descriptive"]` vs
    `results.descriptive_stats` vs ...). This is exactly the
    "string keys with no static checking" problem `significance_tests`
    itself already rejected in favor of dataclasses -- reintroducing
    it one layer up, for a result that itself aggregates typed
    dataclasses, would be inconsistent.
  - Option C matches the project's own established convention:
    `descriptive_statistics.py` and `significance_tests.py` both use
    frozen dataclasses for exactly this reason (explicit contract,
    IDE/type-checker support, immutability of a finished result).
    `AnalysisResults` is the natural top-level record of that same
    convention: one object, attribute access, and it can grow new
    fields later (e.g. a run timestamp or config) without breaking
    every call site that would otherwise be unpacking a tuple.

====================================================================
WARNINGS
====================================================================
DECISION: REJECT merging warnings into a single flat list; ACCEPT
keeping them attributable to the module (and, for significance
results, the cell) that produced them.

JUSTIFICATION:
  - `load_all_metrics` does NOT return its warnings -- it only prints
    them via an internal `_print_summary` call and returns the
    results mapping. This module cannot expose "loader warnings" as a
    return value because the loader's public API does not surface
    them; inventing a way to capture stdout or reaching into private
    loader internals to recover them would duplicate/circumvent the
    loader's own interface, which this module is explicitly told not
    to do. This is documented on `AnalysisResults.loader_warnings`
    below rather than silently omitted.
  - `compute_descriptive_statistics` returns `(results, warnings)` as
    a flat list already scoped to "descriptive statistics computation"
    -- that scoping is preserved as-is (`descriptive_warnings`) rather
    than merged into a pipeline-wide soup, so a reader can still tell
    an N=1 warning from an average-mismatch warning apart from any
    significance-analysis warning by looking at which list it's in.
  - `run_significance_analysis` no longer produces any warnings:
    its only source of warnings was the pairwise Wilcoxon/Holm stage
    (e.g. a missing baseline, or all-zero paired differences), and
    that stage has been removed along with `MetricSignificanceResult
    .pairwise` and `.warnings`. `AnalysisResults` therefore exposes
    the significance results unchanged (`significance`) with no
    accompanying `significance_warnings` property -- there is nothing
    left for it to flatten.
  - Logging is rejected as the primary channel: this is a library
    function, not an application entry point, and warnings that are
    only visible in log output are not inspectable/testable by a
    notebook or a unit test the way a returned list is. `load_metrics`
    already made a similar choice (print instead of return) for its
    own reasons; this module does not compound that by *also* logging
    instead of returning where a return value can be given.

====================================================================
EXPORT
====================================================================
DECISION: ACCEPT -- this module writes no files.

JUSTIFICATION: File export is a caller/notebook decision (which
format, which path, which sheet names) exactly as already established
in `generate_tables.py`. Baking a CSV/Excel/JSON/pickle write into the
orchestration layer would force one opinion about output location on
every caller, including tests, which should be able to call
`run_analysis` in-memory with no filesystem side effects beyond the
read `load_all_metrics` itself performs.

====================================================================
ERROR HANDLING
====================================================================
DECISION: ACCEPT letting exceptions propagate; REJECT catching and
re-wrapping them.

JUSTIFICATION: `MetricLoaderError` from `load_metrics` and any `scipy`
exceptions from `significance_tests` already carry a specific,
actionable message about *what* failed and *where*
(e.g. which model/horizon/metric). Catching them here and re-raising a
generic `AnalysisError` would strip that context for no benefit --
this module adds no information a caller doesn't already get from the
original exception, so a try/except here would exist only to satisfy
a "this module should handle errors" reflex rather than to genuinely
improve the API. The one exception boundary worth adding is *none*:
there is no partial/degraded mode this pipeline can usefully continue
in after a fatal error from an earlier stage (each stage's output is
required input to the next), so propagation is strictly correct here,
not merely convenient.

====================================================================
PUBLIC API
====================================================================
A single public function, `run_analysis`, plus the `AnalysisResults`
dataclass it returns. No other helper functions are exposed; the
pipeline has exactly one linear path through it and does not need
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from analysis.descriptive_statistics import (
    StatisticsResults,
    compute_descriptive_statistics,
)
from analysis.generate_tables import PublicationTables, generate_all_tables
from analysis.load_metrics import DEFAULT_REQUIRED_METRICS, load_all_metrics
from analysis.significance_tests import (
    ALPHA_DEFAULT,
    SignificanceResults,
    run_significance_analysis,
)

__all__ = ["AnalysisResults", "run_analysis"]


@dataclass(frozen=True)
class AnalysisResults:
    """Complete output of the end-to-end statistical analysis pipeline.

    Attributes:
        descriptive: Per-model, per-horizon, per-metric descriptive
            statistics, as returned by
            `compute_descriptive_statistics`.
        descriptive_warnings: Non-fatal warnings collected while
            computing descriptive statistics (e.g. N=1 cells,
            declared-average/computed-mean mismatches).
        significance: Per-horizon, per-metric Friedman results, as
            returned by `run_significance_analysis`. The pairwise
            Wilcoxon/Holm stage has been removed, so
            `MetricSignificanceResult` no longer carries a `pairwise`
            or `warnings` field -- there is nothing left to flatten
            here, so this dataclass no longer exposes a
            `significance_warnings` property.
        tables: The three publication-ready DataFrames ("descriptive",
            "friedman", "average_ranks"), as returned by
            `generate_all_tables`.
        loader_warnings: Always an empty tuple. `load_all_metrics`
            does not return its warnings -- it only prints them via an
            internal summary -- so no loader warnings are available to
            surface here. Kept as an explicit field (rather than
            omitted) so callers relying on `AnalysisResults` don't hit
            an `AttributeError` if the loader's API is later changed
            to return its warnings.
    """

    descriptive: StatisticsResults
    descriptive_warnings: tuple[str, ...]
    significance: SignificanceResults
    tables: PublicationTables
    loader_warnings: tuple[str, ...] = ()


def run_analysis(
    evaluation_dir: Path | str,
    *,
    horizons: tuple[str, ...] | None = None,
    required_metrics: tuple[str, ...] = DEFAULT_REQUIRED_METRICS,
    alpha: float = ALPHA_DEFAULT,
    proposed_model: str = "proposed",
) -> AnalysisResults:
    """Run the full statistical analysis pipeline end to end.

    Loads raw per-run metrics, computes descriptive statistics, runs
    the Friedman significance pipeline, and builds publication-ready
    tables -- in that order, threading each stage's output into the
    next.

    Args:
        evaluation_dir: Path to the top-level `evaluation/` directory,
            forwarded to `load_all_metrics`.
        horizons: Optional explicit set of horizon directory names to
            load. Forwarded to `load_all_metrics`; see that function
            for details.
        required_metrics: Metric names that must be present in every
            metrics file. Forwarded to `load_all_metrics`.
        alpha: Significance threshold for the Friedman test. Forwarded
            to `run_significance_analysis`.
        proposed_model: Name of the model required to be present in
            every (horizon, metric) cell. Forwarded to
            `run_significance_analysis`, which still validates its
            presence even though pairwise comparisons against it have
            been removed.

    Returns:
        An `AnalysisResults` instance containing every stage's output.

    Raises:
        analysis.load_metrics.MetricLoaderError: If loading fails, per
            `load_all_metrics`. Propagated unchanged; see module
            docstring ("ERROR HANDLING").
        Exception: Any exception raised by
            `compute_descriptive_statistics` or
            `run_significance_analysis` is likewise propagated
            unchanged.
    """
    all_metrics = load_all_metrics(
        evaluation_dir,
        horizons=horizons,
        required_metrics=required_metrics,
    )

    descriptive, descriptive_warnings = compute_descriptive_statistics(all_metrics)

    significance = run_significance_analysis(
        all_metrics,
        alpha=alpha,
        proposed_model=proposed_model,
    )

    tables = generate_all_tables(descriptive, significance, all_metrics)

    return AnalysisResults(
        descriptive=descriptive,
        descriptive_warnings=tuple(descriptive_warnings),
        significance=significance,
        tables=tables,
    )