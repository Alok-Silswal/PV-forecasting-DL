"""
analysis/run_statistical_analysis.py

Executable entry point for the complete statistical analysis
pipeline. Running

    python analysis/run_statistical_analysis.py

calls `run_analysis()`, saves the resulting publication tables to
disk, writes a plain-text summary, and prints a concise console
report. It duplicates no statistical logic, no table-shaping logic,
and no filesystem-traversal logic -- everything of substance happens
in `analysis.run_analysis` and the modules it orchestrates; this
script is I/O (writing results) and presentation (console output)
only.

====================================================================
OUTPUT DIRECTORY
====================================================================
DECISION: ACCEPT adding one new constant to config.py,
`STATISTICAL_ANALYSIS_DIR`; REJECT reusing `EVALUATION_DIR` or any
other existing constant.

JUSTIFICATION:
  - `EVALUATION_DIR` is the pipeline's *input* -- it's the directory
    `load_all_metrics` scans, and it discovers models/horizons by
    listing directory entries rather than an explicit allowlist
    (see `discover_models` / `_discover_horizon_dirs` in
    load_metrics.py). Writing this script's *output* files into that
    same tree would risk a future loader run treating a stray output
    file/directory as model data. Input and output directories must
    stay disjoint for a loader that discovers its inputs by scanning.
  - No other existing constant in config.py is a plausible fit:
    `ARTIFACT_DIR` is for preprocessing artifacts (scalers, tensors),
    `EXPERIMENTS_DIR` is per-run training output, `MODELS_DIR` is
    unused elsewhere in the shown config but is clearly for model
    files, not cross-model statistical summaries. None of them
    represent "the place cross-model summary results live" -- inventing
    a use for one of them would be a worse fit than adding one
    single-purpose constant.
  - The project's existing top-level layout is flat
    (data/, artifacts/, models/, experiments/, evaluation/); a new
    top-level `results/` directory matches that convention, and
    `results/statistical_analysis/` nested under it leaves room for
    other kinds of results later without a second config edit.
  - This is one new constant, not a duplicate -- `EVALUATION_DIR`
    remains the only constant for its purpose (loader input);
    `STATISTICAL_ANALYSIS_DIR` is the only constant for this new
    purpose (analysis output).

====================================================================
ERROR HANDLING
====================================================================
DECISION: REJECT wrapping `run_analysis()` (or anything else) in a
try/except.

JUSTIFICATION: `run_analysis` already propagates
`MetricLoaderError` and any exception from the statistics layers
unchanged and on purpose (see run_analysis.py's own "ERROR HANDLING"
section). A try/except here that prints a friendlier message and
exits would strip the specific, actionable context those exceptions
already carry (which file, which model/horizon/metric) for no
compensating benefit -- and for a single-command CLI script, an
uncaught exception with Python's default traceback IS the correct
and complete error report; there is no partial/degraded output this
script could usefully produce after a fatal pipeline failure. The
only genuinely new failure mode this script itself introduces is
disk I/O when saving files (permissions, disk full); Python's default
tracebacks for `OSError` are already specific enough that catching
and rethrowing would add nothing.

====================================================================
FILE SAVING
====================================================================
Uses `DataFrame.to_csv` and `pandas.ExcelWriter` exclusively, per the
brief -- no manual CSV writing. `ExcelWriter` is used as a context
manager (the documented-correct way to guarantee the workbook is
flushed/closed) writing all three sheets to one workbook in one
`with` block, rather than three separate `ExcelWriter` opens, since
they belong to a single logical workbook.

====================================================================
SUMMARY CONTENT
====================================================================
The summary (both the text file and the console block) reports only
figures that are NOT already fully visible in the three CSV tables:
counts, not the row-level data itself (which is what the CSV/XLSX
files are for). Model/horizon names are listed because "how many"
alone doesn't tell a reader *which* models/horizons were covered,
and that identifying information does not require opening a table to
get.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import EVALUATION_DIR, STATISTICAL_ANALYSIS_DIR  # noqa: E402
from analysis.run_analysis import AnalysisResults, run_analysis  # noqa: E402

DESCRIPTIVE_CSV_NAME = "descriptive_statistics.csv"
FRIEDMAN_CSV_NAME = "friedman_results.csv"
AVERAGE_RANKS_CSV_NAME = "average_ranks.csv"
WORKBOOK_NAME = "statistical_analysis.xlsx"
SUMMARY_NAME = "analysis_summary.txt"


def _save_tables(results: AnalysisResults, output_dir: Path) -> list[Path]:
    """Save the three publication tables as CSV and as one XLSX workbook.

    Args:
        results: Completed pipeline output from `run_analysis`.
        output_dir: Directory the files are written into. Must already
            exist.

    Returns:
        Paths to every file written, in the order they were written.
    """
    descriptive_df = results.tables["descriptive"]
    friedman_df = results.tables["friedman"]
    average_ranks_df = results.tables["average_ranks"]

    descriptive_path = output_dir / DESCRIPTIVE_CSV_NAME
    friedman_path = output_dir / FRIEDMAN_CSV_NAME
    average_ranks_path = output_dir / AVERAGE_RANKS_CSV_NAME
    workbook_path = output_dir / WORKBOOK_NAME

    descriptive_df.to_csv(descriptive_path, index=False)
    friedman_df.to_csv(friedman_path, index=False)
    average_ranks_df.to_csv(average_ranks_path, index=False)

    with pd.ExcelWriter(workbook_path) as writer:
        descriptive_df.to_excel(writer, sheet_name="Descriptive Statistics", index=False)
        friedman_df.to_excel(writer, sheet_name="Friedman Test", index=False)
        average_ranks_df.to_excel(writer, sheet_name="Average Ranks", index=False)

    return [descriptive_path, friedman_path, average_ranks_path, workbook_path]


def _build_summary_lines(
    results: AnalysisResults,
    *,
    timestamp: str,
    evaluation_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Build the shared summary content used by both the text file and console.

    Args:
        results: Completed pipeline output from `run_analysis`.
        timestamp: ISO-8601 UTC timestamp string for this run.
        evaluation_dir: The evaluation directory that was analysed.
        output_dir: The directory results were written to.

    Returns:
        Lines of the summary, without trailing newlines.
    """
    models = sorted(results.descriptive.keys())
    horizons = sorted(
        {horizon for per_horizon in results.descriptive.values() for horizon in per_horizon}
    )

    friedman_df = results.tables["friedman"]
    average_ranks_df = results.tables["average_ranks"]

    n_friedman = len(friedman_df)
    n_friedman_significant = (
        int(friedman_df["Significant"].sum()) if not friedman_df.empty else 0
    )
    average_rank_table_generated = not average_ranks_df.empty

    return [
        f"Timestamp: {timestamp}",
        f"Evaluation directory: {evaluation_dir}",
        f"Output directory: {output_dir}",
        f"Models analysed ({len(models)}): {', '.join(models)}",
        f"Horizons analysed ({len(horizons)}): {', '.join(horizons)}",
        f"Descriptive warnings: {len(results.descriptive_warnings)}",
        f"Number of Friedman tests performed: {n_friedman}",
        f"Number of significant Friedman tests: {n_friedman_significant}",
        f"Average rank table generated: {average_rank_table_generated}",
    ]


def main() -> None:
    """Run the full pipeline, save all outputs, and print a summary."""
    output_dir = STATISTICAL_ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_analysis(EVALUATION_DIR)

    saved_paths = _save_tables(results, output_dir)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary_lines = _build_summary_lines(
        results,
        timestamp=timestamp,
        evaluation_dir=EVALUATION_DIR,
        output_dir=output_dir,
    )

    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    saved_paths.append(summary_path)

    separator = "-" * 40
    print(separator)
    print("Statistical analysis completed.")
    print()
    for line in summary_lines:
        print(line)
    print()
    print(f"Files generated ({len(saved_paths)}):")
    for path in saved_paths:
        print(f"  - {path}")
    print(separator)


if __name__ == "__main__":
    main()