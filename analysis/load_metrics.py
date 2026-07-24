"""
analysis/load_metrics.py

Metric loading utilities for the PV power forecasting statistical
analysis package.

This module is responsible ONLY for discovering models and horizons,
reading their per-run `evaluation_metrics.json` files (and, if
present, `average_metrics.json`), validating the raw data, and
returning clean Python objects for downstream statistical modules.

No statistical computation (means, tests, corrections), aggregation
logic, or file export is performed here.

Expected directory layout::

    evaluation/
        <model_name>/
            <horizon_name>/          e.g. horizon_15, horizon_30, ...
                average/
                    average_metrics.json
                run_1/
                    results/
                        evaluation_metrics.json
                run_2/
                    results/
                        evaluation_metrics.json
                ...

Models AND horizons are discovered automatically by scanning the
`evaluation/` directory; nothing is hardcoded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the directory holding the averaged metrics for a model/horizon.
AVERAGE_DIRNAME: str = "average"

#: Filename of the averaged metrics file.
AVERAGE_METRICS_FILENAME: str = "average_metrics.json"

#: Filename of the per-run evaluation metrics file.
EVALUATION_METRICS_FILENAME: str = "evaluation_metrics.json"

#: Sub-path (relative to a run directory) where the metrics file lives.
RESULTS_DIRNAME: str = "results"

#: Default metrics that MUST be present in every evaluation_metrics.json.
#: Callers may override this via the `required_metrics` parameter to
#: adapt the loader to future projects with different metric sets.
DEFAULT_REQUIRED_METRICS: tuple[str, ...] = ("rmse", "mae", "r2", "nrmse", "mape")

#: Pattern that a directory name must match to be treated as a run
#: directory (e.g. "run_1", "run_12"). Anything else (backup/, old/,
#: temp/, .ipynb_checkpoints/, ...) is ignored.
RUN_DIR_PATTERN: re.Pattern[str] = re.compile(r"^run_(\d+)$")

#: Pattern that a directory name must match to be treated as a horizon
#: directory (e.g. "horizon_15", "horizon_30").
HORIZON_DIR_PATTERN: re.Pattern[str] = re.compile(r"^horizon_(\d+)$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MetricLoaderError(Exception):
    """Raised for fatal errors that must stop execution of the loader."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Clean container for a single model/horizon's loaded metrics.

    Attributes:
        model_name: Name of the model (directory name under
            `evaluation/`).
        horizon: Name of the horizon directory this data was loaded
            from (e.g. "horizon_15").
        run_count: Number of individual runs discovered for this
            model/horizon.
        metrics: Mapping of metric name -> list of per-run values, in
            the order the runs were discovered (run_1, run_2, ...).
            Example: {"rmse": [11.4, 11.6], "mae": [5.0, 5.1], ...}
        average_metrics: Mapping of metric name -> value as read from
            `average_metrics.json`, or an empty dict if that file was
            missing. No comparison against computed means is done
            here; that belongs to the statistics modules.
    """

    model_name: str
    horizon: str
    run_count: int
    metrics: dict[str, list[float]] = field(default_factory=dict)
    average_metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content as a dictionary.

    Raises:
        MetricLoaderError: If the file cannot be read or does not
            contain valid JSON, or if the top-level JSON value is not
            an object.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MetricLoaderError(f"Could not read file '{path}': {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise MetricLoaderError(f"Invalid JSON in file '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise MetricLoaderError(
            f"Expected a JSON object at the top level of '{path}', "
            f"got {type(data).__name__}."
        )

    return data


def _normalize_metric_key(key: str) -> str:
    """Normalize a metric name for case-insensitive comparison.

    Args:
        key: Raw metric name as found in a JSON file.

    Returns:
        The lowercased metric name.
    """
    return key.strip().lower()


def _validate_and_extract_metrics(
    data: dict[str, Any],
    *,
    required_metrics: tuple[str, ...],
    source: Path,
    warnings: list[str],
) -> dict[str, float]:
    """Validate a metrics JSON payload and extract required metrics.

    Args:
        data: Parsed JSON content (metric name -> value).
        required_metrics: Metric names that must be present.
        source: Path the data was read from, used for error/warning
            messages.
        warnings: List to which non-fatal warning messages are
            appended.

    Returns:
        Mapping of required metric name -> numeric value.

    Raises:
        MetricLoaderError: If a required metric is missing or if a
            required metric's value is not numeric.
    """
    normalized: dict[str, Any] = {
        _normalize_metric_key(key): value for key, value in data.items()
    }

    extracted: dict[str, float] = {}
    for metric_name in required_metrics:
        if metric_name not in normalized:
            raise MetricLoaderError(
                f"Required metric '{metric_name}' missing in '{source}'."
            )

        value = normalized[metric_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricLoaderError(
                f"Metric '{metric_name}' in '{source}' is not numeric "
                f"(got {type(value).__name__}: {value!r})."
            )

        extracted[metric_name] = float(value)

    extra_keys = set(normalized) - set(required_metrics)
    if extra_keys:
        warnings.append(
            f"Extra metrics {sorted(extra_keys)} found in '{source}' "
            "and were ignored."
        )

    return extracted


def _discover_horizon_dirs(model_dir: Path) -> list[Path]:
    """Discover horizon directories (e.g. horizon_15, horizon_30) for a model.

    Only directories matching `HORIZON_DIR_PATTERN` are treated as
    horizons; other entries (docs, README, misc folders) are ignored
    silently since they are not part of the expected data contract at
    this level.

    Args:
        model_dir: Path to a model directory (e.g. `evaluation/cnn`).

    Returns:
        Sorted list of horizon directory paths (sorted numerically by
        the horizon value).
    """
    horizon_dirs = [
        entry
        for entry in model_dir.iterdir()
        if entry.is_dir() and HORIZON_DIR_PATTERN.match(entry.name)
    ]

    def sort_key(path: Path) -> int:
        match = HORIZON_DIR_PATTERN.match(path.name)
        assert match is not None  # guaranteed by the filter above
        return int(match.group(1))

    return sorted(horizon_dirs, key=sort_key)


def _discover_run_dirs(
    horizon_dir: Path,
    *,
    model_name: str,
    horizon_name: str,
    warnings: list[str],
) -> list[Path]:
    """Discover run directories (run_1, run_2, ...) under a horizon dir.

    Only directories matching `RUN_DIR_PATTERN` are treated as runs.
    Anything else (e.g. `backup/`, `old/`, `temp/`, `average/`) is
    ignored, aside from a low-noise informational warning so the user
    is aware an unrecognized directory was skipped.

    Args:
        horizon_dir: Path to the model's horizon directory.
        model_name: Name of the model, used in warning messages.
        horizon_name: Name of the horizon, used in warning messages.
        warnings: List to which warning messages are appended.

    Returns:
        Sorted list of run directory paths, sorted numerically by run
        index.
    """
    run_dirs: list[Path] = []
    for entry in horizon_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == AVERAGE_DIRNAME:
            continue
        if RUN_DIR_PATTERN.match(entry.name):
            run_dirs.append(entry)
        else:
            warnings.append(
                f"Model '{model_name}' ({horizon_name}): ignoring "
                f"unrecognized directory '{entry.name}' (expected "
                f"'run_<number>' or '{AVERAGE_DIRNAME}')."
            )

    def sort_key(path: Path) -> int:
        match = RUN_DIR_PATTERN.match(path.name)
        assert match is not None  # guaranteed by the filter above
        return int(match.group(1))

    return sorted(run_dirs, key=sort_key)


def _load_run_metrics(
    run_dir: Path,
    *,
    model_name: str,
    horizon_name: str,
    required_metrics: tuple[str, ...],
    warnings: list[str],
) -> dict[str, float] | None:
    """Load and validate a single run's evaluation_metrics.json.

    Args:
        run_dir: Path to the run directory (e.g. `.../run_1`).
        model_name: Name of the model this run belongs to.
        horizon_name: Name of the horizon this run belongs to.
        required_metrics: Metric names that must be present.
        warnings: List to which non-fatal warning messages are
            appended.

    Returns:
        Mapping of required metric name -> value, or None if the
        expected results directory/file was not found (a warning is
        recorded in that case rather than raising).

    Raises:
        MetricLoaderError: If the JSON is invalid, or a required
            metric is missing/non-numeric.
    """
    results_dir = run_dir / RESULTS_DIRNAME
    metrics_path = results_dir / EVALUATION_METRICS_FILENAME

    if not metrics_path.is_file():
        warnings.append(
            f"Model '{model_name}' ({horizon_name}): expected metrics "
            f"file not found at '{metrics_path}'; this run will be "
            "skipped."
        )
        return None

    data = _read_json(metrics_path)
    metrics = _validate_and_extract_metrics(
        data, required_metrics=required_metrics, source=metrics_path, warnings=warnings
    )

    if results_dir.is_dir():
        _warn_unexpected_json_files(
            results_dir,
            expected_names={EVALUATION_METRICS_FILENAME},
            context=f"model '{model_name}' ({horizon_name}) run results directory",
            warnings=warnings,
        )

    return metrics


def _warn_unexpected_json_files(
    directory: Path,
    expected_names: set[str],
    *,
    context: str,
    warnings: list[str],
) -> None:
    """Warn only about unexpected *.json files inside a directory.

    Non-JSON clutter (plots, CSVs, PDFs, notebooks, etc.) is common and
    expected to accumulate over the life of a research project, so it
    is deliberately ignored. Stray JSON files are flagged because they
    could indicate a misnamed or forgotten metrics file.

    Args:
        directory: Directory to inspect.
        expected_names: JSON filenames that are expected and should
            not trigger a warning.
        context: Human-readable context used in the warning message.
        warnings: List to which warning messages are appended.
    """
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix == ".json" and entry.name not in expected_names:
            warnings.append(
                f"Unexpected JSON file '{entry.name}' found in {context} "
                f"('{directory}') and was ignored."
            )


def _load_average_metrics(
    average_dir: Path,
    *,
    model_name: str,
    horizon_name: str,
    required_metrics: tuple[str, ...],
    warnings: list[str],
) -> dict[str, float]:
    """Load and validate a model's average_metrics.json, if present.

    Args:
        average_dir: Path to the model/horizon's `average` directory.
        model_name: Name of the model, used in warning messages.
        horizon_name: Name of the horizon, used in warning messages.
        required_metrics: Metric names that must be present.
        warnings: List to which non-fatal warning messages are
            appended.

    Returns:
        Mapping of required metric name -> value, or an empty dict if
        the file was missing.

    Raises:
        MetricLoaderError: If the file exists but contains invalid
            JSON, or a required metric is missing/non-numeric.
    """
    average_path = average_dir / AVERAGE_METRICS_FILENAME

    if not average_dir.is_dir() or not average_path.is_file():
        warnings.append(
            f"Model '{model_name}' ({horizon_name}): "
            f"'{AVERAGE_METRICS_FILENAME}' not found (expected at "
            f"'{average_path}')."
        )
        return {}

    data = _read_json(average_path)
    return _validate_and_extract_metrics(
        data, required_metrics=required_metrics, source=average_path, warnings=warnings
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_models(evaluation_dir: Path) -> list[str]:
    """Discover model names by scanning the evaluation directory.

    Args:
        evaluation_dir: Path to the top-level `evaluation/` directory.

    Returns:
        Sorted list of discovered model (directory) names.

    Raises:
        MetricLoaderError: If `evaluation_dir` does not exist, is not
            a directory, or contains no model directories.
    """
    if not evaluation_dir.is_dir():
        raise MetricLoaderError(
            f"Evaluation directory not found: '{evaluation_dir}'."
        )

    model_names = sorted(
        entry.name
        for entry in evaluation_dir.iterdir()
        if entry.is_dir()
        and not entry.name.startswith(".")
        and not entry.name.startswith("__")
        and _discover_horizon_dirs(entry)
    )

    if not model_names:
        raise MetricLoaderError(
            f"No models discovered under evaluation directory '{evaluation_dir}'."
        )

    return model_names


def load_model_horizon_metrics(
    model_name: str,
    horizon_dir: Path,
    *,
    required_metrics: tuple[str, ...] = DEFAULT_REQUIRED_METRICS,
    warnings: list[str],
) -> ModelMetrics:
    """Load all metrics (per-run and average) for one model/horizon pair.

    Args:
        model_name: Name of the model (directory name).
        horizon_dir: Path to the model's horizon directory (e.g.
            `evaluation/cnn/horizon_15`).
        required_metrics: Metric names that must be present in every
            metrics file. Defaults to the project's standard set
            (RMSE, MAE, R2, NRMSE, MAPE) but can be overridden for
            future projects with different metric sets.
        warnings: List to which non-fatal warning messages are
            appended.

    Returns:
        A populated `ModelMetrics` instance for this model/horizon.

    Raises:
        MetricLoaderError: If no valid runs are found, or any run/
            average file is invalid.
    """
    horizon_name = horizon_dir.name

    run_dirs = _discover_run_dirs(
        horizon_dir, model_name=model_name, horizon_name=horizon_name, warnings=warnings
    )

    per_run_metrics: list[dict[str, float]] = []
    for run_dir in run_dirs:
        run_metrics = _load_run_metrics(
            run_dir,
            model_name=model_name,
            horizon_name=horizon_name,
            required_metrics=required_metrics,
            warnings=warnings,
        )
        if run_metrics is not None:
            per_run_metrics.append(run_metrics)

    if not per_run_metrics:
        raise MetricLoaderError(
            f"Model '{model_name}' ({horizon_name}): no valid runs with "
            f"'{EVALUATION_METRICS_FILENAME}' were found under '{horizon_dir}'."
        )

    # Reshape from list[dict[metric, value]] to dict[metric, list[value]].
    metrics: dict[str, list[float]] = {
        metric_name: [run[metric_name] for run in per_run_metrics]
        for metric_name in required_metrics
    }

    average_metrics = _load_average_metrics(
        horizon_dir / AVERAGE_DIRNAME,
        model_name=model_name,
        horizon_name=horizon_name,
        required_metrics=required_metrics,
        warnings=warnings,
    )

    return ModelMetrics(
        model_name=model_name,
        horizon=horizon_name,
        run_count=len(per_run_metrics),
        metrics=metrics,
        average_metrics=average_metrics,
    )


def load_all_metrics(
    evaluation_dir: Path | str,
    *,
    horizons: tuple[str, ...] | None = None,
    required_metrics: tuple[str, ...] = DEFAULT_REQUIRED_METRICS,
) -> dict[str, dict[str, ModelMetrics]]:
    """Discover and load evaluation metrics for every model and horizon.

    This is the main entry point for the loader. It discovers all
    models under `evaluation_dir`, discovers all horizon directories
    per model (or uses an explicit `horizons` filter if given), loads
    per-run and average metrics, validates the data, prints a summary,
    and returns clean Python objects for downstream statistical
    analysis.

    Args:
        evaluation_dir: Path to the top-level `evaluation/` directory.
        horizons: Optional explicit set of horizon directory names to
            load (e.g. `("horizon_15",)`). If None (default), all
            horizon directories found under each model are loaded,
            making the loader agnostic to how many horizons the
            project uses.
        required_metrics: Metric names that must be present in every
            metrics file. Defaults to the project's standard set.

    Returns:
        Nested mapping: model name -> horizon name -> `ModelMetrics`.

    Raises:
        MetricLoaderError: If the evaluation directory is missing, no
            models are discovered, no horizons are found for a model,
            or any required file is missing, invalid, or contains
            non-numeric/missing required metrics.
    """
    evaluation_dir = Path(evaluation_dir)
    warnings: list[str] = []

    model_names = discover_models(evaluation_dir)

    results: dict[str, dict[str, ModelMetrics]] = {}
    for model_name in model_names:
        model_dir = evaluation_dir / model_name
        horizon_dirs = _discover_horizon_dirs(model_dir)

        if horizons is not None:
            horizon_dirs = [d for d in horizon_dirs if d.name in horizons]

        if not horizon_dirs:
            raise MetricLoaderError(
                f"Model '{model_name}': no matching horizon directories "
                f"found under '{model_dir}'."
            )

        results[model_name] = {}
        for horizon_dir in horizon_dirs:
            results[model_name][horizon_dir.name] = load_model_horizon_metrics(
                model_name,
                horizon_dir,
                required_metrics=required_metrics,
                warnings=warnings,
            )

    _check_run_count_consistency(results, warnings)
    _print_summary(results, warnings)

    return results


def _check_run_count_consistency(
    results: dict[str, dict[str, ModelMetrics]],
    warnings: list[str],
) -> None:
    """Warn if models have differing run counts within the same horizon.

    Args:
        results: Nested mapping: model name -> horizon name ->
            `ModelMetrics`.
        warnings: List to which warning messages are appended.
    """
    horizons_seen: dict[str, dict[str, int]] = {}
    for model_name, per_horizon in results.items():
        for horizon_name, model_metrics in per_horizon.items():
            horizons_seen.setdefault(horizon_name, {})[model_name] = (
                model_metrics.run_count
            )

    for horizon_name, run_counts in horizons_seen.items():
        if len(set(run_counts.values())) > 1:
            warnings.append(
                f"Models have differing run counts for '{horizon_name}': "
                f"{run_counts}."
            )


def _print_summary(
    results: dict[str, dict[str, ModelMetrics]],
    warnings: list[str],
) -> None:
    """Print the final loader summary to stdout.

    Args:
        results: Nested mapping: model name -> horizon name ->
            `ModelMetrics`.
        warnings: Collected non-fatal warning messages.
    """
    separator = "=" * 43
    print(separator)
    print("Metric Loader Summary")
    print(separator)

    print("Discovered models:")
    for model_name, per_horizon in results.items():
        print(f"  - {model_name}:")
        for horizon_name, model_metrics in per_horizon.items():
            print(f"      {horizon_name}: {model_metrics.run_count} run(s)")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nWarnings: none")

    print("\nLoader completed successfully.")


if __name__ == "__main__":
    # Simple manual smoke test / CLI usage:
    #     python -m analysis.load_metrics <path-to-evaluation-dir>
    import sys

    default_dir = Path("evaluation")
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    try:
        load_all_metrics(target_dir)
    except MetricLoaderError as error:
        print(f"Fatal error: {error}")
        raise SystemExit(1) from error