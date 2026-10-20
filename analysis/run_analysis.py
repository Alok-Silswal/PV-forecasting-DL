"""Load authoritative run metrics and assemble the fixed comparison family."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.descriptive_statistics import compute_descriptive_statistics
from analysis.generate_tables import generate_metric_table
from analysis.significance_tests import (
    HORIZON, METHOD_NOTES, METRICS, MODELS, RUNS, run_significance_analysis,
)


@dataclass(frozen=True)
class AnalysisResults:
    table: pd.DataFrame
    notes: tuple[str, ...]


def load_run_metrics(evaluation_dir: Path | str) -> dict:
    metrics, issues = {}, []
    for model in MODELS:
        directory = Path(evaluation_dir) / model / HORIZON
        expected = {f"run_{run}" for run in RUNS}
        found = {path.name for path in directory.glob("run_*") if path.is_dir()}
        if found != expected:
            issues.append(f"{directory}: expected run_1 through run_5; found {sorted(found)}")
        values = {metric: [] for metric in METRICS}
        for run in RUNS:
            path = directory / f"run_{run}" / "results/evaluation_metrics.json"
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                for metric in METRICS:
                    value = record[metric]
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not np.isfinite(value) or value < 0):
                        raise ValueError(f"{metric} must be a finite nonnegative number")
                for metric in METRICS:
                    values[metric].append(record[metric])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append(f"{path}: {exc}")
        metrics[model] = {metric: np.asarray(runs, dtype=float) for metric, runs in values.items()}
    if issues:
        raise ValueError("Missing/incompatible run metrics; no tests calculated:\n- " + "\n- ".join(issues))
    return metrics


def run_analysis(evaluation_dir: Path | str) -> AnalysisResults:
    metrics = load_run_metrics(evaluation_dir)
    descriptive = compute_descriptive_statistics(metrics)
    comparisons = run_significance_analysis(metrics)
    return AnalysisResults(generate_metric_table(descriptive, comparisons), METHOD_NOTES)
