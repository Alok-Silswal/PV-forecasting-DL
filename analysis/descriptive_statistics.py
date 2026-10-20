"""Run-level arithmetic means and sample standard deviations."""

import numpy as np


def compute_descriptive_statistics(metrics: dict) -> dict:
    summaries = {}
    for model, values in metrics.items():
        row = {}
        for metric, runs in values.items():
            row[f"{metric.upper()} mean"] = float(np.mean(runs))
            row[f"{metric.upper()} SD"] = float(np.std(runs, ddof=1))
        summaries[model] = row
    return summaries
