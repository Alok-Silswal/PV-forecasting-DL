"""One metric-only summary and paired-comparison table."""

import pandas as pd

from analysis.significance_tests import MODELS


def generate_metric_table(descriptive: dict, comparisons: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"model": model, **descriptive[model], **comparisons.get(model, {})}
        for model in MODELS
    ])
