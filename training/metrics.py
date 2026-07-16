"""
Reported Metrics
----------------
- RMSE
- MAE
- MAPE
- R²
- nRMSE (Range Normalized)
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """
    Parameters
    ----------
    predictions : Model predictions.

    targets : Ground truth values.

    """

    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    rmse = np.sqrt(
        mean_squared_error(
            targets,
            predictions,
        )
    )

    mae = mean_absolute_error(
        targets,
        predictions,
    )

    epsilon = 1e-8

    mape = np.mean(
        np.abs(
            (targets - predictions)
            / (targets + epsilon)
        )
    ) * 100.0

    r2 = r2_score(
        targets,
        predictions,
    )

    value_range = np.max(targets) - np.min(targets)

    if value_range == 0:
        nrmse = 0.0
    else:
        nrmse = rmse / value_range

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "nrmse": float(nrmse),
    }