"""Reconstruct only train/validation using the finalized notebook convention."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FEATURES = (
    "Weather_Temperature_Celsius", "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation", "Diffuse_Horizontal_Radiation",
    "Radiation_Global_Tilted", "Radiation_Diffuse_Tilted", "Active_Power",
)
TRAIN_END = 745702
VAL_END = 905495
TOTAL_ROWS = 1065289
LOOKBACK = 24
OUTPUTS = 3
WINDOW_COUNTS = {"train": 745676, "validation": 159767}
ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Windows(Dataset):
    def __init__(self, features, target):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32).reshape(-1)
        if self.features.shape != (len(self.target), 7):
            raise ValueError("Expected seven features and one target per source row.")

    def __len__(self):
        return len(self.target) - LOOKBACK - OUTPUTS + 1

    def __getitem__(self, s):
        if s < 0 or s >= len(self):
            raise IndexError(s)
        return self.features[s:s + 24], self.target[s + 24:s + 27]


def reconstruct(processed_csv):
    """Never request a source row at or beyond the validation boundary."""
    prefix = bytearray()
    train_digest = hashlib.sha256()
    with Path(processed_csv).open("rb", buffering=0) as stream:
        remaining_lines = VAL_END + 1
        while remaining_lines:
            # A line has at least one byte: this cannot read beyond the boundary.
            chunk = stream.read(min(1024 * 1024, remaining_lines))
            if not chunk:
                raise ValueError(f"Processed CSV ends before row {VAL_END}.")
            prefix.extend(chunk)
            remaining_lines -= chunk.count(b"\n")
    fitting_stream = io.BytesIO(prefix)
    for _ in range(TRAIN_END + 1):
        train_digest.update(fitting_stream.readline())
    frame = pd.read_csv(io.BytesIO(prefix), parse_dates=["timestamp"])
    if len(frame) != VAL_END or frame["timestamp"].isna().any():
        raise ValueError("Invalid source prefix; expected one CSV record per data row.")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("Source timestamps must be ordered and unique.")
    features = frame.loc[:, list(FEATURES)]
    target = frame["Active_Power"].to_numpy().reshape(-1, 1)
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite values in finalized processed data.")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_x = feature_scaler.fit_transform(features.iloc[:TRAIN_END])
    train_y = target_scaler.fit_transform(target[:TRAIN_END])
    val_x = feature_scaler.transform(features.iloc[TRAIN_END:VAL_END])
    val_y = target_scaler.transform(target[TRAIN_END:VAL_END])
    datasets = {"train": Windows(train_x, train_y), "validation": Windows(val_x, val_y)}
    for split, dataset in datasets.items():
        if len(dataset) != WINDOW_COUNTS[split]:
            raise ValueError(f"Wrong {split} window count.")
    metadata = {
        "source_path": str(Path(processed_csv).resolve()),
        "source_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "fitting_prefix_sha256": train_digest.hexdigest(),
        "source_rows_read": VAL_END, "test_rows_read": 0,
        "declared_total_rows": TOTAL_ROWS,
        "splits": {"train": [0, TRAIN_END], "validation": [TRAIN_END, VAL_END]},
        "features": list(FEATURES), "target": "Active_Power",
        "lookback": LOOKBACK, "outputs": OUTPUTS, "stride": 1,
        "window_counts": WINDOW_COUNTS, "dtype": "float32",
        "notebook_sha256": file_hash(ROOT / "notebooks/Data_Preprocessing.ipynb"),
        "data_module_sha256": file_hash(__file__),
        "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "torch_version": str(torch.__version__),
        "scalers": {},
    }
    for name, scaler in [("feature", feature_scaler), ("target", target_scaler)]:
        metadata["scalers"][name] = {
            "class": "StandardScaler", "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "variance": scaler.var_.tolist(),
            "fitted_rows": int(scaler.n_samples_seen_),
            "n_features": int(scaler.n_features_in_),
        }
    return datasets, feature_scaler, target_scaler, metadata
