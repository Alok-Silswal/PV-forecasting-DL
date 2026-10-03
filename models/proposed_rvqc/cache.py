"""Immutable, shared train/validation caches; no test-cache interface."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ROOT, WINDOW_COUNTS, file_hash

CACHE_ROOT = ROOT / "artifacts/proposed_rvqc/cache"


def cache_identity(provenance, baseline, run, gate, scope):
    if not gate["passed"] or gate["checkpoint_sha256"] != baseline.sha256:
        raise ValueError("A matching successful validation gate is required.")
    return {
        "provenance": provenance, "checkpoint_sha256": baseline.sha256,
        "run": run, "seed": 41 + run, "scope": scope, "validation_gate": gate,
        "frozen_extractor_sha256": file_hash(ROOT / "models/proposed_rvqc/frozen_baseline.py"),
    }


def write_cache(directory, datasets, baseline, identity, feature_scaler, target_scaler):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {directory}")
    if set(datasets) != {"train", "validation"}:
        raise ValueError("Cache accepts exactly train and validation.")
    directory.mkdir(parents=True)
    files = {}
    for split, dataset in datasets.items():
        size = len(dataset)
        if identity["scope"] == "full" and size != WINDOW_COUNTS[split]:
            raise ValueError(f"Unexpected full-cache size for {split}.")
        arrays = {}
        for name, width in [("pooled", 128), ("baseline", 3), ("target", 3)]:
            filename = f"{split}_{name}.npy"
            arrays[name] = np.lib.format.open_memmap(directory / filename, mode="w+", dtype=np.float32, shape=(size, width))
        offset = 0
        loader = DataLoader(dataset, batch_size=256, shuffle=False,
                            generator=torch.Generator().manual_seed(0))
        for inputs, target in loader:
            pooled, prediction = baseline(inputs)
            end = offset + len(inputs)
            for name, tensor in [("pooled", pooled), ("baseline", prediction), ("target", target)]:
                if not torch.isfinite(tensor).all():
                    raise ValueError("Nonfinite cache tensor.")
                arrays[name][offset:end] = tensor.numpy()
            offset = end
        if offset != size:
            raise ValueError("Incomplete cache extraction.")
        for name, array in arrays.items():
            array.flush()
            filename = f"{split}_{name}.npy"
            files[filename] = {"shape": list(array.shape), "sha256": file_hash(directory / filename)}
        del arrays
    for name, scaler in [("feature_scaler.pkl", feature_scaler), ("target_scaler.pkl", target_scaler)]:
        (directory / name).write_bytes(pickle.dumps(scaler, protocol=4))
        files[name] = {"sha256": file_hash(directory / name)}
    if file_hash(baseline.path) != baseline.sha256:
        raise ValueError("Baseline checkpoint changed during caching.")
    manifest = {"identity": identity, "files": files}
    # A manifest is published only after every array has been completed and hashed.
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_cache(directory, identity):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"] != identity:
        raise ValueError("Cache provenance does not match this reconstruction/checkpoint.")
    for name, metadata in manifest["files"].items():
        if file_hash(directory / name) != metadata["sha256"]:
            raise ValueError(f"Cache integrity failure: {name}")
    return manifest


class CachedSplit:
    def __init__(self, directory, split, scope="full"):
        if split not in ("train", "validation"):
            raise ValueError("Test caches are forbidden.")
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["identity"]["scope"] != scope:
            raise ValueError("Smoke caches cannot be used for real training.")
        self.arrays = [np.load(directory / f"{split}_{name}.npy", mmap_mode="r" if scope == "full" else None)
                       for name in ("pooled", "baseline", "target")]
        size = len(self.arrays[0])
        if [array.shape for array in self.arrays] != [(size, 128), (size, 3), (size, 3)]:
            raise ValueError("Unexpected cache shapes.")
        if scope == "full" and size != WINDOW_COUNTS[split]:
            raise ValueError("Unexpected cache window count.")

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        return tuple(torch.tensor(array[indices]) for array in self.arrays)
