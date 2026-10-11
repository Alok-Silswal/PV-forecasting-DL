"""Reuse RVQC's immutable cache format with strict comparison metadata."""

import json
from pathlib import Path
import numpy as np
import torch
from models.proposed_rvqc.cache import CachedSplit, cache_identity, verify_cache, write_cache
from models.proposed_rvqc.data import ROOT, WINDOW_COUNTS, file_hash, reconstruct
from models.proposed_rvqc.frozen_baseline import FrozenBaseline, validation_gate

CACHE_ROOT = ROOT / "artifacts/controlled_residual/cache/horizon_15"
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}


def verify(directory, identity):
    manifest = verify_cache(directory, identity)
    expected = {"feature_scaler.pkl", "target_scaler.pkl"}
    for split, size in WINDOW_COUNTS.items():
        for name, width in (("pooled", 128), ("baseline", 3), ("target", 3)):
            filename = f"{split}_{name}.npy"
            expected.add(filename)
            array = np.load(Path(directory) / filename, mmap_mode="r")
            if array.shape != (size, width) or array.dtype != np.float32 or manifest["files"][filename]["shape"] != list(array.shape):
                raise ValueError(f"Incompatible cache tensor: {filename}")
    if set(manifest["files"]) != expected:
        raise ValueError("Cache file set differs from the fixed protocol.")
    return manifest


def check_live(directory, datasets, baseline, device, scope="full"):
    for split, dataset in datasets.items():
        indices = torch.tensor(sorted(set((0, len(dataset) // 2, len(dataset) - 1))))
        inputs = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
        targets = torch.stack([dataset[int(i)][1] for i in indices])
        pooled, base = baseline(inputs)
        cached_x, cached_base, cached_y = CachedSplit(directory, split, scope)[indices]
        torch.testing.assert_close(cached_x, pooled.cpu(), **TOLERANCE)
        torch.testing.assert_close(cached_base, base.cpu(), **TOLERANCE)
        if not torch.equal(cached_y, targets):
            raise ValueError("Cache target/sample order differs from live data.")
    return {"sample_indices": "first/middle/last per split", "tolerance": TOLERANCE, "targets_exact": True}


def prepare(processed_csv, run: int, device: torch.device, build: bool):
    datasets, feature_scaler, target_scaler, provenance = reconstruct(processed_csv)
    baseline = FrozenBaseline(run, device=device)
    directory = CACHE_ROOT / f"run_{run}"
    if directory.exists():
        if not (directory / "manifest.json").is_file():
            raise ValueError(f"Incomplete cache (missing manifest), refusing automatic repair: {directory}")
        manifest = json.loads((directory / "manifest.json").read_text())
        old = manifest["identity"]
        gate = old["validation_gate"]
        history = ROOT / f"experiments/proposed/horizon_15/run_{run}/history.json"
        if (old["provenance"] != provenance or not gate["passed"] or
                gate["checkpoint_sha256"] != baseline.sha256 or gate["history_sha256"] != file_hash(history)):
            raise ValueError("Stale cache validation gate/checkpoint/dataset; refusing reuse.")
    else:
        gate = validation_gate(baseline, datasets["validation"], run, device=device)
    if not gate["passed"]:
        raise ValueError("Reconstructed validation does not reproduce the source checkpoint.")
    identity = cache_identity(provenance, baseline, run, gate, "full")
    identity["experiment"] = "controlled_residual_v1"
    identity["dtype"] = "float32"
    status = "reused" if directory.exists() else "not built"
    if build and not directory.exists():
        def extract(inputs):
            pooled, base = baseline(inputs.to(device))
            return pooled.cpu(), base.cpu()
        extract.path, extract.sha256 = baseline.path, baseline.sha256
        write_cache(directory, datasets, extract, identity, feature_scaler, target_scaler)
        status = "generated"
    checks = None
    if directory.exists():
        verify(directory, identity)
        checks = check_live(directory, datasets, baseline, device)
    return directory, identity, baseline, status, checks
