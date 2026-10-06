"""Float32 memory-mapped x/S_bar/T_bar caches; no test-data interface."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import config
from models.proposed_rvqc.data import file_hash
from .extractor import FrozenBranches


class CachedSplit(Dataset):
    def __init__(self, directory, split, size):
        if split not in ("train", "validation"):
            raise ValueError("Only train/validation caches are supported.")
        self.features = np.load(Path(directory) / f"{split}_features.npy", mmap_mode="r")
        self.targets = np.load(Path(directory) / f"{split}_targets.npy", mmap_mode="r")
        if (self.features.shape != (size, 3, 128) or self.targets.shape != (size, 3)
                or self.features.dtype != np.float32 or self.targets.dtype != np.float32):
            self.close()
            raise ValueError("Invalid cached representation shapes or dtype.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.tensor(self.features[index]), torch.tensor(self.targets[index])

    def __getitems__(self, indices):
        features = torch.from_numpy(self.features[indices])
        targets = torch.from_numpy(self.targets[indices])
        return list(zip(features, targets))

    def close(self):
        self.features._mmap.close()
        self.targets._mmap.close()


def cache_identity(baseline, datasets, provenance, batch_size):
    sources = [Path(__file__), Path(__file__).with_name("extractor.py")]
    sources += [config.MODELS_DIR / name for name in (
        "proposed_model.py", "dcnn.py", "feature_attention.py", "residual_bilstm.py",
        "temporal_attention.py", "scalar_gated_fusion.py", "mlp_head.py")]
    return {
        "version": 1, "checkpoint_sha256": baseline.sha256,
        "checkpoint_path": str(baseline.path.resolve()),
        "history_length": 24, "layout": ["x", "S_bar", "T_bar"],
        "feature_shape": [3, 128], "target_shape": [3], "dtype": "float32",
        "window_counts": {split: len(data) for split, data in datasets.items()},
        "provenance": provenance, "torch_version": str(torch.__version__),
        "extraction_device": "cpu", "batch_size": batch_size,
        "source_hashes": {str(path.relative_to(config.PROJECT_ROOT)): file_hash(path) for path in sources},
    }


def prepare_cache(directory, baseline, datasets, provenance, batch_size=256):
    if set(datasets) != {"train", "validation"} or any(len(data) == 0 for data in datasets.values()):
        raise ValueError("Cache requires nonempty train and validation splits only.")
    if file_hash(baseline.path) != baseline.sha256:
        raise ValueError("Checkpoint changed since loading.")
    directory = Path(directory)
    identity = cache_identity(baseline, datasets, provenance, batch_size)
    filenames = {f"{split}_{name}.npy" for split in datasets for name in ("features", "targets")}
    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["identity"] != identity or set(manifest["files"]) != filenames:
            raise ValueError("Cache identity or file inventory mismatch.")
        for filename, digest in manifest["files"].items():
            if file_hash(directory / filename) != digest:
                raise ValueError(f"Cache integrity failure: {filename}")
    else:
        extractor = FrozenBranches(baseline.baseline).cpu().eval()
        directory.mkdir(parents=True)
        files = {}
        for split, dataset in datasets.items():
            size = len(dataset)
            features_path = directory / f"{split}_features.npy"
            targets_path = directory / f"{split}_targets.npy"
            features = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float32,
                                                 shape=(size, 3, 128))
            targets = np.lib.format.open_memmap(targets_path, mode="w+", dtype=np.float32,
                                                shape=(size, 3))
            try:
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    generator=torch.Generator().manual_seed(0))
                offset = 0
                for inputs, target in tqdm(loader, desc=f"Cache {split} frozen branches", leave=False):
                    packed = extractor(inputs)
                    if target.shape != (len(inputs), 3) or not torch.isfinite(packed).all() or not torch.isfinite(target).all():
                        raise ValueError("Invalid cached representations or targets.")
                    end = offset + len(inputs)
                    features[offset:end] = packed.cpu().numpy()
                    targets[offset:end] = target.numpy()
                    offset = end
                if offset != size:
                    raise ValueError("Incomplete extraction.")
                features.flush()
                targets.flush()
            finally:
                features._mmap.close()
                targets._mmap.close()
            files[features_path.name] = file_hash(features_path)
            files[targets_path.name] = file_hash(targets_path)
        if file_hash(baseline.path) != baseline.sha256:
            raise ValueError("Checkpoint changed during extraction.")
        (directory / "manifest.json").write_text(json.dumps({"identity": identity, "files": files}, indent=2))
    cached = {}
    try:
        for split, data in datasets.items():
            cached[split] = CachedSplit(directory, split, len(data))
    except Exception:
        for data in cached.values():
            data.close()
        raise
    return cached, identity
