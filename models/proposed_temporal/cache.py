"""Checkpoint-specific frozen representations shared by screening arms."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.proposed_rvqc.data import file_hash
from .model import TemporalAugmentationModel


class CachedTemporalModel(TemporalAugmentationModel):
    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1:] != (4, 128):
            raise ValueError("Expected cached pooled/moment features [B,4,128].")
        return self.forward_cached(inputs[:, 0], inputs[:, 1:])

    @torch.no_grad()
    def assert_initial_equivalence(self, inputs):
        mode = self.training
        self.eval()
        try:
            torch.testing.assert_close(self(inputs), self.backbone.mlp_head(inputs[:, 0]),
                                       rtol=1e-5, atol=1e-6)
        finally:
            self.train(mode)


class CachedBaseline(torch.nn.Module):
    """Run the existing validation gate on cached pooling without re-extraction."""
    def __init__(self, baseline):
        super().__init__()
        self.head = baseline.baseline.mlp_head
        self.checkpoint = baseline.checkpoint
        self.sha256 = baseline.sha256

    @torch.no_grad()
    def forward(self, inputs):
        pooled = inputs[:, 0]
        return pooled, self.head(pooled)


class CachedSplit(Dataset):
    def __init__(self, directory, split):
        if split not in ("train", "validation"):
            raise ValueError("Only train and validation caches are allowed.")
        self.features = np.load(Path(directory) / f"{split}_features.npy", mmap_mode="r")
        self.targets = np.load(Path(directory) / f"{split}_targets.npy", mmap_mode="r")
        size = len(self.features)
        if self.features.shape != (size, 4, 128) or self.targets.shape != (size, 3):
            raise ValueError("Invalid representation cache shapes.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.tensor(self.features[index]), torch.tensor(self.targets[index])


def prepare_cache(directory, baseline, datasets, history, provenance, batch_size):
    """Publish a manifest only after complete extraction; reject stale caches."""
    if set(datasets) != {"train", "validation"}:
        raise ValueError("Cache requires exactly train and validation splits.")
    directory = Path(directory)
    identity = {
        "checkpoint_sha256": baseline.sha256, "history": history,
        "provenance": provenance, "torch_version": str(torch.__version__),
        "extraction_device": "cpu", "batch_size": batch_size,
        "window_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "module_hashes": {name: file_hash(Path(__file__).with_name(name))
                          for name in ("cache.py", "model.py", "temporal.py")},
        "layout": "[x, M_level, M_trend, M_curvature] rows of width 128",
    }
    manifest_path = directory / "manifest.json"
    if directory.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError(f"Cache identity mismatch: {directory}")
        for name, metadata in manifest["files"].items():
            if file_hash(directory / name) != metadata["sha256"]:
                raise ValueError(f"Cache integrity failure: {name}")
    else:
        extractor = TemporalAugmentationModel(baseline.baseline, "A", history=history).cpu().eval()
        directory.mkdir(parents=True)
        files = {}
        with torch.no_grad():
            for split, dataset in datasets.items():
                size = len(dataset)
                features_path = directory / f"{split}_features.npy"
                targets_path = directory / f"{split}_targets.npy"
                features = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float32,
                                                     shape=(size, 4, 128))
                targets = np.lib.format.open_memmap(targets_path, mode="w+", dtype=np.float32,
                                                    shape=(size, 3))
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    generator=torch.Generator().manual_seed(0))
                offset = 0
                for inputs, target in loader:
                    pooled, moments = extractor.representations(inputs)
                    packed = torch.cat((pooled[:, None], moments), dim=1)
                    if not torch.isfinite(packed).all() or not torch.isfinite(target).all():
                        raise ValueError("Nonfinite cached representations or targets.")
                    end = offset + len(inputs)
                    features[offset:end] = packed.numpy()
                    targets[offset:end] = target.numpy()
                    offset = end
                if offset != size:
                    raise ValueError("Incomplete representation extraction.")
                features.flush()
                targets.flush()
                del features, targets
                for path in (features_path, targets_path):
                    files[path.name] = {"sha256": file_hash(path)}
        manifest = {"identity": identity, "files": files}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return {split: CachedSplit(directory, split) for split in datasets}, identity
