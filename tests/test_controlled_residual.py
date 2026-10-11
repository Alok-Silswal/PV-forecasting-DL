"""Engineering checks for the fixed frozen-backbone comparison."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from models.proposed_rvqc.batch_order import batches
from models.proposed_rvqc.cache import CachedSplit, cache_identity, verify_cache, write_cache
from models.proposed_rvqc.data import Windows
from models.proposed_rvqc.frozen_baseline import FrozenBaseline
from experiments.controlled_residual.cache import check_live, verify
from experiments.controlled_residual.evaluate import evaluate_arrays
from experiments.controlled_residual.model import ARMS, Correction, FrozenExtension
from experiments.controlled_residual.preflight import reference_agreement, smoke
from experiments.controlled_residual.run_experiment import parse_args, train_arm
from experiments.controlled_residual.simulator import RVQCSimulator, pennylane_reference


class TensorCache:
    def __init__(self, pooled, base, target):
        self.values = pooled, base, target

    def __len__(self):
        return len(self.values[0])

    def __getitem__(self, indices):
        return tuple(v[indices] for v in self.values)


class ControlledResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initialization_counts_and_order(self):
        before = torch.get_rng_state().clone()
        classical, quantum = [Correction(arm) for arm in ARMS]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for name in ("weight", "bias"):
            self.assertTrue(torch.equal(getattr(classical.projection, name), getattr(quantum.projection, name)))
        x, base = torch.randn(9, 128), torch.randn(9, 3)
        for model, count in ((classical, 837), (quantum, 819)):
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual(model(x).shape, (9, 3))
            self.assertTrue(torch.equal(base + model(x), base))
        left = batches(513, 42, 2)
        torch.rand(23)
        right = batches(513, 42, 2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))
        with patch("sys.argv", ["run", "--processed-csv", "unused.csv"]):
            args = parse_args()
        self.assertEqual(args.run, 1)
        self.assertEqual(args.backend, "torch")

    def test_upstream_gradients_after_first_update(self):
        for arm in ARMS:
            model = Correction(arm)
            x, base, y = torch.randn(8, 128), torch.randn(8, 3), torch.randn(8, 3)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            torch.nn.functional.mse_loss(base + model(x), y).backward()
            for name, p in model.named_parameters():
                if not name.startswith("readout."):
                    self.assertEqual(p.grad.abs().sum().item(), 0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(base + model(x), y).backward()
            for p in model.parameters():
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_frozen_backbone_and_cache(self):
        generator = torch.Generator().manual_seed(142)
        rows = torch.randn(40, 7, generator=generator).numpy()
        targets = rows[:, 0:1]
        datasets = {s: Windows(rows, targets) for s in ("train", "validation")}
        baseline = FrozenBaseline(1)
        before = {k: v.clone() for k, v in baseline.state_dict().items()}
        feature_scaler, target_scaler = StandardScaler().fit(rows), StandardScaler().fit(targets)
        identity = cache_identity({}, baseline, 1, {"passed": True, "checkpoint_sha256": baseline.sha256}, "smoke")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "cache"
            write_cache(directory, datasets, baseline, identity, feature_scaler, target_scaler)
            verify_cache(directory, identity)
            check_live(directory, datasets, baseline, torch.device("cpu"), "smoke")
            with self.assertRaises(ValueError):
                verify_cache(directory, {**identity, "run": 2})
            cached = CachedSplit(directory, "train", "smoke")
            inputs = torch.stack([datasets["train"][i][0] for i in range(len(cached))])
            pooled, base, y = cached[slice(None)]
            live_x, live_base = baseline(inputs)
            torch.testing.assert_close(pooled, live_x, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(base, live_base, atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.equal(y, torch.stack([datasets["train"][i][1] for i in range(len(cached))])))
            for arm in ARMS:
                extension = FrozenExtension(1, arm).train()
                extension.frozen.load_state_dict(baseline.state_dict())
                self.assertFalse(any(m.training for m in extension.frozen.modules()))
                optimizer = torch.optim.Adam(extension.correction.parameters(), lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    torch.nn.functional.mse_loss(extension.forward_cached(pooled, base), y).backward()
                    optimizer.step()
                extension.eval()
                torch.testing.assert_close(extension(inputs), extension.forward_cached(pooled, base), atol=1e-6, rtol=1e-5)
                self.assertTrue(all(torch.equal(v, extension.frozen.state_dict()[k]) for k, v in before.items()))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in extension.frozen.parameters()))
            (directory / "train_target.npy").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                verify_cache(directory, identity)

    def test_epoch_zero_retained_reload_and_no_resume(self):
        x, base = torch.randn(8, 128), torch.randn(8, 3)
        cache = TensorCache(x, base, base.clone())
        for arm in ARMS:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / arm
                settings = train_arm(arm, cache, cache, directory, {"seed": 42, "backend": "torch"}, torch.device("cpu"), epochs=2, patience=1)
                self.assertEqual(settings["selected_epoch"], 0)
                self.assertTrue(settings["no_correction_selected"])
                checkpoint = torch.load(directory / "checkpoints/best_checkpoint.pt", weights_only=True)
                restored = Correction(arm)
                restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                self.assertTrue(torch.equal(base + restored(x), base))
                self.assertEqual(checkpoint["epoch"], 0)
                self.assertTrue((directory / "checkpoints/last_checkpoint.pt").exists())
                self.assertEqual(json.loads((directory / "history.json").read_text())[0]["epoch"], 0)
                with self.assertRaises(FileExistsError):
                    train_arm(arm, cache, cache, directory, {"seed": 42, "backend": "torch"}, torch.device("cpu"), epochs=1)

    def test_trained_best_checkpoint_reloads(self):
        generator = torch.Generator().manual_seed(77)
        x = torch.randn(8, 128, generator=generator)
        base = torch.randn(8, 3, generator=generator)
        cache = TensorCache(x, base, base + 0.1)
        for arm in ARMS:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / arm
                settings = train_arm(arm, cache, cache, directory, {"seed": 42, "backend": "torch"}, torch.device("cpu"), epochs=2)
                self.assertGreater(settings["selected_epoch"], 0)
                checkpoint = torch.load(directory / "checkpoints/best_checkpoint.pt", weights_only=True)
                restored = Correction(arm).eval()
                restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                loss = torch.nn.functional.mse_loss(base + restored(x), cache.values[2]).item()
                self.assertAlmostEqual(loss, settings["best_val_loss"], places=8)

    def test_cache_shape_dtype_and_metadata_validation(self):
        from models.proposed_rvqc import cache as original_cache
        from experiments.controlled_residual import cache as comparison_cache
        rows = np.zeros((30, 7), dtype=np.float32)
        datasets = {s: Windows(rows, rows[:, :1]) for s in ("train", "validation")}
        baseline = FrozenBaseline(1)
        identity = cache_identity({}, baseline, 1, {"passed": True, "checkpoint_sha256": baseline.sha256}, "full")
        counts = {s: len(d) for s, d in datasets.items()}
        with tempfile.TemporaryDirectory() as temporary, patch.object(original_cache, "WINDOW_COUNTS", counts), patch.object(comparison_cache, "WINDOW_COUNTS", counts):
            directory = Path(temporary) / "cache"
            scaler = StandardScaler().fit(rows[:, :1])
            write_cache(directory, datasets, baseline, identity, scaler, scaler)
            verify(directory, identity)
            with self.assertRaises(ValueError):
                verify(directory, {**identity, "provenance": {"lookback": 12}})
            # A self-consistent file hash alone must not admit a wrong dtype.
            np.save(directory / "train_pooled.npy", np.zeros((counts["train"], 128), dtype=np.float64))
            from models.proposed_rvqc.data import file_hash
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["train_pooled.npy"]["sha256"] = file_hash(directory / "train_pooled.npy")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify(directory, identity)

    def test_shared_inverse_scaling_and_metrics(self):
        scaler = StandardScaler().fit(np.array([[10.0], [30.0], [50.0]]))
        target = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
        base = target + 0.2
        correction = np.full_like(base, -0.1)
        forecasts, actual, metrics = evaluate_arrays(base, {arm: correction.copy() for arm in ARMS}, target, scaler)
        np.testing.assert_array_equal(forecasts["classical"], forecasts["quantum"])
        np.testing.assert_array_equal(actual, scaler.inverse_transform(target.astype(np.float64).reshape(-1, 1)).reshape(4, 3))
        self.assertAlmostEqual(metrics["classical"]["correction"]["mean"], -0.1 * scaler.scale_[0], places=5)
        self.assertEqual(len(metrics["baseline"]["horizons"]), 3)
        self.assertLess(metrics["classical"]["relative_to_baseline"]["rmse"]["absolute_difference"], 0)
        forecasts, _, _ = evaluate_arrays(base, {arm: np.zeros_like(base) for arm in ARMS}, target, scaler)
        self.assertTrue(all(np.array_equal(p, forecasts["baseline"]) for p in forecasts.values()))

    @unittest.skipUnless(importlib.util.find_spec("pennylane"), "PennyLane unavailable")
    def test_reference_and_exact_historical_circuit(self):
        reference_agreement()
        from models.proposed_rvqc.residual_branch import ResidualBranch, canonical_state
        old = ResidualBranch("proposed_rvqc", canonical_state(42))
        generator = torch.Generator().manual_seed(2026)
        theta = torch.rand(5, 6, generator=generator) * 2 * torch.pi - torch.pi
        angles = torch.randn(2, 6, 2, generator=generator)
        actual = RVQCSimulator()(theta, angles)
        expected = torch.stack(old.circuit(theta, angles), dim=1).float()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        reference = pennylane_reference()
        reference(theta, angles)
        operations = reference._tape.operations
        self.assertEqual(sum(op.name == "CNOT" for op in operations), 12)
        self.assertEqual(sum(op.name == "RY" for op in operations), 24)
        self.assertEqual(sum(op.name == "RZ" for op in operations), 12)
        self.assertEqual([tuple(op.wires) for op in operations if op.name == "CNOT"], [(i, (i + 1) % 6) for i in range(6)] * 2)

    def test_tiny_engineering_learning(self):
        self.assertFalse(smoke(torch.device("cpu"))["scientific_evidence"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_simulator_and_correction(self):
        reference_agreement(torch.device("cuda"))
        for arm in ARMS:
            model = Correction(arm).cuda()
            x = torch.randn(4, 128, device="cuda")
            with torch.no_grad():
                model.readout.weight.fill_(0.1)
            model(x).square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
