import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from configs import config
from models.model_factory import get_model
from models.proposed_branch_qfa.cache import prepare_cache
from models.proposed_branch_qfa.extractor import FrozenBranches
from models.proposed_branch_qfa.model import BranchFeatures
from models.proposed_branch_qfa.run_experiment import (
    comparison_summary, loaders, parse_args, print_run_summary, train_arm,
)
from models.proposed_rvqc.frozen_baseline import FrozenBaseline


class BranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.loaded = FrozenBaseline(1)
        cls.baseline = cls.loaded.baseline

    def test_extraction_points_and_shapes(self):
        inputs = torch.randn(4, 24, 7)
        extractor = FrozenBranches(self.baseline)
        spatial, temporal, fused, prediction = extractor.sequences(inputs)
        for value in (spatial, temporal, fused):
            self.assertEqual(value.shape, (4, 24, 128))
        with torch.no_grad():
            attention = self.baseline.feature_attention(self.baseline.dcnn(inputs))
            self.assertEqual(attention.shape, (4, 24, 64))
            expected_s = self.baseline.scalar_gated_fusion.spatial_projection(attention)
            expected_t = self.baseline.temporal_attention(self.baseline.residual_bilstm(inputs))
            torch.testing.assert_close(spatial, expected_s)
            torch.testing.assert_close(temporal, expected_t)
            torch.testing.assert_close(prediction, self.baseline(inputs))
            torch.testing.assert_close(fused, self.baseline.scalar_gated_fusion(attention, temporal))
            packed = extractor(inputs)
            self.assertEqual(packed[:, 0].shape, (4, 128))
            torch.testing.assert_close(packed[:, 0], fused.mean(1))
        self.assertFalse(packed.requires_grad)
        extractor.train()
        self.assertFalse(extractor.training)
        self.assertFalse(extractor.baseline.training)
        self.assertEqual(len(extractor.baseline.scalar_gated_fusion._forward_pre_hooks), 0)

    def test_shared_projection_and_order(self):
        features = BranchFeatures(42).eval()
        spatial, temporal = torch.randn(4, 128), torch.randn(4, 128)
        weight = features.projection.weight
        expected = torch.cat((spatial @ weight.T, temporal @ weight.T), dim=1)
        torch.testing.assert_close(features.raw(spatial, temporal), expected)
        same = features.raw(spatial, spatial)
        torch.testing.assert_close(same[:, :3], same[:, 3:])
        self.assertEqual(weight.numel(), 384)
        self.assertEqual(len(list(features.parameters())), 1)
        self.assertIsNone(features.projection.bias)
        self.assertFalse(features.normalization.affine)
        z = features.normalization(expected)
        torch.testing.assert_close(features(spatial, temporal), (torch.pi / 2) * (1 + torch.tanh(z / 2)))

    def test_equivalence_freezing_and_gradients(self):
        inputs = torch.randn(4, 24, 7)
        packed = FrozenBranches(self.baseline)(inputs)
        for arm in "ABC":
            model = get_model(f"proposed_branch_qfa_{arm.lower()}", baseline=self.baseline)
            model.assert_initial_equivalence(inputs)
            model.cached = True
            model.assert_initial_equivalence(packed)
            if arm != "A":
                self.assertEqual(model.head.hidden_layer.weight[:, 128:].count_nonzero().item(), 0)
                torch.testing.assert_close(model.head.hidden_layer.weight[:, :128], self.baseline.mlp_head.hidden_layer.weight)
            for key, value in self.baseline.mlp_head.output_layer.state_dict().items():
                torch.testing.assert_close(model.head.output_layer.state_dict()[key], value)
            before = {key: value.clone() for key, value in model.frozen.state_dict().items()}
            model.train()
            self.assertTrue(model.head.training)
            self.assertTrue(all(not module.training for module in model.frozen.modules()))
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
            with patch.object(model.frozen, "forward", side_effect=AssertionError("Backbone forwarded")):
                for _ in range(2):
                    optimizer.zero_grad()
                    output = model(packed)
                    self.assertEqual(output.shape, (4, 3))
                    output.square().mean().backward()
                    optimizer.step()
            for key, value in before.items():
                torch.testing.assert_close(model.frozen.state_dict()[key], value, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in model.frozen.parameters()))
            if arm != "A":
                self.assertGreater(model.features.projection.weight.grad.abs().sum().item(), 0)
            if arm == "C":
                self.assertLessEqual(model.transform(torch.randn(4, 6)).abs().max().item(), 1)

    def test_cache_equality_order_and_reuse(self):
        datasets = {split: TensorDataset(torch.randn(8, 24, 7), torch.randn(8, 3))
                    for split in ("train", "validation")}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "cache"
            calls = []
            original = FrozenBranches.forward

            def counted(extractor, inputs):
                calls.append(len(inputs))
                return original(extractor, inputs)

            rng = torch.random.get_rng_state().clone()
            with patch.object(FrozenBranches, "forward", counted):
                cached, identity = prepare_cache(directory, self.loaded, datasets, {"scope": "synthetic"}, 4)
                reused, other_identity = prepare_cache(directory, self.loaded, datasets, {"scope": "synthetic"}, 4)
            self.assertEqual(calls, [4, 4, 4, 4])
            self.assertEqual(identity, other_identity)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            try:
                extractor = FrozenBranches(self.baseline)
                for split, source in datasets.items():
                    online = extractor(source.tensors[0])
                    for index in range(len(source)):
                        features, targets = cached[split][index]
                        for row in range(3):
                            torch.testing.assert_close(features[row], online[index, row])
                        torch.testing.assert_close(targets, source.tensors[1][index], rtol=0, atol=0)
                        torch.testing.assert_close(features, reused[split][index][0], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "identity"):
                    prepare_cache(directory, self.loaded, datasets, {"scope": "different"}, 4)
                with patch.object(config, "EXPERIMENTS_DIR", Path(temporary) / "experiments"), \
                        patch.object(config, "NUM_EPOCHS", 1), \
                        patch.object(FrozenBranches, "forward", side_effect=AssertionError("Runner forwarded backbone")):
                    for arm in "ABC":
                        metrics = train_arm(self.loaded, cached, arm, 42, identity, {}, "cpu")
                        self.assertTrue({"mse", "rmse", "mae", "r2", "nrmse"}.issubset(metrics))
            finally:
                for split in (*cached.values(), *reused.values()):
                    split.close()
            with (directory / "train_features.npy").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"):
                prepare_cache(directory, self.loaded, datasets, {"scope": "synthetic"}, 4)
        with self.assertRaisesRegex(ValueError, "train and validation"):
            prepare_cache("unused", self.loaded, {**datasets, "test": datasets["validation"]}, {})

    def test_private_initialization_and_batch_order(self):
        rng = torch.random.get_rng_state().clone()
        left = get_model("proposed_branch_qfa_c", baseline=self.baseline, seed=43)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.rand(31)
        right = get_model("proposed_branch_qfa_c", baseline=self.baseline, seed=43)
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)
        control = get_model("proposed_branch_qfa_b", baseline=self.baseline, seed=43)
        torch.testing.assert_close(left.features.projection.weight, control.features.projection.weight)
        datasets = {split: TensorDataset(torch.arange(12), torch.arange(12)) for split in ("train", "validation")}
        order = []
        for arm in "ABC":
            get_model(f"proposed_branch_qfa_{arm.lower()}", baseline=self.baseline)
            train, _ = loaders(datasets, 42)
            order.append([torch.cat([x for x, _ in train]) for _ in range(2)])
        for run in order[1:]:
            for first, next_order in zip(order[0], run):
                torch.testing.assert_close(first, next_order, rtol=0, atol=0)

    def test_initial_checkpoint_candidate(self):
        packed = FrozenBranches(self.baseline)(torch.randn(8, 24, 7))
        datasets = {split: TensorDataset(packed, torch.randn(8, 3)) for split in ("train", "validation")}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(config, "EXPERIMENTS_DIR", Path(temporary)), \
                patch.object(config, "NUM_EPOCHS", 1), patch.object(config, "LEARNING_RATE", 0.0):
            metrics = train_arm(self.loaded, datasets, "A", 42, {}, {}, "cpu")
            self.assertEqual(metrics["epoch"], -1)
            path = Path(temporary) / "proposed_branch_qfa_a/horizon_15/run_1/checkpoints/best_checkpoint.pt"
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], -1)

    def test_defaults_and_summary(self):
        arguments = ["--processed-csv", "unused.csv"]
        args = parse_args(arguments)
        self.assertEqual(list(args.seeds), [42])
        self.assertEqual(list(args.arms), ["A", "B", "C"])
        self.assertEqual(args.device, "cpu")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(arguments + ["--seeds", "42", "43"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(arguments + ["--arms", "E"])
        datasets = {"train": range(8), "validation": range(4)}
        for options, count in (([], 3), (["--build-cache-only"], 0),
                               (["--seeds", "42", "43", "44", "--allow-multiple-seeds"], 9)):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_run_summary(parse_args(arguments + options), datasets, self.loaded, Path("nonexistent-cache"))
            self.assertIn(f"Total training runs: {count}", output.getvalue())
            self.assertIn("Train samples: 8", output.getvalue())
            self.assertIn("Validation samples: 4", output.getvalue())

    def test_comparison_is_descriptive(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary = comparison_summary({"A": {"rmse": 1, "mae": 1},
                                          "B": {"rmse": 0.995, "mae": 1.03},
                                          "C": {"rmse": 0.997, "mae": 0.99}})
        self.assertAlmostEqual(summary["B"]["rmse_improvement_percent"], 0.5)
        self.assertIn("more than 2%", output.getvalue())
        self.assertNotIn("passed", summary["B"])

    @unittest.skipUnless(importlib.util.find_spec("pennylane"), "PennyLane is optional for Stage 0")
    def test_quantum_topology_output_and_gradients(self):
        import pennylane as qml
        from models.proposed_branch_qfa.quantum import QuantumFeatures
        theta = torch.rand(2, 6, requires_grad=True)
        trainable = QuantumFeatures(10042)
        frozen = QuantumFeatures(10042, trainable=False)
        self.assertEqual(trainable.angles.shape, (2, 6, 2))
        self.assertEqual(trainable.angles.numel(), 24)
        tape = qml.workflow.construct_tape(trainable.circuit)(theta, trainable.angles)
        operations = tape.operations
        self.assertEqual(len(trainable.circuit.device.wires), 6)
        names = [op.name for op in operations]
        self.assertEqual(names[:6], ["RY"] * 6)
        self.assertEqual(names[6:18], ["RZ", "RY"] * 6)
        self.assertEqual(names[18:21], ["CZ"] * 3)
        self.assertEqual(names[21:], ["RZ", "RY"] * 6)
        self.assertEqual([list(op.wires) for op in operations[18:21]], [[0, 3], [1, 4], [2, 5]])
        self.assertEqual([list(op.wires) for op in operations[:6]], [[i] for i in range(6)])
        self.assertEqual([measurement.obs.name for measurement in tape.measurements], ["PauliZ"] * 6)
        self.assertEqual([list(measurement.obs.wires) for measurement in tape.measurements], [[i] for i in range(6)])
        q = trainable(theta)
        self.assertEqual(q.shape, (2, 6))
        self.assertEqual(q.dtype, theta.dtype)
        self.assertLessEqual(q.abs().max().item(), 1.000001)
        torch.testing.assert_close(q, frozen(theta))
        q.square().sum().backward()
        self.assertGreater(theta.grad.abs().sum().item(), 0)
        self.assertGreater(trainable.angles.grad.abs().sum().item(), 0)
        self.assertFalse(frozen.angles.requires_grad)
        frozen_input = theta.detach().clone().requires_grad_(True)
        frozen(frozen_input).sum().backward()
        self.assertGreater(frozen_input.grad.abs().sum().item(), 0)
        self.assertIsNone(frozen.angles.grad)
        packed = FrozenBranches(self.baseline)(torch.randn(2, 24, 7))
        for arm in "DE":
            model = get_model(f"proposed_branch_qfa_{arm.lower()}", baseline=self.baseline, cached=True)
            model.assert_initial_equivalence(packed)
            self.assertEqual(model.transform.angles.requires_grad, arm == "E")
            self.assertTrue(model.features.projection.weight.requires_grad)
            self.assertTrue(all(p.requires_grad for p in model.head.parameters()))


if __name__ == "__main__":
    unittest.main()
