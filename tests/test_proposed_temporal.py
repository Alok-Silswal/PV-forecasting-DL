import unittest
import importlib.util
import logging
import tempfile

import torch

from models.proposed_model import ProposedModel
from models.proposed_temporal.model import TemporalAugmentationModel
from models.proposed_temporal.temporal import TemporalSummary
from training.trainer import Trainer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_basis_and_order(self):
        for history in (12, 24):
            summary = TemporalSummary(history)
            torch.testing.assert_close(summary.basis.T @ summary.basis, torch.eye(3))
            self.assertFalse(summary.basis.requires_grad)
            self.assertGreater(summary.basis[-1, 1].item(), 0)
            self.assertGreater(summary.basis[-1, 2].item(), 0)
            self.assertLess(summary.basis[history // 2, 2].item(), 0)
            fused = torch.randn(4, 24, 128)
            self.assertEqual(summary(fused).shape, (4, 3, 128))
        model = TemporalAugmentationModel(ProposedModel(), "B")
        model.eval()
        moments = torch.randn(4, 3, 128)
        projected = model.features.projection(moments)
        ordered = torch.stack([projected[:, k, s] for s in range(2) for k in range(3)], dim=1)
        expected = (torch.pi / 2) * (1 + torch.tanh(model.features.normalization(ordered) / 2))
        torch.testing.assert_close(model.features(moments), expected)
        self.assertEqual(model.features.projection.weight.numel(), 256)

    def test_equivalence_freezing_and_gradients(self):
        baseline = ProposedModel().eval()
        inputs = torch.randn(4, 24, 7)
        for arm in "ABC":
            model = TemporalAugmentationModel(baseline, arm)
            model.assert_initial_equivalence(inputs)
            model.eval()
            with torch.no_grad():
                pooled, moments = model.representations(inputs)
                torch.testing.assert_close(model(inputs), model.forward_cached(pooled, moments))
            before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
            model.train()
            self.assertFalse(model.backbone.training)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
            for _ in range(2):
                optimizer.zero_grad()
                output = model(inputs)
                self.assertEqual(output.shape, (4, 3))
                output.square().mean().backward()
                optimizer.step()
            for key, value in before.items():
                torch.testing.assert_close(model.backbone.state_dict()[key], value, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
            if arm != "A":
                self.assertGreater(model.features.projection.weight.grad.abs().sum().item(), 0)

    def test_private_initialization(self):
        baseline = ProposedModel()
        state = torch.random.get_rng_state().clone()
        left = TemporalAugmentationModel(baseline, "C", seed=43)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.rand(17)
        right = TemporalAugmentationModel(baseline, "C", seed=43)
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)

    def test_pretrained_checkpoint_equivalence(self):
        from models.proposed_rvqc.frozen_baseline import FrozenBaseline
        baseline = FrozenBaseline(1).baseline
        for arm in "ABC":
            model = TemporalAugmentationModel(baseline, arm)
            model.assert_initial_equivalence(torch.randn(4, 24, 7))
            if arm != "A":
                self.assertEqual(model.head.hidden_layer.weight[:, 128:].count_nonzero().item(), 0)

    def test_trainer_integration(self):
        dataset = TensorDataset(torch.randn(8, 24, 7), torch.randn(8, 3))
        baseline = ProposedModel()
        orders = []
        for arm in "ABC":
            model = TemporalAugmentationModel(baseline, arm)
            sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(20042))
            orders.append(torch.tensor(list(sampler)))
            loader = DataLoader(dataset, batch_size=4,
                                sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(20042)),
                                generator=torch.Generator().manual_seed(30042))
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
            with tempfile.TemporaryDirectory() as directory:
                trainer = Trainer(model, loader, loader, torch.nn.MSELoss(), optimizer,
                                  None, torch.device("cpu"), logging.getLogger("test"),
                                  directory, 2, num_epochs=1)
                history = trainer.train()
                self.assertEqual(len(history["val_loss"]), 1)
                self.assertTrue(trainer.checkpoint_path.exists())
        self.assertTrue(all(torch.equal(orders[0], order) for order in orders))

    @unittest.skipUnless(importlib.util.find_spec("pennylane"), "PennyLane is optional for Step 0")
    def test_quantum_circuit_and_gradients(self):
        from models.proposed_temporal.quantum import QuantumFeatures
        import pennylane as qml
        theta = torch.rand(2, 6, requires_grad=True)
        trainable = QuantumFeatures(10042)
        frozen = QuantumFeatures(10042, trainable=False)
        q = trainable(theta)
        self.assertEqual(q.shape, (2, 6))
        self.assertEqual(trainable.angles.numel(), 24)
        self.assertLessEqual(q.abs().max().item(), 1.000001)
        torch.testing.assert_close(q, frozen(theta))
        q.square().sum().backward()
        self.assertGreater(theta.grad.abs().sum().item(), 0)
        self.assertGreater(trainable.angles.grad.abs().sum().item(), 0)
        self.assertFalse(frozen.angles.requires_grad)
        tape = qml.workflow.construct_tape(trainable.circuit)(theta, trainable.angles)
        names = [op.name for op in tape.operations]
        self.assertEqual(names[:6], ["RY"] * 6)
        self.assertEqual(names.count("CZ"), 12)
        self.assertEqual(names.count("RY"), 18)
        self.assertEqual(names.count("RZ"), 12)
        self.assertEqual(names[-1], "RY")
        for block in range(2):
            start = 6 + block * 18
            self.assertEqual(names[start:start + 6], ["CZ"] * 6)
            self.assertEqual(names[start + 6:start + 18], ["RZ", "RY"] * 6)
        baseline = ProposedModel()
        for arm in "DE":
            model = TemporalAugmentationModel(baseline, arm)
            model.assert_initial_equivalence(torch.randn(2, 24, 7))


if __name__ == "__main__":
    unittest.main()
