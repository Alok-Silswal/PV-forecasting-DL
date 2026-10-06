"""Engineering checks on synthetic inputs; no dataset training/evaluation."""

import io
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from configs import config
from models.model_factory import get_model
from models.proposed_model import ProposedModel
from models.proposed_qtm.quantum_temporal import QuantumTemporalWeighting
from models.proposed_qtm.vector_gated_fusion import VectorGatedFusion
from training.trainer import Trainer


class QTMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self):
        torch.manual_seed(42)
        return get_model("proposed_qtm", seed=42)

    def test_shapes_and_gradients(self):
        model = self.make_model().train()
        inputs = torch.randn(4, 24, 7)
        captured = {}
        hooks = []
        for name in ("dcnn", "feature_attention", "spatial_projection", "residual_bilstm",
                     "quantum_temporal", "vector_gated_fusion", "mlp_head"):
            def capture(module, args, output, key=name):
                captured[key] = output.shape
            hooks.append(getattr(model, name).register_forward_hook(capture))
        try:
            prediction = model(inputs)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(prediction.shape, (4, 3))
        self.assertEqual(captured["feature_attention"], (4, 24, 64))
        for name in ("spatial_projection", "residual_bilstm", "quantum_temporal", "vector_gated_fusion"):
            self.assertEqual(captured[name], (4, 24, 128))
        self.assertEqual(captured["mlp_head"], (4, 3))
        nn.functional.mse_loss(prediction, torch.randn(4, 3)).backward()
        for name, parameter in model.named_parameters():
            self.assertTrue(parameter.requires_grad, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module in (model.dcnn, model.feature_attention, model.spatial_projection,
                       model.residual_bilstm, model.quantum_temporal.scorer,
                       model.quantum_temporal.quantum, model.vector_gated_fusion, model.mlp_head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        self.assertGreater(model.quantum_temporal.gamma.grad.abs().item(), 0)

    def test_dct_and_weights(self):
        temporal = QuantumTemporalWeighting().eval()
        basis = temporal.basis
        self.assertEqual(basis.shape, (24, 6))
        torch.testing.assert_close(basis.T @ basis, torch.eye(6), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(basis.sum(0), torch.zeros(6), rtol=0, atol=1e-6)
        expected = math.sqrt(2 / 24) * torch.cos(
            math.pi * (torch.arange(24, dtype=torch.float64)[:, None] + 0.5)
            * torch.arange(1, 7, dtype=torch.float64)[None, :] / 24)
        torch.testing.assert_close(basis, expected.float(), rtol=0, atol=0)
        self.assertIn("basis", dict(temporal.named_buffers()))
        self.assertNotIn("basis", dict(temporal.named_parameters()))
        self.assertIsNone(temporal.scorer.bias)
        self.assertFalse(temporal.normalization.affine)
        hidden = torch.zeros(4, 24, 128)
        weights, q, theta = temporal.weighting_details(hidden)
        torch.testing.assert_close(theta, torch.full((4, 6), math.pi / 2))
        self.assertEqual(q.shape, (4, 6))
        self.assertEqual(weights.shape, (4, 24))
        torch.testing.assert_close(weights.sum(1), torch.full((4,), 24.0))
        self.assertTrue(torch.isfinite(weights).all())
        # Neutral input with small random angles should not concentrate mass.
        self.assertLess(weights.max().item(), 1.5)
        self.assertGreater(weights.min().item(), 0.5)
        hidden = torch.randn(4, 24, 128)
        with torch.no_grad():
            temporal.gamma.zero_()
        torch.testing.assert_close(temporal(hidden), hidden)

    def test_vector_fusion(self):
        fusion = VectorGatedFusion()
        spatial, temporal = torch.randn(4, 24, 128), torch.randn(4, 24, 128)
        gate = fusion.gate(spatial, temporal)
        self.assertEqual(gate.shape, (4, 128))
        self.assertTrue(((gate > 0) & (gate < 1)).all())
        expected_gate = torch.sigmoid(fusion.gate_generator(torch.cat((spatial.mean(1), temporal.mean(1)), dim=1)))
        torch.testing.assert_close(gate, expected_gate)
        torch.testing.assert_close(fusion(spatial, temporal), gate[:, None] * temporal + (1 - gate[:, None]) * spatial)
        zeros = torch.zeros_like(spatial)
        torch.testing.assert_close(fusion.gate(zeros, zeros), torch.full((4, 128), 0.5))
        self.assertEqual(sum(p.numel() for p in fusion.parameters()), 32896)

    def test_state_dict_and_batchnorm(self):
        model = self.make_model().train()
        inputs = torch.randn(4, 24, 7)
        normalization = model.quantum_temporal.normalization
        model(inputs)
        self.assertEqual(normalization.num_batches_tracked.item(), 1)
        model.eval()
        before = {key: value.clone() for key, value in normalization.state_dict().items()}
        with torch.no_grad():
            expected = model(inputs)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.make_model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
        for key, value in before.items():
            torch.testing.assert_close(normalization.state_dict()[key], value, rtol=0, atol=0)

    def test_parameter_groups_and_pipeline(self):
        import logging
        model = self.make_model()
        groups = model.optimizer_parameter_groups(config.LEARNING_RATE, config.WEIGHT_DECAY,
                                                  config.QTM_QUANTUM_LR_MULTIPLIER)
        quantum_ids = {id(model.quantum_temporal.quantum.angles), id(model.quantum_temporal.gamma)}
        self.assertEqual({id(p) for p in groups[1]["params"]}, quantum_ids)
        self.assertEqual(groups[1]["weight_decay"], 0)
        self.assertEqual(groups[1]["lr"], config.LEARNING_RATE * config.QTM_QUANTUM_LR_MULTIPLIER)
        grouped = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters()})
        optimizer = torch.optim.AdamW(groups)
        dataset = TensorDataset(torch.randn(8, 24, 7), torch.randn(8, 3))
        loader = DataLoader(dataset, batch_size=4, generator=torch.Generator().manual_seed(42))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(model, loader, loader, nn.MSELoss(), optimizer, scheduler,
                              torch.device("cpu"), logging.getLogger("qtm-test"), directory,
                              2, gradient_clip_value=1.0, num_epochs=1)
            history = trainer.train()
            self.assertEqual(len(history["val_loss"]), 1)
            self.assertTrue(Path(trainer.checkpoint_path).is_file())
            trainer.load_checkpoint(trainer.checkpoint_path)
            self.assertEqual(trainer.start_epoch, 1)

    def test_diagnostics_and_uniform_inference(self):
        model = self.make_model().eval()
        inputs = torch.randn(4, 24, 7)
        with torch.no_grad():
            expected = model(inputs)
            prediction, report = model.forward_with_diagnostics(inputs)
        torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
        self.assertEqual(len(report["temporal_weight_mean"]), 24)
        self.assertEqual(len(report["q_mean"]), 6)
        self.assertGreater(report["temporal_weight_entropy_mean"], 0)
        self.assertEqual(report["gamma"], 1)
        with torch.no_grad():
            spatial = model.spatial_projection(model.feature_attention(model.dcnn(inputs)))
            hidden = model.residual_bilstm(inputs)
            expected_uniform = model.mlp_head(model.vector_gated_fusion(spatial, hidden).mean(1))
            torch.testing.assert_close(model.predict_uniform(inputs), expected_uniform, rtol=0, atol=0)
        model.train()
        with self.assertRaisesRegex(ValueError, "inference-only"):
            model.predict_uniform(inputs)

    def test_autocast_boundary_and_reproducibility(self):
        temporal = QuantumTemporalWeighting().eval()
        state = torch.random.get_rng_state().clone()
        other = QuantumTemporalWeighting().eval()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for key, value in temporal.state_dict().items():
            torch.testing.assert_close(other.state_dict()[key], value, rtol=0, atol=0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            weights, q, theta = temporal.weighting_details(torch.randn(4, 24, 128, dtype=torch.bfloat16))
        for value in (weights, q, theta):
            self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(temporal.quantum.simulator.statevector(theta, temporal.quantum.angles).dtype, torch.complex64)
        model = self.make_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 103204)
        self.assertIsInstance(get_model("proposed"), ProposedModel)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_and_amp(self):
        model = self.make_model().cuda().train()
        inputs = torch.randn(4, 24, 7, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(inputs)
            loss = nn.functional.mse_loss(prediction, torch.randn(4, 3, device="cuda"))
        loss.backward()
        self.assertEqual(prediction.shape, (4, 3))
        self.assertTrue(all(p.device.type == "cuda" for p in model.parameters()))
        self.assertTrue(all(b.device.type == "cuda" for b in model.buffers()))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
