"""Small model, circuit and existing-entry-point checks on synthetic data only."""

import csv
import importlib.util
import io
import logging
import math
import pickle
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import main
import evaluation.evaluate as evaluation
from configs import config
from models.model_factory import MODEL_NAMES, get_model
from models.proposed_no_quantum import ProposedNoQuantum
from models.quantum_simulator import RVQCSimulator, pennylane_reference
from training.loss import get_loss_function
from training.trainer import Trainer


def reference_class(commit, path, name):
    source = subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True,
                                     cwd=config.PROJECT_ROOT)
    namespace = {"__name__": "model_reference"}
    exec(compile(source, path, "exec"), namespace)
    return namespace[name]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = reference_class("496c26d", "models/proposed_model.py", "ProposedModel")
        cls.backbone_reference = reference_class("671df7e", "models/backbone.py", "ForecastBackbone")
        cls.quantum_reference = reference_class("994ccaa", "models/proposed_model.py", "ResidualForecast")

    def test_factory_and_preserved_original(self):
        inputs = torch.randn(3, 24, 7)
        for name in MODEL_NAMES:
            with self.subTest(model=name), torch.no_grad():
                model = get_model(name).eval()
                output = model(inputs)
                self.assertEqual(output.shape, (3, 3))
                self.assertTrue(torch.isfinite(output).all())
                if name == "proposed_no_quantum" or name.startswith("proposed_no_"):
                    self.assertIsInstance(model, ProposedNoQuantum)
                    flags = {key: getattr(model, key) for key in
                             ("use_feature_attention", "use_temporal_attention", "use_scalar_gated_fusion")}
                    old = self.original(**flags).eval()
                    self.assertEqual(list(old.state_dict()), list(model.state_dict()))
                    old.load_state_dict(model.state_dict(), strict=True)
                    torch.testing.assert_close(model(inputs), old(inputs), rtol=0, atol=0)
        for retired in ("proposed_qtm", "proposed_rvqc", "proposed_quantum"):
            with self.assertRaises(ValueError):
                get_model(retired)
        for seed in (42, 46):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                old = self.original()
                torch.manual_seed(seed)
                renamed = ProposedNoQuantum()
            for key, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, renamed.state_dict()[key]), key)

    def test_quantum_reference_initialization_and_checkpoint(self):
        for run in range(1, 6):
            seed = config.RANDOM_SEED + run - 1
            with patch.object(config, "RUN_NUMBER", run):
                before = torch.get_rng_state().clone()
                quantum = get_model("proposed")
                previous = self.quantum_reference()
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(list(quantum.state_dict()), list(previous.state_dict()))
            self.assertEqual(quantum.architecture_id, previous.architecture_id)
            for key, value in quantum.state_dict().items():
                self.assertTrue(torch.equal(value, previous.state_dict()[key]), key)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                backbone = self.backbone_reference()
            for key, value in quantum.backbone.state_dict().items():
                self.assertTrue(torch.equal(value, backbone.state_dict()[key]), key)
            generator = torch.Generator().manual_seed(seed)
            for name in ("weight", "bias"):
                value = getattr(quantum.projection, name)
                expected = torch.empty_like(value).uniform_(-1 / math.sqrt(128), 1 / math.sqrt(128), generator=generator)
                self.assertTrue(torch.equal(value, expected))
            expected_angles = torch.empty(2, 6, 2).uniform_(
                -0.1, 0.1, generator=torch.Generator().manual_seed(seed + 10000))
            self.assertTrue(torch.equal(quantum.quantum_angles, expected_angles))
            self.assertEqual(quantum.readout.weight.count_nonzero().item(), 0)
            self.assertEqual(quantum.readout.bias.count_nonzero().item(), 0)
            parts = quantum.eval().forward_components(torch.ones(2, 24, 7))
            self.assertEqual(parts["pooled"].shape, (2, 128))
            self.assertTrue(torch.equal(parts["final"], parts["base"]))
            quantum.load_state_dict(previous.state_dict(), strict=True)
            with torch.no_grad():
                quantum.readout.weight.fill_(0.1)
                quantum.readout.bias.fill_(0.2)
            previous.load_state_dict(quantum.state_dict(), strict=True)
            inputs = torch.randn(2, 24, 7)
            torch.testing.assert_close(quantum(inputs), previous.eval()(inputs), rtol=0, atol=0)

    def test_joint_gradients_after_readout_update(self):
        inputs, targets = torch.randn(4, 24, 7), torch.randn(4, 3)
        model = get_model("proposed").train()
        optimizer = main.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.assertEqual({id(p) for p in model.parameters()},
                         {id(p) for group in optimizer.param_groups for p in group["params"]})
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        for step in range(2):
            optimizer.zero_grad()
            parts = model.forward_components(inputs)
            torch.testing.assert_close(parts["final"], parts["base"] + parts["correction"], rtol=0, atol=0)
            get_loss_function()(parts["final"], targets).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            for module in (model.backbone.dcnn, model.backbone.feature_attention,
                           model.backbone.residual_bilstm, model.backbone.temporal_attention,
                           model.backbone.scalar_gated_fusion, model.backbone.mlp_head, model.readout):
                self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
            upstream = model.quantum_angles
            if step == 0:
                self.assertEqual(upstream.grad.count_nonzero().item(), 0)
                self.assertEqual(model.projection.weight.grad.count_nonzero().item(), 0)
            else:
                self.assertGreater(upstream.grad.abs().sum().item(), 0)
                self.assertGreater(model.projection.weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_tcn_causality_and_receptive_field(self):
        model = get_model("tcn").eval()
        inputs = torch.randn(2, 40, 7)
        changed = inputs.clone()
        changed[:, 12:] += 20
        torch.testing.assert_close(model.network(inputs.transpose(1, 2))[:, :, :12],
                                   model.network(changed.transpose(1, 2))[:, :, :12], rtol=0, atol=0)
        with torch.no_grad():
            for parameter in model.network.parameters():
                parameter.fill_(0.1)
        inputs = torch.ones(1, 40, 7, requires_grad=True)
        model.network(inputs.transpose(1, 2))[:, :, -1].sum().backward()
        self.assertEqual(inputs.grad[:, :11].count_nonzero().item(), 0)
        self.assertTrue((inputs.grad[:, 11:] > 0).all())
        self.assertEqual(model(inputs).shape, (1, 3))

    def test_torch_path_without_pennylane(self):
        with patch.dict(sys.modules, {"pennylane": None}):
            self.assertEqual(get_model("proposed").eval()(torch.ones(2, 24, 7)).shape, (2, 3))
        simulator = RVQCSimulator()
        for wire in range(6):
            state = torch.zeros(1, 64, dtype=torch.complex64)
            state[:, 0] = 1
            result = simulator.ry(state, torch.tensor([math.pi]), wire)
            self.assertEqual(result.abs().argmax().item(), 1 << (5 - wire))
            self.assertEqual(result.dtype, torch.complex64)

    @unittest.skipUnless(importlib.util.find_spec("pennylane"), "PennyLane reference unavailable")
    def test_simulator_against_independent_reference(self):
        import pennylane as qml
        reference, simulator = pennylane_reference(), RVQCSimulator()
        generator = torch.Generator().manual_seed(43)
        for batch in (1, 3, 7):
            theta = torch.randn(batch, 6, generator=generator, requires_grad=True)
            angles = (torch.randn(2, 6, 2, generator=generator) * 0.1).requires_grad_()
            actual = simulator(theta, angles)
            expected = torch.stack(reference(theta, angles), dim=1).float()
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            coefficients = torch.randn(batch, 6, generator=generator)
            actual_grads = torch.autograd.grad((actual * coefficients).sum(), (theta, angles))
            expected_grads = torch.autograd.grad((expected * coefficients).sum(), (theta, angles))
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)
        tape = qml.workflow.construct_tape(reference)(theta, angles)
        expected_gates = []
        for _ in range(2):
            expected_gates.extend((gate, [wire]) for wire in range(6) for gate in ("RY", "RZ", "RY"))
            expected_gates.extend(("CNOT", [wire, (wire + 1) % 6]) for wire in range(6))
        self.assertEqual([(op.name, list(op.wires)) for op in tape.operations], expected_gates)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable locally")
    def test_cuda_autograd(self):
        inputs = torch.randn(2, 24, 7, device="cuda")
        for name in ("proposed", "tcn"):
            model = get_model(name).cuda()
            model(inputs).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_existing_training_and_evaluation_routing(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(config.__dict__, config.__dict__.copy()):
            directory = Path(temporary)
            x, y = torch.randn(4, 24, 7), torch.rand(4, 3) + 1
            for name in ("train", "val", "test"):
                torch.save({"X": x, "y": y}, directory / f"{name}_15.pt")
            scaler = StandardScaler().fit(np.array([[0.], [1.], [2.]]))
            scaler_path = directory / "target_scaler.pkl"
            scaler_path.write_bytes(pickle.dumps(scaler))
            config.EXPERIMENTS_DIR, config.EVALUATION_DIR = directory / "experiments", directory / "evaluation"
            config.TRAIN_15_FILE, config.VAL_15_FILE = directory / "train_15.pt", directory / "val_15.pt"
            config.TEST_15_FILE, config.TARGET_SCALER_FILE = directory / "test_15.pt", scaler_path
            config.NUM_EPOCHS, config.BATCH_SIZE = 1, 4
            logger = logging.getLogger("model-tests")
            for name in ("proposed", "tcn"):
                argv = ["main.py", name, "--horizon", "15", "--run1"]
                with patch.object(sys, "argv", argv), patch("main._select_device", return_value=torch.device("cpu")), \
                        patch("main.get_logger", return_value=logger), patch("main.run_baseline_analysis"), \
                        patch("main.Trainer", wraps=Trainer) as trainer, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    main.main()
                dependencies = trainer.call_args.kwargs
                optimizer = dependencies["optimizer"]
                self.assertIsInstance(optimizer, torch.optim.Adam)
                self.assertEqual(len(optimizer.param_groups), 1)
                self.assertEqual({id(p) for p in dependencies["model"].parameters()},
                                 {id(p) for p in optimizer.param_groups[0]["params"]})
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], config.WEIGHT_DECAY)
                argv = ["evaluate", "--model", name, "--horizon", "15", "--run", "1"]
                with patch.object(sys, "argv", argv), patch("evaluation.evaluate._select_device", return_value=torch.device("cpu")), \
                        patch("evaluation.evaluate.EvaluationPlotter"), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    evaluation.main()
                checkpoint = torch.load(config.BEST_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
                self.assertEqual(set(checkpoint), {"epoch", "model_state_dict", "optimizer_state_dict",
                                                  "scheduler_state_dict", "best_val_loss"})
                with config.PREDICTIONS_FILE.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, ["Actual", "Predicted"])
                    rows = list(reader)
                self.assertEqual(len(rows), 12)
                np.testing.assert_array_equal([float(row["Actual"]) for row in rows],
                                              scaler.inverse_transform(y.numpy().reshape(-1, 1)).ravel())


if __name__ == "__main__":
    unittest.main()
