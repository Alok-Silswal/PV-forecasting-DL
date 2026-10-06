"""Mandatory simulator equivalence when PennyLane is installed."""

import importlib.util
import unittest

import torch

from models.proposed_qtm.quantum_simulator import SixQubitSimulator, pennylane_reference


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_state_norm_and_batching(self):
        generator = torch.Generator().manual_seed(42)
        theta = torch.rand(8, 6, generator=generator) * torch.pi
        angles = torch.randn(2, 6, 2, generator=generator) * 0.1
        simulator = SixQubitSimulator()
        state = simulator.statevector(theta, angles)
        self.assertEqual(state.shape, (8, 64))
        self.assertEqual(state.dtype, torch.complex64)
        torch.testing.assert_close(state.abs().square().sum(1), torch.ones(8), rtol=1e-5, atol=1e-5)
        result = simulator(theta, angles)
        self.assertEqual(result.shape, (8, 6))
        self.assertLessEqual(result.abs().max().item(), 1.00001)
        torch.testing.assert_close(result[:4], simulator(theta[:4], angles))
        # With identity trainable blocks, CZ cannot change Z expectations.
        torch.testing.assert_close(simulator(theta, torch.zeros_like(angles)), theta.cos(), rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(importlib.util.find_spec("pennylane"), "PennyLane reference unavailable")
    def test_reference_outputs_gradients_and_topology(self):
        import pennylane as qml
        reference = pennylane_reference()
        simulator = SixQubitSimulator()
        generator = torch.Generator().manual_seed(43)
        for batch_size in (1, 4):
            for scale in (0.1, 0.7):
                theta = (torch.rand(batch_size, 6, generator=generator) * torch.pi).requires_grad_()
                angles = (torch.randn(2, 6, 2, generator=generator) * scale).requires_grad_()
                actual = simulator(theta, angles)
                expected = torch.stack(reference(theta, angles), dim=1).float()
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                coefficients = torch.randn(batch_size, 6, generator=generator)
                grads = torch.autograd.grad((actual * coefficients).sum(), (theta, angles))
                reference_grads = torch.autograd.grad((expected * coefficients).sum(), (theta, angles))
                for actual_grad, expected_grad in zip(grads, reference_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)
        tape = qml.workflow.construct_tape(reference)(theta, angles)
        names = [op.name for op in tape.operations]
        self.assertEqual(len(reference.device.wires), 6)
        self.assertEqual(names[:6], ["RY"] * 6)
        self.assertEqual(names[6:18], ["RZ", "RY"] * 6)
        self.assertEqual(names[18:23], ["CZ"] * 5)
        self.assertEqual(names[23:], ["RZ", "RY"] * 6)
        self.assertEqual([list(op.wires) for op in tape.operations[18:23]], [[i, i + 1] for i in range(5)])
        self.assertEqual([measurement.obs.name for measurement in tape.measurements], ["PauliZ"] * 6)
        self.assertEqual(angles.numel(), 24)


if __name__ == "__main__":
    unittest.main()
