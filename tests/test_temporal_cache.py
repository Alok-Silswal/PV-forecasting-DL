import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.proposed_model import ProposedModel
from models.proposed_temporal.cache import CachedBaseline, CachedTemporalModel, prepare_cache
from models.proposed_temporal.model import TemporalAugmentationModel
from models.proposed_temporal.run_experiment import parse_args, print_run_summary, train_arm
from configs import config
from utils.logger import get_logger


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_seed_guard(self):
        import contextlib
        import io
        arguments = ["--processed-csv", "unused.csv"]
        self.assertEqual(list(parse_args(arguments).seeds), [42])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as rejection:
                parse_args(arguments + ["--seeds", "42", "43", "44"])
        self.assertEqual(rejection.exception.code, 2)
        explicit = parse_args(arguments + ["--seeds", "42", "43", "44", "--allow-multiple-seeds"])
        self.assertEqual(explicit.seeds, [42, 43, 44])

    def test_run_summary(self):
        import contextlib
        import io
        arguments = ["--processed-csv", "unused.csv"]
        datasets = {"train": range(8), "validation": range(4)}
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "new-cache"
            for options, count in (([], 3), (["--seeds", "42", "43", "44", "--allow-multiple-seeds"], 9),
                                   (["--build-cache-only"], 0)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    print_run_summary(parse_args(arguments + options), "cpu", datasets, cache_dir, 42)
                self.assertIn(f"Total training runs: {count}", output.getvalue())
                self.assertIn("Train samples: 8", output.getvalue())
                self.assertIn("Validation samples: 4", output.getvalue())
                self.assertIn("being generated", output.getvalue())
            cache_dir.mkdir()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_run_summary(parse_args(arguments), "cpu", datasets, cache_dir, 42)
            self.assertIn("cached (pending integrity verification)", output.getvalue())

    def test_extraction_reuse_and_cached_training(self):
        baseline = SimpleNamespace(baseline=ProposedModel().requires_grad_(False).eval(),
                                   sha256="checkpoint-one", checkpoint={})
        datasets = {split: TensorDataset(torch.randn(8, 24, 7), torch.randn(8, 3))
                    for split in ("train", "validation")}
        calls = []
        original = TemporalAugmentationModel.representations

        def counted(model, inputs):
            calls.append(len(inputs))
            return original(model, inputs)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "cache"
            rng_before = torch.random.get_rng_state().clone()
            with patch.object(TemporalAugmentationModel, "representations", counted):
                cached, identity = prepare_cache(directory, baseline, datasets, 24, {}, 4)
                self.assertEqual(calls, [4, 4, 4, 4])
                reused, same_identity = prepare_cache(directory, baseline, datasets, 24, {}, 4)
                self.assertEqual(calls, [4, 4, 4, 4])
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            self.assertEqual(identity, same_identity)
            extractor = TemporalAugmentationModel(baseline.baseline, "A").eval()
            for split, source in datasets.items():
                raw, target = source.tensors
                pooled, moments = extractor.representations(raw)
                for index in range(len(source)):
                    packed, cached_target = cached[split][index]
                    torch.testing.assert_close(packed[0], pooled[index])
                    torch.testing.assert_close(packed[1:], moments[index])
                    torch.testing.assert_close(cached_target, target[index], rtol=0, atol=0)
                    torch.testing.assert_close(packed, reused[split][index][0], rtol=0, atol=0)
            inputs, targets = next(iter(DataLoader(cached["train"], batch_size=4)))
            _, expected = CachedBaseline(baseline)(inputs)
            for arm in "ABC":
                model = CachedTemporalModel(baseline.baseline, arm, seed=42)
                with patch.object(model.backbone, "forward", side_effect=AssertionError("Backbone called")):
                    model.assert_initial_equivalence(inputs)
                    model.eval()
                    torch.testing.assert_close(model(inputs), expected)
                    model.train()
                    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
                    for _ in range(2):
                        optimizer.zero_grad()
                        torch.nn.functional.mse_loss(model(inputs), targets).backward()
                        optimizer.step()
                    model.eval()
                    self.assertEqual(model(inputs).shape, (4, 3))
                    if arm in "BC":
                        self.assertGreater(model.features.projection.weight.grad.abs().sum().item(), 0)
            online = TemporalAugmentationModel(baseline.baseline, "A")
            cached_a = CachedTemporalModel(baseline.baseline, "A")
            optimizers = [torch.optim.Adam(model.head.parameters()) for model in (online, cached_a)]
            raw = datasets["train"].tensors[0][:4]
            for _ in range(2):
                for model, optimizer, features in zip((online, cached_a), optimizers, (raw, inputs)):
                    model.train()
                    optimizer.zero_grad()
                    # Pair dropout masks as well as samples when comparing updates.
                    torch.manual_seed(42)
                    prediction = model(features)
                    torch.nn.functional.mse_loss(prediction, targets).backward()
                    optimizer.step()
                for key, value in online.head.state_dict().items():
                    torch.testing.assert_close(value, cached_a.head.state_dict()[key])
            with patch.object(config, "EXPERIMENTS_DIR", Path(temporary) / "experiments"), \
                    patch.object(config, "NUM_EPOCHS", 1), \
                    patch.object(TemporalAugmentationModel, "representations",
                                 side_effect=AssertionError("Runner re-extracted backbone")):
                for arm in "ABC":
                    metrics = train_arm(baseline, cached, arm, 42, 24, {}, {}, "cpu", identity)
                    self.assertIn("mse", metrics)
                    self.assertEqual(metrics["epoch"], 0)
                    logger = get_logger(config.EXPERIMENTS_DIR / f"proposed_temporal_{arm.lower()}" /
                                        "horizon_15" / "history_24" / "run_1")
                    for handler in logger.handlers[:]:
                        handler.close()
                        logger.removeHandler(handler)
            changed = SimpleNamespace(baseline=baseline.baseline, sha256="checkpoint-two")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                prepare_cache(directory, changed, datasets, 24, {}, 4)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                prepare_cache(directory, baseline, datasets, 12, {}, 4)
            for dataset in (*cached.values(), *reused.values()):
                dataset.close()
            with (directory / "train_features.npy").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                prepare_cache(directory, baseline, datasets, 24, {}, 4)


if __name__ == "__main__":
    unittest.main()
