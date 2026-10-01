"""Read-only resolution behavior, using synthetic files outside the real datasets."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.residual_learning.data_paths import path_diagnostics, resolve_processed_csv


class ProcessedCSVPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.working = Path(self.temporary.name) / "working"
        self.local = self.root / "data/processed/DKASC_Preprocessed.csv"
        self.kaggle = self.working / "Processed.csv"

    def create(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic path test\n", encoding="utf-8")

    def resolve(self, explicit: Path | None = None) -> Path:
        return resolve_processed_csv(explicit, self.root, self.working)

    def test_local_layout(self) -> None:
        self.create(self.local)
        self.assertEqual(self.resolve(), self.local.resolve())

    def test_historical_kaggle_layout(self) -> None:
        self.create(self.kaggle)
        self.assertEqual(self.resolve(), self.kaggle.resolve())

    def test_ambiguity_requires_override(self) -> None:
        self.create(self.local)
        self.create(self.kaggle)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.resolve()
        self.assertEqual(self.resolve(self.kaggle), self.kaggle.resolve())

    def test_explicit_external_and_relative_paths(self) -> None:
        self.create(self.kaggle)
        before = self.kaggle.read_bytes()
        self.assertEqual(self.resolve(self.kaggle), self.kaggle.resolve())
        relative = Path(os.path.relpath(self.kaggle, Path.cwd()))
        self.assertEqual(self.resolve(relative), self.kaggle.resolve())
        self.assertEqual(before, self.kaggle.read_bytes())

    def test_missing_reports_context_and_artifacts(self) -> None:
        artifact = self.root / "artifacts/val_15.pt"
        self.create(artifact)
        with self.assertRaises(FileNotFoundError) as caught:
            self.resolve()
        message = str(caught.exception)
        for text in ("Project root", "Current working directory", str(self.local),
                     str(self.kaggle), str(artifact), "exists=False", "--processed-csv"):
            self.assertIn(text, message)

    def test_invalid_explicit_path_never_falls_back(self) -> None:
        self.create(self.local)
        with self.assertRaisesRegex(FileNotFoundError, "Explicit"):
            self.resolve(self.working / "missing.csv")

    def test_no_recursive_or_tensor_fallback(self) -> None:
        self.create(self.root / "other/DKASC_Preprocessed.csv")
        self.create(self.root / "artifacts/val_15.pt")
        self.create(self.root / "notebooks/DKASC_Preprocessed.csv.gz")
        with self.assertRaises(FileNotFoundError):
            self.resolve()
        self.assertIn("DKASC_Preprocessed.csv.gz", path_diagnostics(None, self.root, self.working))

    def test_kaggle_detection_matches_config_without_importing_it(self) -> None:
        with patch("experiments.residual_learning.data_paths.Path.exists", return_value=True):
            from experiments.residual_learning.data_paths import processed_csv_candidates
            self.assertIn(Path("/kaggle/working/Processed.csv").resolve(), processed_csv_candidates(self.root))


if __name__ == "__main__":
    unittest.main()
