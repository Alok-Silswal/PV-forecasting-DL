"""Resolve existing notebook outputs without importing config or changing data."""

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def processed_csv_candidates(project_root: Path = ROOT,
                             kaggle_working: Path | None = None) -> list[Path]:
    """Mirror the local/Kaggle processed-file locations in configs/config.py."""
    paths = [project_root / "data/processed/DKASC_Preprocessed.csv"]
    if kaggle_working is None and Path("/kaggle").exists():
        kaggle_working = Path("/kaggle/working")
    if kaggle_working is not None:
        paths.append(kaggle_working / "Processed.csv")
    return list(dict.fromkeys(path.resolve() for path in paths))


def path_diagnostics(processed_csv: Path | None = None, project_root: Path = ROOT,
                     kaggle_working: Path | None = None) -> str:
    """Report exact expected paths and available artifacts; never search recursively."""
    candidates = processed_csv_candidates(project_root, kaggle_working)
    if processed_csv is not None:
        candidates = list(dict.fromkeys([processed_csv.expanduser().resolve(), *candidates]))
    lines = [f"Project root: {project_root.resolve()}", f"Current working directory: {Path.cwd()}",
             "Processed CSV paths checked:"]
    lines += [f"  {path}: exists={path.exists()}, is_file={path.is_file()}" for path in candidates]
    artifact_dir = project_root / "artifacts"
    artifacts = sorted(path for path in artifact_dir.glob("*")
                       if path.is_file() and path.suffix in {".pt", ".pkl", ".json"})
    artifacts += sorted(path for path in (artifact_dir / "qubo_lag_selection").glob("*")
                        if path.is_file() and path.suffix in {".pt", ".pkl", ".json", ".npy"})
    lines.append("Relevant preprocessing artifacts present (not CSV substitutes):")
    lines += [f"  {path}" for path in artifacts] or ["  none"]
    gzip_paths = list(dict.fromkeys((base / "DKASC_Preprocessed.csv.gz").resolve()
                                   for base in [Path.cwd(), project_root, project_root / "notebooks",
                                                *[path.parent for path in candidates]]))
    lines.append("Notebook's optional cwd-relative compressed output:")
    lines += [f"  {path}: exists={path.exists()}, is_file={path.is_file()}" for path in gzip_paths]
    lines += ["Override: python -m experiments.residual_learning.run_residual_audit "
              "--extract-if-missing --smoke-test --processed-csv /path/to/existing/Processed.csv",
              "The first preparation notebook writes the CSV; the second reads it and writes .pt/scaler artifacts.",
              "No files are generated, copied, or modified by path resolution."]
    return "\n".join(lines)


def resolve_processed_csv(processed_csv: Path | None = None, project_root: Path = ROOT,
                          kaggle_working: Path | None = None) -> Path:
    """Require a unique known candidate, or honor an explicit existing file."""
    if processed_csv is not None:
        selected = processed_csv.expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError("Explicit processed CSV does not exist.\n" +
                                    path_diagnostics(processed_csv, project_root, kaggle_working))
    else:
        matches = [path for path in processed_csv_candidates(project_root, kaggle_working) if path.is_file()]
        if not matches:
            raise FileNotFoundError("Cannot resolve the historical processed CSV.\n" +
                                    path_diagnostics(None, project_root, kaggle_working))
        if len(matches) > 1:
            raise ValueError("Ambiguous processed CSV: use --processed-csv to select explicitly.\n" +
                             path_diagnostics(None, project_root, kaggle_working))
        selected = matches[0]
    logging.info("Selected processed CSV: %s (%s)", selected,
                 "explicit override" if processed_csv is not None else "known notebook output")
    return selected
