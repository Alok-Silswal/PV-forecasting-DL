"""Run with: python analysis/run_statistical_analysis.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import EVALUATION_DIR, STATISTICAL_ANALYSIS_DIR
from analysis.run_analysis import run_analysis
from analysis.significance_tests import METRICS


def main() -> None:
    try:
        results = run_analysis(EVALUATION_DIR)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print("\n".join(results.notes))
    compact = results.table.drop(columns=[f"{metric.upper()} paired differences" for metric in METRICS])
    print(compact.to_string(index=False, na_rep="-", float_format=lambda value: f"{value:.6g}"))
    STATISTICAL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.table.to_csv(STATISTICAL_ANALYSIS_DIR / "metric_comparisons.csv", index=False, na_rep="")


if __name__ == "__main__":
    main()
