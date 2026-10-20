"""Conditional, exploratory exact Wilcoxon comparisons of five paired RMSEs."""

import numpy as np
from scipy.stats import PermutationMethod, wilcoxon
from statsmodels.stats.multitest import multipletests

REFERENCE = "proposed"
COMPARATORS = (
    "cnn", "lstm", "cnn_lstm", "tcn", "dcnn_rbilstm",
    "proposed_no_quantum", "proposed_no_ta", "proposed_no_fa",
    "proposed_no_fusion", "proposed_no_dcnn", "proposed_no_rbilstm",
)
MODELS = (REFERENCE, *COMPARATORS)
RUNS = (1, 2, 3, 4, 5)
METRICS = ("rmse", "mae")
HORIZON = "horizon_15"
ALPHA = 0.05

METHOD_NOTES = (
    "DM cannot be calculated from aggregate metrics; predictions are unavailable.",
    "Five nonzero pairs give a minimum exact two-sided p-value of 0.0625. "
    "This analysis cannot establish significance at alpha=0.05 with five pairs. "
    "Nonsignificance does not establish equivalence.",
    "Exploratory: results were already inspected. These results concern "
    "run-to-run variability on one assumed common test period, not "
    "independent-dataset generalization.",
    "Pairing is inferred and conditional on matching runs 1-5 being valid paired "
    "training replicates on the same evaluation data. Neither historical seed "
    "identity nor dataset identity is verified by directory names/current config.",
    "The 60 metric JSONs and training logs contain no explicit seeds or dataset "
    "fingerprints. Logs span July/October and include legacy model paths "
    "(notably dcnn_rbilstm run 1 and proposed_no_quantum). README documents the "
    "historical proposed rename; legacy paths still leave provenance ambiguous.",
    "Inspected records do not explicitly identify conflicting paired seeds or "
    "datasets. Similar target ranges inferred from RMSE/nRMSE are compatible "
    "with a common target scale, but do not establish dataset/input identity. "
    "If pairing is contradicted, the paired tests must not be interpreted/run.",
    "RMSE only: exhaustive two-sided Wilcoxon sign permutations, average ranks "
    "for tied absolute differences, zero differences omitted (wilcox). All-zero "
    "differences give W=0, p=1. Holm covers all 11 declared comparisons; "
    "alpha=0.05. Exactness assumes independent, sign-exchangeable paired "
    "differences under the null (the signed-rank symmetry assumption).",
    "Differences = comparator - proposed in run order 1-5; positive favors "
    "proposed. W/L/T counts proposed wins/losses/ties. MAE is descriptive only. "
    "Means/sample SDs use run-level JSONs, never saved averages.",
)


def exact_wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    if differences.shape != (len(RUNS),) or not np.isfinite(differences).all():
        raise ValueError("Wilcoxon requires exactly five finite paired differences")
    if not np.any(differences):
        return 0.0, 1.0
    result = wilcoxon(
        differences, alternative="two-sided", zero_method="wilcox",
        method=PermutationMethod(n_resamples=np.inf), nan_policy="raise",
    )
    return float(result.statistic), float(result.pvalue)


def run_significance_analysis(metrics: dict) -> dict:
    if set(metrics) != set(MODELS):
        raise ValueError(f"Expected the complete declared model family: {MODELS}")
    comparisons = {}
    for model in COMPARATORS:
        row = {}
        for metric in METRICS:
            differences = metrics[model][metric] - metrics[REFERENCE][metric]
            row[f"{metric.upper()} paired differences"] = differences.tolist()
            row[f"mean {metric.upper()} difference"] = float(differences.mean())
            row[f"{metric.upper()} W/L/T"] = (
                f"{np.sum(differences > 0)}/{np.sum(differences < 0)}/{np.sum(differences == 0)}"
            )
        row["Wilcoxon W"], row["raw two-sided p"] = exact_wilcoxon(
            metrics[model]["rmse"] - metrics[REFERENCE]["rmse"]
        )
        comparisons[model] = row
    reject, adjusted, _, _ = multipletests(
        [row["raw two-sided p"] for row in comparisons.values()],
        alpha=ALPHA, method="holm",
    )
    for row, significant, p_value in zip(comparisons.values(), reject, adjusted):
        row["Holm-adjusted p"] = float(p_value)
        direction = "proposed better" if row["mean RMSE difference"] > 0 else "proposed worse"
        row["adjusted decision"] = (
            direction + " (conditional)" if significant else "not significant (conditional)"
        )
    return comparisons
