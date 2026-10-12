# Controlled residual recovery report

Inspection date: 2026-10-12. No notebook was executed; no training, test evaluation, commit, push, or Git mutation was performed. Both notebooks and all existing recovery evidence were preserved. This report is a new derived file.

## Recovered notebook evidence

The files in pv-recovery/run_N are derived evidence, not original metrics.json, history.json, settings.json, checkpoints, or prediction arrays. For all runs, the extracted evaluation JSON parses and equals the saved notebook evaluation output; all three arms are present. The extracted training console text matches the saved training cell after line-ending normalization, and contains both correction-arm logs in classical-then-quantum order.

| Run/seed | Notebook eval JSON | Notebook B/C training output | Original B/C checkpoints, histories/settings, metadata, caches, per-sample predictions |
|---|---|---|---|
| 1/42 | baseline/classical/quantum | both arms present | not found locally |
| 2/43 | baseline/classical/quantum | both arms present | not found locally |
| 3/44 | baseline/classical/quantum | both arms present | not found locally |
| 4/45 | baseline/classical/quantum | both arms present | not found locally |
| 5/46 | baseline/classical/quantum | both arms present | not found locally |

The original ProposedModel checkpoint and its source history.json exist locally for runs 1–5. data/processed/DKASC_Preprocessed.csv is present. These do not replace residual checkpoints or predictions.

### Selected epochs and printed validation metrics

Values are from selected-epoch console lines, rounded to seven decimals. They are not full-precision histories. Validation metrics are in the training output's standardized target scale.

| Run/seed | Classical: epoch; val MSE/RMSE/MAE | Quantum: epoch; val MSE/RMSE/MAE |
|---|---|---|
| 1/42 | 15; 0.0244469/0.1563549/0.0534279 | 31; 0.0243999/0.1562046/0.0528833 |
| 2/43 | 41; 0.0243611/0.1560805/0.0531447 | 24; 0.0243652/0.1560934/0.0530281 |
| 3/44 | 3; 0.0244108/0.1562395/0.0533347 | 3; 0.0244108/0.1562395/0.0533347 |
| 4/45 | 7; 0.0244642/0.1564103/0.0532943 | 7; 0.0244642/0.1564103/0.0532943 |
| 5/46 | 38; 0.0244085/0.1562322/0.0533405 | 25; 0.0244228/0.1562780/0.0537697 |

### Saved notebook evaluation summary

Copied from the parsed notebook JSON; no new metrics were computed. Full results remain in each metrics_from_notebook.json.

| Run/seed | Baseline A RMSE/MAE | Classical B RMSE/MAE | Quantum C RMSE/MAE |
|---|---:|---:|---:|
| 1/42 | 11.034017/4.271579 | 11.043376/4.217204 | 11.046456/4.176602 |
| 2/43 | 10.865557/4.134019 | 10.895395/4.122221 | 10.884090/4.114103 |
| 3/44 | 10.877750/4.116823 | 10.895141/4.121295 | 10.900298/4.131160 |
| 4/45 | 10.982371/4.288687 | 10.984499/4.137029 | 10.987144/4.194845 |
| 5/46 | 10.949132/4.273577 | 10.983739/4.200329 | 10.977988/4.147093 |

The run preambles record seeds 42–46, Torch backend, CUDA device, 745,676/159,767 train/validation windows, and baseline checkpoint paths. They do not preserve exact full settings/history; source defaults alone do not prove execution settings.

## Notebook provenance

The manifest names notebooks/pv-forecasting-dl.ipynb but its SHA-256 398AAD959045A09A9C519C782D9922F72327A9860B3806A019E5AC89212DD149 matches neither file present: original notebook DCF78DFD303FD0411C2B890D595D250B2C0BE372C409F5C236F283B4CB7E9D0B; redacted notebook 0CA41C13B0CF7423434AE167C05390050DB688868A7DCE7DEEF283EC4DE6DEDE. Cells 10–19 have matching source and outputs in both notebooks and match the extracted evidence. The original has 55 cells; the redacted copy has 70, and only the redacted copy contains incident cells 65/67. The hash discrepancy cannot be attributed specifically to credential redaction without the pre-redaction file; it does not invalidate the matching run outputs.

The redacted notebook's saved cells record commit 60e9afa (137 files, 7,202,133 insertions, 200 deletions) and HTTP 408/remote disconnect after writing a 2.25 GiB pack. “Everything up-to-date” is not confirmation of persistence.

## Local artifacts and prediction alignment

No files were found under the expected controlled-residual cache, experiment, or evaluation paths. No relevant archive or Git bundle was found in the project or scoped Downloads. Two Downloads NPZ files named assessment_predictions*.npz were inspected by keys/shapes only; one contains baseline/ridge/MLP arrays, the other a quantum-6Q-reuploading prediction and centered residual. Neither contains run/seed identity or the paired classical/quantum outputs for this experiment. They were preserved.

Original ProposedModel prediction CSVs exist for all five runs (479,304 flattened rows each) and may provide candidate baseline forecasts after alignment checks. Target values in Proposed run 2 differ from the old RVQC run-2 CSV in 312,246/479,304 positions; max absolute difference 3.0517578125e-5, mean 1.7879632424450831e-6. The RVQC run-2 targets match Proposed run 1. Proposed runs 2–5 share the same target text; run 1 differs at small numerical precision. These historical files were not modified.

The controlled-residual evaluator loads one test dataset/scaler and shares targets/base forecasts across A/B/C within each run, so it avoids within-run arm target drift by design. Notebook JSON has no target arrays, so cross-run equality for the controlled outputs cannot be verified. Do not substitute historical Proposed CSVs for missing controlled outputs without checking target order and values.

## Git state

Local main and local origin/main point to 39044be9d930f4e97a2d64e25a92c6fc8901b984. The local origin/main tree includes source commit 5fbe4e22ef8298ab6ee3b5b93740218accb44067, which adds controlled-residual source/tests only; no five-run outputs appear in that history. 60e9afa does not resolve locally; inspection of 3,263 Git objects/244 commits found no matching object prefix or commit message. Initial status showed the original notebook and pv-recovery as untracked.

A live remote refresh could not connect to GitHub. Current remote main and remote commit availability are therefore unverified; this access failure is not evidence that the commit is absent. Prior independent checks reported GitHub did not resolve 60e9afa and did contain the controlled-residual source; current local refs agree on the source but could not refresh GitHub state.

The current tree has 43 already-tracked legacy prediction CSVs larger than 10 MiB (about 18.5–19.2 MB each); preserve those historical results. Existing .gitignore rules cover artifacts/**/cache, generated npy/npz/pkl files, checkpoints, and controlled evaluator prediction outputs. Ignore rules do not untrack committed files.

## Kaggle recovery and next commands

Status: no access to inspect saved versions. Notebook metadata/cells contain no Kaggle URL, owner, or slug; public title search found no match. Kaggle CLI is not installed and local Kaggle config is absent. No versions/output inventories were inspected, so there is no evidence that matching outputs were absent or never persisted.

The local notebook filename identifies a local file, not its Kaggle URL. Please provide the exact Kaggle notebook URL or owner/kernel-slug. After local CLI setup, enumerate versions in Kaggle's version history, then inspect/download each relevant version into a new directory:

    python -m pip install kaggle
    kaggle kernels list -m -s "PV-forecasting" --page-size 100 -p 1 --sort-by dateRun
    kaggle kernels files OWNER/SLUG/VERSION --page-size 200
    kaggle kernels output OWNER/SLUG/VERSION -p pv-recovery/kaggle_version_VERSION

Repeat the version-qualified files/output commands for every version; follow file-list page tokens, and verify sizes, hashes, and run coverage. The official Kaggle CLI supports version-qualified refs and paged output downloads: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md. These are inspection/download commands, not run/publish commands.

## Recovery and Git proposal

Aggregate metrics cannot reconstruct per-origin predictions. Baseline CSVs may supply arm A after target/order validation. Arms B/C require saved correction checkpoints and matching settings/identity; none were found locally. If Kaggle versions yield checkpoints/settings, evaluation-only regeneration is possible after validating checkpoint, processed CSV, scaler/split identity, and a fresh output directory. No training is needed in that case. If only notebook summaries survive, paired per-origin DM tests are unavailable.

The oversized commit is absent from local Git objects, so do not reset this repository to remove it. No history cleanup was performed. If an accessible copy later appears, back up its Git objects and results first; remove only generated large files from tracking while preserving external copies. A later deletion commit alone would leave large blobs earlier in history. Going forward, keep source and compact metadata in Git, store checkpoints/scalers/arrays/predictions in durable versioned storage, and maintain a compact manifest for source commit, dataset/scaler, seed, settings, dependency versions, hashes, and locations. Verify two durable copies before ending Kaggle, then inspect staged names and sizes.

Read-only commands for a network-available Git recheck:

    git status --short
    git ls-remote origin refs/heads/main
    git cat-file -e '60e9afa^{commit}'
    git diff --cached --stat
    git diff --cached --name-only

No files were extracted from archives; notebooks, saved outputs, and existing result artifacts were left untouched.
