# Frozen Proposed residual experiments

Families: `proposed_rvqc` (24 trainable circuit angles) and
`proposed_rvqc_frozen` (the same initial angles held fixed).
Projection/readout tensors and all epoch permutations are paired exactly.
The classical Proposed checkpoint stays frozen and in evaluation mode.

## Preflight

```powershell
.venv\Scripts\python.exe -m models.proposed_rvqc.preflight --processed-csv data/processed/DKASC_Preprocessed.csv --run 1
```

Only the first 905495 data rows are read. Both scalers fit only the first
745702 rows. Obsolete local `.pt` artifacts and scalers are never used.
Window inputs are rows `[s,s+24)`; targets are `[s+24,s+27)`, within each
split. Expected counts: train 745676, validation 159767.

The full frozen validation gate must match checkpoint-associated history
before residual checks or cache acceptance. MSE tolerance is `rtol=1e-5,
atol=1e-7`; normalized metric tolerance is `rtol=1e-4, atol=1e-6` to allow
float32 backend/reduction differences. No tolerance is adapted to results.
An exactly matching saved successful gate may be reused with `--reuse-gate`.
Smoke checks use eight windows per split and one residual optimizer update
per family, not a scientific training run.

## Locked protocol

- Run seeds: 42, 43, 44, 45, 46.
- Common initialization generator: run seed.
- Angles: private generator at seed + 10000; `Uniform(-0.1,0.1)`, `[2,6,2]`.
- Epoch permutations: private generator at seed + 20000 + zero-based epoch.
- Each block: data RY, variational RZ/RY, CNOT ring `0→1→2→3→4→5→0`.
- Six Pauli-Z measurements, trainable `Linear(6,3)` readout.
- Adam, LR 0.001, weight decay 0.00001, batch 256, 100 epochs, patience 15.
- CosineAnnealingLR: `T_max=100`, `eta_min=1e-6`, stepped after each epoch.
- Select lowest validation MSE; no baseline fallback.

The analytic PennyLane `default.qubit` circuit uses the
[PyTorch backprop interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html).
This implementation uses CPU execution. Cached float32 `[N,128]`, `[N,3]`,
`[N,3]` arrays are shared by the paired families and checked against manifest
hashes. Approximate cache storage is 485 MB per run, excluding metadata.

## Explicit experiment commands

Do not run these until reviewing the preflight results.

```powershell
.venv\Scripts\python.exe -m models.proposed_rvqc.run_experiment --processed-csv data/processed/DKASC_Preprocessed.csv --run 1 --family both
```

Repeat for runs 2–5. The runner repeats the full reconstruction gate and all
preflight checks before accepting/building the full cache. It never accesses
test rows, and refuses implicit resume or overwriting scientific outputs.
Epoch order and LR remain identical for every epoch shared by both families;
early stopping may give different training lengths.

Training/evaluation retain the existing Proposed filenames and schemas.
Separate `settings.json` and `preflight.json` record protocol/provenance.
`baseline_metrics.json` follows the existing final-epoch convention, not the
best-checkpoint validation score.

Test evaluation requires only the selected completed run's settings, preflight
evidence and residual checkpoint, plus its original Proposed checkpoint.
Training/model source hashes and data provenance are checked. The notebook hash
is informational; executable `data.py` provenance remains enforced. Train-only
scaler reconstruction must match the saved statistics before those exact statistics
are restored. No generated train/validation cache is required; no test cache
is built.

```powershell
.venv\Scripts\python.exe -m models.proposed_rvqc.evaluate --processed-csv data/processed/DKASC_Preprocessed.csv --run 1 --family proposed_rvqc
```

## Validation-only depth pilot

The official branch, defaults and checkpoints remain unchanged. The isolated
depth extension uses the official circuit directly at two blocks. Other depths
change only the repeated block count. Common tensors and earlier quantum blocks
have identical initialization for the same seed; epoch permutations are shared.

First run the lightweight checks (four-window optimizer smoke steps, not full training):

```powershell
python -m models.proposed_rvqc.depth_preflight --processed-csv /kaggle/working/Processed.csv --run 1
```

Then launch each validation-only pilot explicitly:

```powershell
python -m models.proposed_rvqc.depth_pilot --processed-csv /kaggle/working/Processed.csv --run 1 --blocks 1
python -m models.proposed_rvqc.depth_pilot --processed-csv /kaggle/working/Processed.csv --run 1 --blocks 2
python -m models.proposed_rvqc.depth_pilot --processed-csv /kaggle/working/Processed.csv --run 1 --blocks 3
python -m models.proposed_rvqc.depth_pilot --processed-csv /kaggle/working/Processed.csv --run 1 --blocks 4
python -m models.proposed_rvqc.depth_summary
```

`--blocks 1 2 3 4` also runs depths sequentially. Use `--run 2` for seed 43
and only the manually selected depths. For a matched control, add
`--family proposed_rvqc_frozen`; controls are never launched automatically.

Trainable outputs are `experiments/proposed_rvqc_depth_pilot/blocks_L/run_N/`;
controls are under `experiments/proposed_rvqc_depth_pilot/frozen/blocks_L/run_N/`.
Shared caches are isolated under `artifacts/proposed_rvqc/cache/depth_pilot/`.
Existing pilot directories are not resumed or overwritten implicitly.
No official results or `evaluation/` outputs are written.

The bounded train/validation reconstruction and full historical baseline
validation gate run before cache creation/training. Test rows are never read.
Adam, cosine schedule, batch size, stopping and loss follow the official protocol;
the first trained epoch is eligible and there is no baseline/epoch-zero fallback.

`validation_metrics.json` reports the best checkpoint's metrics in
**train-standardized target units**, matching the official residual trainer.
Epochs are one-based. `history.json` includes each epoch's learning rate;
`quantum_angle_analysis.json` compares initial angles with the best checkpoint,
including each block's changes. Pilot checkpoints contain the residual branch
state plus baseline checkpoint provenance, not a second baseline copy.
The summary reads only completed pilot validation outputs, verifies checkpoint
hashes, writes `depth_summary.csv`, and never ranks or selects a depth.
Baseline columns use the frozen Proposed validation gate's measured MSE/RMSE.
Summary deltas are pilot minus baseline; negative values indicate improvement.
Per-block angle movement stays in `quantum_angle_analysis.json`; the CSV contains
only overall movement and validation metrics, with no test-set columns.
