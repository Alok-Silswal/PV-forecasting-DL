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
Training/model source hashes and data provenance are checked. Train-only scaler
reconstruction must match the saved statistics before those exact statistics
are restored. No generated train/validation cache is required; no test cache
is built.

```powershell
.venv\Scripts\python.exe -m models.proposed_rvqc.evaluate --processed-csv data/processed/DKASC_Preprocessed.csv --run 1 --family proposed_rvqc
```
