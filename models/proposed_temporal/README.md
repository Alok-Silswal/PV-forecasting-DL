# Temporal feature augmentation

Separate family; the finalized ProposedModel and existing results are unchanged.

- A: pretrained head continuation on mean-pooled features.
- B: append six mapped level/trend/curvature features.
- C: transform the same features with `6 -> 8 -> 6`, ReLU then tanh.
- E: trainable VQC; D: frozen VQC. These are factory-supported modules only.

The frozen backbone stays in eval mode. All head parameters train in A/B/C.
Expanded heads copy pretrained weights and zero the six new columns.
The fixed discrete polynomial basis has positive trend toward recent time and
positive curvature at both ends. Recent-history mode summarizes only the last
12 steps; classical pooling still uses all 24.

Run validation-only Step 0:

```powershell
python -m models.proposed_temporal.run_experiment --processed-csv data/processed/DKASC_Preprocessed.csv --arms A B C --history 24 --device cpu
python -m unittest discover -s tests -v
```

Initial screening defaults to seed 42: three A/B/C runs. Multiple seeds are
rejected before loading data unless `--allow-multiple-seeds` explicitly enables
later replication. The runner prints the planned run count before loading data.
Training budgets and arm definitions are unchanged.
The pre-training summary lists arms, seeds, run count, cache status/path, device,
and split sizes. Add `--build-cache-only` to extract/verify without training,
including when experiment output directories already exist. Use `--device cuda`
for cached head training on an available GPU; extraction remains on CPU for
consistent cache generation. CUDA is optional and unavailable devices fail
before loading the data.

Seeds 42–46 map to Proposed checkpoints in runs 1–5. The runner verifies the
finalized reconstructed validation split against each checkpoint before training.
It reuses Trainer, Adam, ReduceLROnPlateau, gradient clipping, and config budgets.
Independent sampler generators pair batch ordering across arms. No test data is
loaded. Metrics retain the existing standardized-target formulas (including
range-normalized nRMSE). Best-epoch validation metrics, histories, settings, and
checkpoints are saved under `experiments/proposed_temporal_<arm>/horizon_15/history_<T>/run_<N>`.
Existing destinations are never overwritten. Compare A/B/C manually; no pass rule
is encoded. Full screening training is intentionally not executed during implementation.

The runner extracts `x [N,128]` and `M [N,3,128]` once per pretrained checkpoint
and history setting for train and validation, preserving source sample order.
It stores float32 NumPy memory-mapped arrays under
`artifacts/proposed_temporal/cache/history_<T>/<checkpoint_sha256>` and shares
them across A/B/C. Different seeds use different pretrained checkpoints and
therefore require separate caches. Later invocations reuse verified caches.
The manifest checks provenance, extraction code/settings, and file hashes;
incomplete, stale, or corrupted caches are rejected. Validation gating uses the
cached pooled features and unchanged baseline head. Training and validation then
call only `forward_cached`, never the backbone. Projection and BatchNorm remain
live trainable components; their outputs are not cached. No preprocessing `.pt`
artifacts are created. `representations` and `forward_cached` remain available
for direct use outside the runner.
With the finalized split sizes, float32 features and targets occupy about
1.87 GB (1.74 GiB) per checkpoint/history cache. Full fused tensors and learned
projection outputs are not stored.

Construct D/E through `get_model('proposed_temporal_e', baseline=loaded_model,
seed=42)` (or `_d`). PennyLane is imported only for D/E. `quantum_seed` selects
independent frozen draws; default D/E initialization is paired. The circuit uses
one RY encoding, two CZ-ring/RZ/RY blocks, 24 angles, and six Z expectations.
No quantum experiment CLI is enabled yet. `feature_diagnostics` reports feature
and angle distributions, edge fractions, parameter movement, and prediction
change under a sample permutation of q. Its permutation statistic is a sensitivity
measure, not a forecasting metric; validation targets are needed to assess error.
