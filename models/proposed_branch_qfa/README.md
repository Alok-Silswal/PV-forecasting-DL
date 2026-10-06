# Branch-aware feature augmentation

Final VQC placement candidate; the ProposedModel forward and scalar fusion are
unchanged. This family has its own runner and output directories.

## Extraction and cache

The actual spatial branch is width 64 after Feature Attention. `FrozenBranches`
hooks the existing fusion `spatial_projection` output to obtain S `[B,24,128]`.
T is captured at the fusion input after Temporal Attention. F is captured at
the MLP input. No modules or operations are replaced.

Each checkpoint is evaluated once per train/validation sample. Float32 NumPy
memory maps store rows `[x, S_bar, T_bar]` `[N,3,128]` and targets `[N,3]`, in
source order. Neither full sequences nor learned projection outputs are stored.
The cache is about 1.40 GB (1.31 GiB) per checkpoint for the finalized splits.

Location: `artifacts/proposed_branch_qfa/cache/<checkpoint_sha256>/`.
Identity includes checkpoint path/hash, 24-step history, source/scaler
provenance, split counts, shapes, dtype, extraction code hashes, and PyTorch
version. Reuse verifies identity and file hashes. Incomplete/incompatible caches
are rejected; no automatic overwrite or rebuilding occurs. Cache extraction
runs on CPU. Cached head training supports CPU or explicit CUDA.

The existing finalized CSV reconstruction is reused from `proposed_rvqc.data`.
It fits scalers only on training rows and never reads test rows. The existing
checkpoint-validation gate runs on cached x with the pretrained head.

## Arms

| Arm | Additional features | Trainable components |
| --- | --- | --- |
| A | None | Pretrained prediction head |
| B | Direct mapped branch factors | Shared projection and augmented head |
| C | `6 -> 8 -> 6`, ReLU then tanh | Shared projection, transform, augmented head |
| E, later | Trainable VQC expectations | Shared projection, VQC, augmented head |
| D, later | Frozen VQC expectations | Shared projection and augmented head |

B/C/D/E apply one shared bias-free `128 -> 3` projection to S_bar and T_bar
(384 weights), concatenate `[s1,s2,s3,t1,t2,t3]`, then apply affine-free
BatchNorm and `theta = (pi/2) * (1 + tanh(z/2))`. The head copies all pretrained
weights, with six zero input columns appended to its first layer. Initial
predictions match the pretrained function in eval mode. The frozen backbone
remains in eval mode under `.train()`; cached training never forwards it.

## Stage 0

```powershell
python -m models.proposed_branch_qfa.run_experiment --processed-csv data/processed/DKASC_Preprocessed.csv --arms A B C --device cpu
python -m models.proposed_branch_qfa.run_experiment --processed-csv data/processed/DKASC_Preprocessed.csv --build-cache-only --device cpu
```

Default seed is 42: exactly three training runs. Seeds 42–46 map to pretrained
Proposed runs 1–5. Later replication requires both explicit `--seeds` and
`--allow-multiple-seeds`. The CLI accepts only A/B/C; it cannot launch quantum
training. It prints run count, cache status, checkpoint, device, split counts,
and cache path before extraction/training. `--threads` defaults to 4.

All arms use config Adam LR/weight decay, ReduceLROnPlateau, gradient clipping,
epoch budget, and early stopping. All head parameters train at the same LR as
new components. Private sampler/loader generators pair complete batch order
across arms for each seed; dropout RNG is reset through the existing seed
protocol. Validation uses the existing standardized-target metric formulas.

Initial validation is checkpoint candidate epoch `-1` (displayed as epoch 0).
Later checkpoints use the shared Trainer's zero-based epoch indexing. Selection
is lowest validation MSE, including the initial candidate, identically for all
arms. No shared training infrastructure is modified.

Results live under `experiments/proposed_branch_qfa_<arm>/horizon_15/run_<N>/`.
The runner saves settings/provenance, histories, initial and best validation
metrics (MSE, RMSE, MAE, MAPE, R², nRMSE), and the usual best checkpoint. A combined
JSON report records effect sizes relative to A. Existing arm outputs are never
overwritten or implicitly resumed. Cache-only mode does not train or require
empty experiment directories.

Decision remains manual: best(B,C) should improve A's validation RMSE by at least
0.3%, with MAE no worse. Gains around 0.1–0.3% are insufficient to justify quantum
training. A B MAE increase above 2% is flagged descriptively; nothing triggers
recalibration, reruns, or quantum training. If Stage 0 fails, abandon this final
placement; no further placements or variant searches are part of this family.

## Optional VQC

PennyLane imports only when D/E are instantiated via the factory. Qubits 0–2
encode spatial factors; 3–5 encode temporal factors. The circuit is one RY
encoding, RZ/RY block 1, CZ(0,3)/CZ(1,4)/CZ(2,5), RZ/RY block 2, six Z expectations.
There are 24 trainable angles, no ring or intra-branch edges, no re-uploading,
and no final CZ readout layer. D/E share initialization by default;
`quantum_seed` can select a later frozen draw. The default is Uniform(-0.1,0.1)
using a private generator at seed + 10000. No D/E experiment CLI is enabled.

Tests (only tiny synthetic batches; quantum test optional in system Python):

```powershell
python -m unittest discover -s tests -v
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
