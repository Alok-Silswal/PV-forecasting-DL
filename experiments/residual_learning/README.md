# Stage 0 residual audit

Run from project root. Existing models, preprocessing and artifacts remain read only.

```bash
python -m experiments.residual_learning.run_residual_audit --extract-if-missing --smoke-test
python -m experiments.residual_learning.run_residual_audit --extract-if-missing
```

Or extract separately:

```bash
python -m experiments.residual_learning.extract_residual_dataset --batch-size 256
python -m experiments.residual_learning.run_residual_audit
```

CUDA is automatic; CPU fallback uses one thread and flushes subnormal numbers.
Required inputs are the existing processed CSV and the original
`experiments/proposed/horizon_15/run_1/checkpoints/best_checkpoint.pt`.
Locally the CSV is `data/processed/DKASC_Preprocessed.csv`. On Kaggle the first
preparation notebook writes `/kaggle/working/Processed.csv` through
`config.PROCESSED_DATA_FILE`; the second notebook reads that file and writes
repository-root `artifacts/*.pt` and scalers. The extractor detects these two
known CSV locations, logs its choice, and rejects ambiguity. It never uses old
six-feature `.pt` files as substitutes or regenerates/copies preprocessing data.

Read-only diagnostics and an explicit Kaggle override:

```bash
python -m experiments.residual_learning.extract_residual_dataset --diagnose-paths
python -m experiments.residual_learning.run_residual_audit --extract-if-missing --smoke-test --processed-csv /kaggle/working/Processed.csv
```

Both scripts accept `--processed-csv PATH`. Relative overrides resolve from cwd.
External source paths are recorded in the manifest and still undergo the same
contract checks and SHA256 verification when reusing extracted datasets.
The optional `DKASC_Preprocessed.csv.gz` output is written relative to the first
notebook's cwd; it is reported by diagnostics but not automatically selected.
It may be selected explicitly with `--processed-csv` (pandas detects compression).

The original validation period supplies 159,767 windows. Boundaries are 60%/80%
of window starts. Discarding starts `[95860,95886)` and `[127813,127839)` leaves:

| Block | Start indices (end exclusive) | Samples |
|---|---|---:|
| Residual training | `[0,95860)` | 95,860 |
| Tuning | `[95886,127813)` | 31,927 |
| Assessment | `[127839,159767)` | 31,928 |

The 27-row input/target footprints are disjoint across blocks. Overlap within
each block is preserved. All scalers for residual learners fit the training block
only. Ridge selects among five alphas on tuning. The small MLP selects its epoch
on tuning. Models/settings are saved before assessment is loaded.

`--max-samples N` caps **each** block at its first N samples and writes exclusively
to `_smoke/max_N/proposed/horizon_15/run_1`; it defaults to three MLP epochs. `--smoke-test` uses N=1024.
Smoke results are not research evidence. Existing output directories are never
overwritten; partial extraction directories also cause an explicit error.

Outputs:

- `artifacts/residual_learning/proposed/horizon_15/run_1/`: three compressed NPZ
  datasets, provenance manifest, `residual_controls/` scalers, models and settings.
- `evaluation/residual_learning/proposed/horizon_15/run_1/`: JSON/text summary,
  aggregate/per-output metrics CSV, assessment predictions NPZ, two plots.
- Smoke outputs use a separate `_smoke/max_N/` tree under both residual-learning roots.

This is exploratory: validation already selected the frozen backbone. Historical
gaps, interpolation and preprocessing leakage risks remain unchanged. Output
horizons are three future rows, not guaranteed five-minute intervals. Aggregate
metrics pool all outputs; per-output metrics and chronological thirds are also
saved. Negative error deltas mean improvement. There is no automatic GO threshold.

No original test data is extracted or used. Deleting the new residual-learning
directories removes this experiment without changing the baseline project.
