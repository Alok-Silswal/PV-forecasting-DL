# PV forecasting

- `proposed`: jointly trained original backbone/head plus a quantum residual.
- `proposed_no_quantum`: the preserved original Proposed model; historical
  results now live under this name.
- `tcn`: causal 32-channel residual blocks, dilations 1/2/4, receptive field 29.

The quantum residual model uses pooled fusion [B,128], the existing 128-64-3 head,
and a parallel 128-6 projection followed by pi*tanh and a 6-3 correction readout.
The quantum branch has two repeated RY/RZ/RY encoding/rotation blocks and the
ordered CNOT ring 0-1-2-3-4-5-0.
The readout starts at zero; upstream correction gradients start at zero and become
active after readout updates. Everything is trained jointly with the existing
comparison-model Adam policy, trainer, loss and run seeds. PennyLane is optional;
the active simulator uses batched float32/complex64 PyTorch on CPU or CUDA.

Quantum ablations use the same end-to-end residual model: `proposed_no_ta`
bypasses temporal attention; `proposed_no_fa` bypasses feature attention;
`proposed_no_fusion` uses `0.5 * spatial_projection(spatial) + 0.5 * temporal`.
All retain the forecast head and full quantum correction. They initialize fresh
with matching retained weights at each run seed. These are new experiments;
component contributions remain untested until training. Use the same commands
below with the corresponding model name.

Use the existing preprocessing notebooks in order on Kaggle:
`PV_Dataset_Analysis_AND_Preparation.ipynb`, then `Data_Preprocessing.ipynb`.
They retain the original preparation and train-only scaling method. The configured
`artifacts/train_15.pt`, `val_15.pt` and `test_15.pt` must contain
`X: [N,24,7]` and `y: [N,3]`; evaluation uses the existing saved target scaler.
The current local six-feature/one-target artifacts are incompatible and unchanged.
Re-execute the original notebooks on Kaggle before training with incompatible
artifacts. Existing Kaggle configuration also expects the raw dataset attached.

From the repository root, run 1 uses the existing training and evaluation CLIs:
```bash
python main.py proposed --horizon 15 --run1
python -m evaluation.evaluate --model proposed --horizon 15 --run 1
python main.py tcn --horizon 15 --run1
python -m evaluation.evaluate --model tcn --horizon 15 --run 1
```

The existing checkpoint/resume policy and output conventions are unchanged.
Historical log contents and metric values are preserved. Matching tensor shapes
do not prove identical historical dataset provenance.

Git ignore rules exclude bulky generated tensors, scalers, caches, checkpoints
and prediction arrays/CSVs; they do not persist Kaggle outputs. Save required
artifacts and experiment outputs durably before ending a Kaggle session.
