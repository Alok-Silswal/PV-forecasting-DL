# PV forecasting

- `proposed`: an end-to-end hybrid model with additive outputs from the
  forecast head and quantum residual module.
- `proposed_no_quantum`: the preserved original Proposed model; historical
  results now live under this name.
- `tcn`: causal 32-channel residual blocks, dilations 1/2/4, receptive field 29.

The pooled fusion [B,128] feeds the 128-64-3 forecast head and a parallel quantum
residual module: a 128-6 projection, pi*tanh, a six-qubit circuit and a 6-3 readout.
The final prediction is the forecast-head output plus the quantum residual.
The quantum residual module has two repeated RY/RZ/RY encoding/rotation blocks
and the ordered CNOT ring 0-1-2-3-4-5-0.
The readout starts at zero; upstream gradients in the quantum residual module
start at zero and become active after readout updates. Everything is trained
jointly with the existing comparison-model Adam policy, trainer, loss and run
seeds. PennyLane is optional;
the active simulator uses batched float32/complex64 PyTorch on CPU or CUDA.

Quantum ablations retain the same additive output structure: `proposed_no_ta`
bypasses temporal attention; `proposed_no_fa` bypasses feature attention;
`proposed_no_fusion` uses `0.5 * spatial_projection(spatial) + 0.5 * temporal`.
All retain the forecast head and full quantum residual module. They initialize
fresh with matching retained weights at each run seed. These are new experiments;
component contributions remain untested until training. Use the same commands
below with the corresponding model name.

`proposed_no_dcnn` and `proposed_no_rbilstm` remove one encoder using a trainable
per-timestep linear replacement: 7-to-64 and 7-to-128 at the default configured
widths. Both retain feature/temporal attention, learned fusion, the forecast head
and quantum residual module. Replacement seeds use private offsets +20000/+30000;
retained weights match the full model. Parameter counts differ between variants.

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
