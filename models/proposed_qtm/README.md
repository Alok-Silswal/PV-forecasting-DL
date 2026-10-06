# ProposedQTM

Final simulator-based hybrid architecture. The original ProposedModel and its
classical components remain unchanged. This is an end-to-end model, with no
frozen backbone, cached latents, screening arms, or auxiliary quantum predictor.

## Architecture and shapes

| Stage | Shape |
| --- | --- |
| Input X | `[B,24,7]` |
| Existing DCNN → Feature Attention | `[B,24,64]` |
| Xavier-initialized spatial projection 64 → 128 | S `[B,24,128]` |
| Existing Residual BiLSTM | H `[B,24,128]` |
| Shared bias-free linear scorer 128 → 1 | u `[B,24]` |
| Fixed DCT compression `u @ Phi` | z `[B,6]` |
| Affine-free BatchNorm → angle map | theta `[B,6]` |
| Six-qubit circuit | q `[B,6]` |
| `gamma * (q @ Phi.T)` | logits `[B,24]` |
| `a = 24 * softmax(logits, dim=1)` | a `[B,24]` |
| `T_q = H * a[:,:,None]` | `[B,24,128]` |
| Sample-dependent vector fusion | F `[B,24,128]` |
| Temporal mean | `[B,128]` |
| Existing MLP: Linear → ReLU → dropout → Linear | `[B,3]` |

Temporal Attention and Scalar Gated Fusion are not used in this family.
The spatial projection follows the existing fusion's width-matching design,
using Xavier weights and zero bias; it lives explicitly in the new model.
The temporal weighting adds no extra residual connection.

## Quantum temporal weighting

Phi is a float32 registered buffer constructed deterministically in float64:

`Phi[t,k] = sqrt(2/24) * cos(pi * (t+0.5) * k/24)`, for `t=0..23`, `k=1..6`.

Columns are orthonormal and exclude DC. The same Phi compresses the scorer's
24 values and reconstructs temporal logits. BatchNorm has no affine parameters;
running statistics are checkpointed and used in eval mode. The angle map is
`theta = (pi/2) * (1 + tanh(z_norm/2))`. Mathematically theta lies in `(0,pi)`;
float32 can round extreme saturated inputs to an endpoint.

The locked circuit uses wires 0–5 for DCT modes 1–6:

1. RY(theta_i), once on each wire.
2. Trainable RZ then RY on each wire.
3. CZ(0,1), CZ(1,2), CZ(2,3), CZ(3,4), CZ(4,5).
4. Trainable RZ then RY on each wire.
5. Six Pauli-Z expectations.

There are 24 quantum angles, initialized Normal(0,0.1²) using a private generator
at run seed + 10000. There is no ring, re-uploading, extra depth, terminal CZ, or
ZZ readout. Gamma is one unconstrained learnable scalar initialized to 1.
Temporal weights sum to 24; uniform weights make temporal weighting the identity.
Small initial circuit angles yield approximately neutral weighting, not forced
exact agreement with the classical ProposedModel.

## Vector fusion and parameter counts

`g = sigmoid(Linear(256,128)(concat(mean(S), mean(T_q))))`.
`F = g[:,None] * T_q + (1-g[:,None]) * S`.

The gate varies by sample and channel, and is shared across time. Its weights
use Normal(0,0.01²), with zero bias, placing initial gates around 0.5.

Default trainable parameter counts: quantum angles **24**, gamma **1**, timestep
scorer **128**, vector fusion **32,896**, complete model **103,204**. DCT and
simulator sign tables are nontrainable buffers. All model parameters train.

## Simulation and precision

`QTM_QUANTUM_BACKEND = "torch"` selects a batched differentiable `[B,64]`
complex64 statevector. Only fixed wire/gate loops are used, never sample loops.
The five commuting CZ gates share one precomputed diagonal sign table. Z
expectations use state probabilities. Buffers follow `.to(device)`; the normal
forward path makes no CPU/GPU transfers.

`"pennylane"` selects the matching batched `default.qubit` Torch/backprop QNode,
primarily for reference tests. Both backends use identical parameter names and
the same circuit. PennyLane is imported only for the reference backend/tests.

Scoring/DCT, normalization, angles, quantum gates, expectations, and temporal
logits run outside autocast in float32; the Torch statevector is complex64.
Weights are converted to H's dtype when applying them. Classical components
can be run under autocast; do not convert the whole model's parameters to fp16.
The standard project Trainer currently trains in float32: this patch does not
enable AMP/GradScaler automatically. CPU autocast boundary checks passed. CUDA
and CUDA AMP tests are available but require an actual GPU.

## Training and evaluation

Use the existing preprocessing artifacts and standard pipeline from repository
root. After Kaggle GPU quota resets, run seed 42 only:

```bash
python main.py proposed_qtm --horizon 15 --run1
```

Main selects CUDA when available. Configuration defaults remain `MODEL_NAME =
"proposed"`; model selection is explicit. Existing model names and optimizer
behavior are preserved. ProposedQTM uses AdamW with the existing classical LR
and weight decay. Angles and gamma use zero weight decay and LR multiplier
`QTM_QUANTUM_LR_MULTIPLIER = 3.0`; this is one fixed setting, not a search.
BatchNorm has no learnable parameters. Existing MSE, ReduceLROnPlateau, epoch
budget, early stopping, clipping at 1.0, logger, and checkpoints are reused.
The existing seed protocol maps run 1 to 42, run 2 to 43, etc.

Output checkpoints follow `experiments/proposed_qtm/horizon_15/run_1/` and can be
loaded by the normal evaluator. Evaluation later, after training:

```bash
python -m evaluation.evaluate --model proposed_qtm --horizon 15 --run 1
```

No dataset, scaler, split, evaluation formula, or shared Trainer was changed.
No full training or test evaluation was performed during implementation.

## Opt-in diagnostics

`model.forward_with_diagnostics(X)` returns predictions and batch aggregates:
mean/std temporal profile, entropy, q mean/std, gamma, theta range, and fraction
within 0.05 radians of an endpoint. It evaluates the circuit once and stores no
sample history. Diagnostics are not collected automatically during training.

In eval mode, `model.predict_uniform(X)` bypasses weighting (`T_q=H`) as an
inference-only sensitivity check, with no retraining or trained control arm.

## Engineering verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m models.proposed_qtm.preflight --device cpu
```

Preflight uses 16 synthetic windows, defaults to 40 optimization steps, and is
capped at 100. Its throughput profile warms up then measures only three batches
per backend at batch sizes 16 and 64. It never reads forecasting datasets.

On the current CPU with two Torch threads, all nine applicable tests passed;
the CUDA test was skipped. Reference maximum absolute errors were 5.96e-7 for
outputs, 1.67e-6 for input gradients, and 2.03e-6 for angle gradients. Tests also
cover DCT modes/orthogonality, complete gradient flow, shapes, normalization
restore, checkpoint roundtrip, vector fusion, and standard Trainer integration.

The tiny overfit reduced eval MSE from 0.7842 to 0.04754 (93.94%) in 40 steps.
This is engineering evidence only, not a forecasting result.

| CPU quantum forward + backward | Torch | PennyLane |
| --- | --- | --- |
| Batch 16 | 67.0 ms | 129.4 ms |
| Batch 64 | 82.8 ms | 109.2 ms |

These few-batch timings are indicative, not a full-training or GPU estimate.
Simulator performance is hardware-dependent; GPU throughput still needs testing.
This is simulation-only. No quantum computational advantage or quantum speedup
is claimed. Forecasting claims require later benchmark results against the
selected conventional deep-learning models.
