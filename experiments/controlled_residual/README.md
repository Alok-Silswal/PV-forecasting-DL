# Controlled residual comparison

Three forecasts share one pretrained, frozen ProposedModel: unchanged base (A), classical correction (B), quantum correction (C). Neither backbone fine-tuning nor architecture searches are supported. QTM and historical RVQC remain unchanged.

## Fixed corrections

Both use pooled fusion [B,128] → Linear(128,6) → pi*tanh → six features → Linear(6,3), added in standardized target units.

- B: Linear(6,6,bias=True) → tanh; 837 trainable parameters.
- C: exact existing RVQC: two blocks of input RY, trainable RZ/RY per wire, directed CNOT ring 0→1→2→3→4→5→0; six Z readouts. Data is reuploaded in block 2. There are 24 angles and 819 correction parameters.

These controls are approximately parameter-matched, not equivalent in capacity. Projection initialization is shared Uniform(±1/sqrt(128)), using a private seed. Both readouts start at zero. Quantum angles use Uniform(-0.1,0.1), seed+10000. Classical transform weights/biases use Uniform(±1/sqrt(6)), seed+11000. Upstream gradients appear after the first readout update.

Torch is the default batched float32/complex64 backend; it reuses QTM rotation primitives without changing QTM. The PennyLane CPU reference represents the identical RVQC circuit. Preflight checks output/input-gradient/angle-gradient agreement with atol=rtol=2e-5 on batches 1,3,7. No quantum advantage or hardware execution is claimed.

## Training and integrity

Run 1/seed 42 is the default. Both corrections use Adam (LR 1e-3, weight decay 1e-5), batch 256, 100 epochs maximum, patience 15, norm clipping 1.0, and CosineAnnealingLR(T_max=100, eta_min=1e-6). Private seed+20000+epoch permutations match across arms. Epoch 0 is a valid best-checkpoint candidate. Selection uses standardized final-forecast validation MSE only. Epoch numbering is 0=untrained, 1..100=completed training epochs.

Train/validation caches reuse RVQC's chronological windows and train-only scalers. Cache reuse checks checkpoint, history, dataset provenance, feature order, scaler state, shapes/counts, file hashes, and live first/middle/last samples (rtol=1e-5, atol=1e-6; targets exact). The frozen model is evaluated once during cache extraction, not inside correction epochs. Full validation-gate verification is separate; a matching saved cache gate can be reused.

Training saves best/last correction checkpoints, optimizer/scheduler state, history, settings, baseline identity, selected epoch, and duration. Resume and experiment/evaluation overwrites are rejected. Partial caches are rejected rather than repaired. Baseline checkpoint files must remain available at their original repository-relative paths.

## Commands (from repository root)

Use Kaggle's existing CUDA-enabled Torch. Keep source checkpoints at their repository-relative paths. The existing project config also requires the raw Combined_Output_All_Arrays.csv dataset to be attached on Kaggle, even when an explicit processed CSV is supplied. Install missing reference/data dependencies if necessary:

    python -m pip install "pennylane==0.45.1" pandas scikit-learn matplotlib tqdm
    python -c "import torch,pennylane,pandas,sklearn; print(torch.__version__,pennylane.__version__,torch.cuda.is_available())"
    python -m unittest discover -s tests -p "test_controlled_residual.py" -v

Bounded synthetic verification (no dataset):

    python -m experiments.controlled_residual.preflight --device cuda

Mandatory data validation preflight; supply your actual processed CSV path:

    python -m experiments.controlled_residual.preflight --processed-csv /kaggle/working/Processed.csv --device cuda --run 1

Run 1 training and subsequent evaluation (run explicitly when ready):

    python -m experiments.controlled_residual.run_experiment --processed-csv /kaggle/working/Processed.csv --device cuda --run 1
    python -m experiments.controlled_residual.evaluate --processed-csv /kaggle/working/Processed.csv --device cuda --run 1

Optional cache preparation uses the same runner with --build-cache-only. Local CPU commands can use data/processed/DKASC_Preprocessed.csv. Use --device cpu on CPU; --backend pennylane requires CPU. Additional runs require explicit --run 2 etc. No command defaults to all five runs.

## Outputs

- Cache: artifacts/controlled_residual/cache/horizon_15/run_1/ (~463 MiB float32 tensors per checkpoint).
- Training: experiments/controlled_residual/horizon_15/run_1/{classical,quantum}/; baseline validation and shared settings in the parent.
- Evaluation: evaluation/controlled_residual/horizon_15/run_1/; shared targets/base, all forecasts, metrics and metadata.

One reconstructed test dataset and one frozen-base forward per batch serve all three arms. One scaler performs float64 inverse transformations for all arms. Corrections are multiplied by scaler scale without adding its mean. Aggregate original-unit MSE/RMSE/MAE/R², horizon metrics, range-normalized nRMSE, correction magnitudes and signed differences from baseline are saved. MAPE is omitted. Negative differences mean improvement. Test data never selects checkpoints.

Historical run-2 target numerical discrepancies are not corrected or inherited: new evaluation uses one common target tensor and scaler. Single-run results do not establish significance, quantum advantage or publishability.
