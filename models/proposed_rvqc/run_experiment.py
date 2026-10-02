"""Train one joint Proposed-RVQC run; smoke/benchmark never access test data.

Run from project root: python -m models.proposed_rvqc.run_experiment --run-number 1.
"""

import argparse
import hashlib
import io
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from configs import config
from main import _set_seed
from training.loss import get_loss_function
from training.trainer import Trainer
from experiments.residual_learning.extract_residual_dataset import (
    ROOT, FEATURES, TRAIN_END, EXPECTED_ROWS, LOOKBACK, OUTPUTS,
    resolve_processed_csv, scaler_stats, sha256, save_json,
)
from models.proposed_rvqc import ProposedRVQC

VAL_END = TRAIN_END + int(0.15 * EXPECTED_ROWS)
TRAINING_ROOT = ROOT / "experiments/proposed_rvqc"
EVALUATION_ROOT = ROOT / "evaluation/proposed_rvqc"


def fingerprint(path: Path) -> str:
    """Keep Git text newline conversion from invalidating unchanged results."""
    if path.suffix in (".py", ".json", ".csv"):
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return sha256(path)


class WindowDataset(Dataset):
    """Notebook create_sequences indexing without materializing overlapping windows."""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 starts: np.ndarray | None = None) -> None:
        self.features = features
        self.targets = targets
        count = len(features) - LOOKBACK - OUTPUTS + 1
        self.starts = np.arange(count) if starts is None else starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        return (torch.from_numpy(self.features[start:start + LOOKBACK].copy()),
                torch.from_numpy(self.targets[start + LOOKBACK:start + LOOKBACK + OUTPUTS].copy()))


def load_fitting_data(csv_path: Path) -> tuple[WindowDataset, WindowDataset, StandardScaler, StandardScaler, dict]:
    """Read training/validation rows only; preserve the historical train-only scaling."""
    # Bound the parser input so read-ahead cannot consume original test rows.
    with csv_path.open("rb") as handle:
        prefix = b"".join(handle.readline() for _ in range(VAL_END + 1))
    frame = pd.read_csv(io.BytesIO(prefix), parse_dates=["timestamp"])
    source = {"fitting_csv_sha256": hashlib.sha256(prefix).hexdigest(),
              "test_row_byte_offset": len(prefix), "csv_columns": list(frame.columns)}
    validate_frame(frame, VAL_END)
    feature_scaler = StandardScaler().fit(frame[FEATURES].iloc[:TRAIN_END])
    target_scaler = StandardScaler().fit(frame.Active_Power.iloc[:TRAIN_END].to_numpy().reshape(-1, 1))
    manifest_path = ROOT / "artifacts/residual_learning/proposed/horizon_15/run_1/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Verified historical scaler contract missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["features"] != FEATURES or manifest["split_row_boundaries"] != {
            "train": [0, TRAIN_END], "validation": [TRAIN_END, VAL_END], "test": [VAL_END, EXPECTED_ROWS]}:
        raise ValueError("Historical feature/split contract changed.")
    for name, scaler in (("feature_scaler", feature_scaler), ("target_scaler", target_scaler)):
        for key, value in scaler_stats(scaler).items():
            np.testing.assert_allclose(value, manifest[name][key], atol=1e-12, rtol=1e-12)
    features = feature_scaler.transform(frame[FEATURES]).astype(np.float32)
    targets = target_scaler.transform(frame.Active_Power.to_numpy().reshape(-1, 1))[:, 0].astype(np.float32)
    return (WindowDataset(features[:TRAIN_END], targets[:TRAIN_END]),
            WindowDataset(features[TRAIN_END:], targets[TRAIN_END:]), feature_scaler, target_scaler, source)


def validate_frame(frame: pd.DataFrame, expected_rows: int) -> None:
    if len(frame) != expected_rows or not set(["timestamp", *FEATURES]).issubset(frame.columns):
        raise ValueError("Processed CSV row/column contract differs.")
    if not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError("Nonfinite processed features.")
    if frame.timestamp.isna().any() or frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError("Processed timestamps must be valid, unique and sorted.")


def historical_snapshot() -> dict[Path, str]:
    # Source checks must not read historical test data or prediction archives.
    paths = [ROOT / name for name in ("main.py", "evaluation/evaluate.py", "evaluation/evaluator.py")]
    for directory in ("models", "training", "configs", "notebooks", "experiments/residual_learning"):
        paths += [p for p in (ROOT / directory).rglob("*") if p.is_file()
                  and p.suffix in (".py", ".ipynb")
                  and "__pycache__" not in p.parts and "proposed_rvqc" not in p.parts]
    return {p: sha256(p) for p in paths if p.is_file()}


def assert_protected(snapshot: dict[Path, str]) -> None:
    for path, digest in snapshot.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Historical file changed: {path}")


def gradient_check(model: ProposedRVQC, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Require live gradients in every named path and all 24 quantum angles."""
    captured: dict[str, torch.Tensor] = {}
    handles = [model.rvqc.projection.register_forward_hook(
        lambda m, i, o: captured.update(projection=o)),
        model.rvqc.readout.register_forward_pre_hook(lambda m, i: captured.update(quantum=i[0]))]
    original_hooks = len(model.backbone.scalar_gated_fusion._forward_hooks)
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        components = model.forward_components(inputs)
        quantum_to_fusion = torch.autograd.grad(
            components["correction_cpu"].square().mean(), components["fusion"], retain_graph=True)[0]
        if not torch.isfinite(quantum_to_fusion).all() or not torch.count_nonzero(quantum_to_fusion):
            raise RuntimeError("RVQC-only objective does not reach the live backbone fusion tensor.")
        for name in ("fusion", "pooled", "cpu_latent", "correction_cpu", "correction"):
            components[name].retain_grad()
        loss = get_loss_function()(components["final"], targets)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    expected = {"fusion": [len(inputs),24,128], "pooled": [len(inputs),128],
                "projection": [len(inputs),6], "quantum": [len(inputs),6],
                "correction": [len(inputs),3], "final": [len(inputs),3]}
    shapes = {key: list((captured if key in captured else components)[key].shape) for key in expected}
    if shapes != expected or len(model.backbone.scalar_gated_fusion._forward_hooks) != original_hooks:
        raise RuntimeError("Forward shape or hook cleanup failed.")
    groups = {name: getattr(model.backbone, name) for name in (
        "dcnn", "feature_attention", "residual_bilstm", "temporal_attention", "scalar_gated_fusion", "mlp_head")}
    groups.update(projection=model.rvqc.projection, readout=model.rvqc.readout)
    gradient_norms = {}
    for name, module in groups.items():
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"Missing/nonfinite gradient in {name}.")
        gradient_norms[name] = sum(float(g.norm()) for g in grads)
        if gradient_norms[name] == 0:
            raise RuntimeError(f"Zero gradient path: {name}.")
    quantum = model.rvqc.weights.grad
    if quantum is None or quantum.numel() != 24 or not torch.isfinite(quantum).all() or (quantum == 0).any():
        raise RuntimeError(f"Missing/zero/nonfinite quantum gradient: {quantum}")
    for name in ("fusion", "pooled", "cpu_latent", "correction_cpu", "correction"):
        grad = components[name].grad
        if grad is None or not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
            raise RuntimeError(f"Broken autograd transfer: {name}")
    torch.testing.assert_close(components["final"], components["baseline"] + components["correction"], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        ordinary = model.backbone(inputs)
        parts = model.forward_components(inputs)
        torch.testing.assert_close(ordinary, parts["baseline"], rtol=0, atol=0)
    if model.rvqc.parameter_counts() != {"projection":774,"quantum":24,"readout":21,"total":819}:
        raise RuntimeError("RVQC parameter count changed.")
    model.zero_grad(set_to_none=True)
    return {"shapes": shapes, "gradient_norms": gradient_norms,
            "input_shape": list(inputs.shape), "quantum_only_fusion_gradient_norm": float(quantum_to_fusion.norm()),
            "quantum_min_abs_gradient": float(quantum.abs().min()), "quantum_parameters": 24,
            "backbone_device": str(inputs.device), "rvqc_device": "cpu",
            "mixed_device_verified": inputs.device.type == "cuda", "hook_prediction_identity_exact": True,
            "live_transfer_gradients": True, "loss": float(loss.detach())}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model: ProposedRVQC, train: WindowDataset, validation: WindowDataset,
              device: torch.device) -> dict:
    """Measure actual final-model batches, including transfers and full backward."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    times = {"training": [], "validation": []}
    for phase, dataset in (("training", train), ("validation", validation)):
        starts = np.concatenate([np.arange(start, start + config.BATCH_SIZE)
                                 for start in np.linspace(0, len(dataset) - config.BATCH_SIZE, 4, dtype=int)])
        representative = WindowDataset(dataset.features, dataset.targets, starts)
        for index, (inputs, targets) in enumerate(DataLoader(representative, batch_size=config.BATCH_SIZE)):
            inputs, targets = inputs.to(device), targets.to(device)
            model.train(phase == "training")
            sync(device); started = time.perf_counter()
            with torch.set_grad_enabled(phase == "training"):
                loss = get_loss_function()(model(inputs), targets)
                if phase == "training":
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
                    optimizer.step()
            sync(device)
            if index > 0:
                times[phase].append(time.perf_counter() - started)
            if index == 3:
                break
    means = {key: float(np.mean(value)) for key, value in times.items()}
    train_seconds = means["training"] * int(np.ceil(len(train) / config.BATCH_SIZE))
    validation_seconds = means["validation"] * int(np.ceil(len(validation) / config.BATCH_SIZE))
    return {"batch_size": config.BATCH_SIZE, "measured_batches_per_phase": 3,
            "warmup_batches_excluded": 1, "batch_seconds": means,
            "extrapolated_training_seconds": train_seconds, "extrapolated_validation_seconds": validation_seconds,
            "extrapolated_epoch_seconds": train_seconds + validation_seconds,
            "extrapolated_100_epoch_seconds": 100 * (train_seconds + validation_seconds),
            "extrapolated_16_epoch_seconds": 16 * (train_seconds + validation_seconds),
            "extrapolated_test_seconds": means["validation"] * int(np.ceil((EXPECTED_ROWS-VAL_END-26)/config.BATCH_SIZE)),
            "note": "Actual joint-model batch timing; extrapolation excludes CSV loading/epoch I/O. "
                    "Patience does not predict selected epoch: 16 is the earliest stopping epoch, not a forecast."}


def make_trainer(model: ProposedRVQC, train: WindowDataset, validation: WindowDataset,
                 device: torch.device, destination: Path, epochs: int) -> Trainer:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=config.SCHEDULER_MODE, factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE, min_lr=config.SCHEDULER_MIN_LR)
    return Trainer(model, DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_TRAIN,
                                    num_workers=config.NUM_WORKERS),
                   DataLoader(validation, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS),
                   get_loss_function(), optimizer, scheduler, device, logging.getLogger("proposed_rvqc"),
                   destination / "checkpoints", config.EARLY_STOPPING_PATIENCE,
                   config.GRADIENT_CLIP_VALUE, epochs)


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.run_number <= config.NUM_RUNS:
        raise ValueError("Run number outside project NUM_RUNS.")
    if config.NUM_EPOCHS != 100 or config.EARLY_STOPPING_PATIENCE != 15 or config.ACTIVE_HORIZON != '15':
        raise ValueError("Expected finalized 100 epoch/patience 15/horizon 15 config.")
    if (config.FEATURE_COLUMNS != FEATURES or config.LOOKBACK != LOOKBACK or config.STRIDE != 1
            or config.TARGET_COLUMN != 'Active_Power' or config.HORIZON_TO_OUTPUT_DIM['15'] != OUTPUTS):
        raise ValueError("Project feature/window/target contract differs from the verified experiment.")
    if qml.__version__ != '0.45.1':
        raise ValueError("Validated RVQC requires PennyLane 0.45.1.")
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='auto' else args.device if args.device!='auto' else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; cross-device verification cannot be claimed.")
    csv_path=resolve_processed_csv(args.processed_csv)
    seed=config.RANDOM_SEED+args.run_number-1
    sources=[Path(__file__), ROOT/"models/proposed_rvqc/evaluate.py",ROOT/'models/proposed_rvqc/proposed_rvqc.py',ROOT/'models/proposed_rvqc/__init__.py',ROOT/'models/proposed_model.py',
             ROOT/'models/residual_learning/quantum_residual_vqc.py',ROOT/'models/residual_learning/quantum_residual_reupload_vqc.py',
             ROOT/'training/trainer.py',ROOT/'configs/config.py',ROOT/'notebooks/Data_Preprocessing.ipynb',
             ROOT/'experiments/residual_learning/extract_residual_dataset.py',ROOT/'main.py',
             ROOT/'training/loss.py',ROOT/'experiments/residual_learning/run_residual_audit.py',
             ROOT/'artifacts/residual_learning/proposed/horizon_15/run_1/manifest.json']
    sources += [ROOT/f'models/{name}.py' for name in ('dcnn','feature_attention','residual_bilstm',
                'temporal_attention','scalar_gated_fusion','mlp_head')]
    settings={"experiment":"end-to-end proposed_rvqc","run_number":args.run_number,"seed":seed,
              "seed_formula":"config.RANDOM_SEED + run_number - 1","initialization":"from scratch; no pretrained weights",
              "backbone_device":str(device),"rvqc_device":"cpu","cpu_threads":args.cpu_threads,
              "configuration":{key:getattr(config,key) for key in ('BATCH_SIZE','NUM_EPOCHS','EARLY_STOPPING_PATIENCE','LEARNING_RATE','WEIGHT_DECAY','GRADIENT_CLIP_VALUE','SCHEDULER_MODE','SCHEDULER_FACTOR','SCHEDULER_PATIENCE','SCHEDULER_MIN_LR','SHUFFLE_TRAIN','NUM_WORKERS')},
              "loss":"MSE on final normalized forecast; no pilot residual bias/scaling",
              "features":FEATURES,"lookback":LOOKBACK,"outputs":OUTPUTS,"stride":config.STRIDE,
              "source_sha256":{str(p.relative_to(ROOT)):fingerprint(p) for p in sources},
              "software":{"python":platform.python_version(),"torch":str(torch.__version__),"pennylane":qml.__version__,"sklearn":sklearn.__version__},
              "hardware":torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
              "smoke_test":args.smoke_test,"benchmark_only":args.benchmark}
    suffix=Path(f'horizon_15/run_{args.run_number}')
    if args.smoke_test or args.benchmark: suffix=Path('_smoke' if args.smoke_test else '_benchmark')/suffix
    destination=TRAINING_ROOT/suffix
    summary_path=destination/('benchmark.json' if args.benchmark else 'training_history.json')
    settings_path=destination/'settings.json'
    if summary_path.exists():
        recorded=json.loads(settings_path.read_text(encoding='utf-8'))
        if any(recorded.get(key)!=value for key,value in settings.items()): raise ValueError("Completed run provenance/configuration conflicts.")
        with csv_path.open("rb") as handle:
            prefix = b"".join(handle.readline() for _ in range(VAL_END + 1))
        if hashlib.sha256(prefix).hexdigest() != recorded['fitting_csv_sha256']:
            raise ValueError("Completed training CSV prefix changed.")
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        if not args.benchmark: summary=summary['completion']
        if summary['settings_sha256']!=fingerprint(settings_path):raise ValueError("Completed settings changed.")
        for base,items in ((destination,summary.get('training_files_sha256',{})),):
            for name,digest in items.items():
                path=base/name
                if not path.is_file() or fingerprint(path)!=digest:raise ValueError(f"Completed artifact changed: {path}")
        logging.info('Completed run verified; no retraining or overwrite: %s',summary_path)
        logging.info('Training complete; test accessed=False. Run models.proposed_rvqc.evaluate separately.')
        return
    existing=[p for base in (destination,) if base.exists() for p in base.rglob('*') if p.is_file()]
    if existing:raise FileExistsError(f"Partial run is not safely resumable; archive/explicitly clean before restarting: {existing}")
    protected=historical_snapshot(); started=time.perf_counter()
    try:
        torch.set_num_threads(args.cpu_threads); _set_seed(seed)
        train,validation,feature_scaler,target_scaler,source=load_fitting_data(csv_path)
        settings.update(source)
        model=ProposedRVQC().to(device)
        probe=next(iter(DataLoader(train,batch_size=8)))
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            checks=gradient_check(model,probe[0].to(device),probe[1].to(device))
        # Probe must not change batch-normalization state of the actual training initialization.
        _set_seed(seed); model=ProposedRVQC().to(device)
        settings.update(feature_scaler=scaler_stats(feature_scaler),target_scaler=scaler_stats(target_scaler),
                        split_rows={'train':[0,TRAIN_END],'validation':[TRAIN_END,VAL_END],'test':[VAL_END,EXPECTED_ROWS]},
                        samples={'train':len(train),'validation':len(validation),'test':EXPECTED_ROWS-VAL_END-26},
                        parameter_counts={'backbone':sum(p.numel() for p in model.backbone.parameters()),'rvqc':model.rvqc.parameter_counts()})
        destination.mkdir(parents=True,exist_ok=True)
        if args.benchmark or args.smoke_test:
            report=benchmark(model,train,validation,device)
            logging.info('Actual model benchmark: %s',json.dumps(report))
            _set_seed(seed); model=ProposedRVQC().to(device)
        else:report=None
        if args.benchmark:
            save_json(settings_path,settings)
            save_json(summary_path,{'settings_sha256':fingerprint(settings_path),'gradient_checks':checks,'benchmark':report,'test_accessed':False})
            logging.info('Benchmark complete; test accessed=False.')
            return
        if args.smoke_test:
            train=WindowDataset(train.features,train.targets,np.arange(2*config.BATCH_SIZE))
            validation=WindowDataset(validation.features,validation.targets,np.arange(config.BATCH_SIZE))
        save_json(settings_path,settings)
        trainer=make_trainer(model,train,validation,device,destination,1 if args.smoke_test else config.NUM_EPOCHS)
        history=trainer.train()
        checkpoint=destination/'checkpoints/best_checkpoint.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        restored=ProposedRVQC().to(device); restored.load_state_dict(state['model_state_dict'],strict=True);restored.eval()
        model.load_state_dict(state['model_state_dict'],strict=True); model.eval()
        with torch.no_grad():torch.testing.assert_close(model(probe[0].to(device)),restored(probe[0].to(device)),rtol=0,atol=0)
        summary={'settings_sha256':fingerprint(settings_path),'run_number':args.run_number,'seed':seed,
                 'selected_epoch':state['epoch']+1,'best_validation_loss':state['best_val_loss'],
                 'gradient_checks':checks,'checkpoint_reload_exact':True,'benchmark':report,
                 'training_files_sha256':{'checkpoints/best_checkpoint.pt':sha256(checkpoint)},
                 'test_accessed':False,
                 'warnings':['Historical interpolation/timestamp gaps preserved. Original test has prior project evaluations.',
                             'Recorded historical checkpoints do not store their seeds; pairing uses project run convention.']}
        summary['runtime_seconds']=time.perf_counter()-started
        history['completion']=summary
        save_json(summary_path,history)
        logging.info('Finished %s (test accessed=%s)',summary_path,summary['test_accessed'])
    finally:
        assert_protected(protected)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-number',type=int,required=True)
    parser.add_argument('--processed-csv',type=Path)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto',help='Backbone device only; RVQC always CPU.')
    parser.add_argument('--cpu-threads',type=int,default=1)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--smoke-test',action='store_true');mode.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if args.cpu_threads<1:parser.error('--cpu-threads must be positive')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    run(args)


if __name__=='__main__':
    main()
