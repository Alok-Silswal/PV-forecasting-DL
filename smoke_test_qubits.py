"""
smoke_test_qubits.py

Computational-feasibility smoke test for the PHN qubit-scaling
experiment (RESEARCH QUESTION: does increasing VQC capacity beyond the
existing 2-qubit `proposed_phn` baseline improve performance?).

This script does NOT train a model and does NOT produce any
predictive-performance claim (RMSE/MAE/R²/nRMSE). It exists ONLY to
answer: "is each candidate qubit count computationally tractable on
the actual GPU (Tesla T4), before committing to an overnight training
run?"

For each qubit count in {3, 4, 5, 6, 7, 8, 9, 10}, this script:

  1. Constructs a fresh `proposed_phn` model
     (get_model("proposed_phn", ...) -> ProposedModel(use_quantum_branch=True))
     with VQC_NUM_QUBITS overridden via config for that run only.
  2. Runs 2 warm-up iterations, then 5 timed forward/backward
     iterations, on a synthetic batch of shape (4, 24, 7) -- matching
     LOOKBACK=24, NUM_FEATURES=7 from configs/config.py.
  3. Measures construction time, parameter counts, projection shape,
     forward/backward/step timing, peak GPU/CPU memory, output shape,
     output/gradient finiteness, and whether a single optimizer step
     succeeds.
  4. Optionally runs a handful of additional optimizer steps
     (PRELIMINARY OPTIMIZATION SIGNAL ONLY -- see section below) to
     check that gradients stay finite and loss does not explode.
  5. Prints a summary table across all qubit counts and an
     order-of-magnitude full-training runtime ESTIMATE based on the
     ACTUAL step/epoch counts in configs/config.py
     (NUM_EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE), scaled from
     this run's measured per-step time -- NOT assumed to scale
     linearly with qubit count.

IMPORTANT: this script uses a FRESH SUBPROCESS per qubit count (where
practical) so that PennyLane device/QNode state, CUDA memory, and
Python-level caching from one qubit count cannot leak into or bias the
measurement for the next. If subprocess isolation is unavailable for
any reason, it falls back to in-process measurement with explicit
CUDA cache clearing and garbage collection between configurations, and
labels the results accordingly.

The 2-qubit baseline is NOT re-measured here -- it already has a
completed 5-run experiment (see the research spec). This script only
benchmarks the NEW candidate qubit counts: 3, 4, 5, 6, 7, 8, 9, 10.

USAGE
-----
    python smoke_test_qubits.py

Optional flags:
    python smoke_test_qubits.py --qubits 3 4 5        # subset only
    python smoke_test_qubits.py --batch-size 4         # default is 4
    python smoke_test_qubits.py --opt-steps 5           # 0 disables the
                                                          # preliminary
                                                          # optimization
                                                          # signal
    python smoke_test_qubits.py --no-subprocess          # force in-process
                                                          # (single Python
                                                          # process) mode
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# NOTE ON PROJECT LAYOUT
# ---------------------------------------------------------------------------
# This script assumes it is placed at the project root (next to
# `configs/`, `models/`, `main.py`, etc.), consistent with the existing
# CLI entry points. It imports `configs.config` and
# `models.model_factory` exactly as the rest of the pipeline does.
# ---------------------------------------------------------------------------

CANDIDATE_QUBITS = [3, 4, 5, 6, 7, 8, 9, 10]

WARMUP_ITERS = 2
TIMED_ITERS = 5


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class QubitBenchmarkResult:
    num_qubits: int
    status: str = "PENDING"  # PASS / FAIL
    failure_reason: Optional[str] = None

    construction_time_s: Optional[float] = None
    total_param_count: Optional[int] = None
    vqc_param_count: Optional[int] = None
    projection_shape: Optional[str] = None

    forward_ms_mean: Optional[float] = None
    forward_ms_std: Optional[float] = None
    backward_ms_mean: Optional[float] = None
    backward_ms_std: Optional[float] = None
    step_ms_mean: Optional[float] = None
    step_ms_std: Optional[float] = None

    peak_gpu_memory_mb: Optional[float] = None
    peak_cpu_memory_mb: Optional[float] = None

    output_shape: Optional[str] = None
    output_finite: Optional[bool] = None
    gradient_finite: Optional[bool] = None
    optimizer_step_succeeded: Optional[bool] = None

    # PRELIMINARY OPTIMIZATION SIGNAL ONLY -- see module docstring.
    # This is NOT a predictive-performance measurement.
    prelim_opt_losses: list = field(default_factory=list)
    prelim_opt_finite_throughout: Optional[bool] = None

    estimated_full_training_hours: Optional[float] = None

    # Estimated per-step time at the REAL training batch size
    # (config.BATCH_SIZE), extrapolated linearly from this run's
    # measured per-sample step time. Declared as a real dataclass
    # field (not a dynamically-attached attribute) so it survives the
    # asdict()/JSON round-trip used for subprocess isolation.
    real_batch_step_ms_estimate: Optional[float] = None


# =============================================================================
# Environment info
# =============================================================================

def print_environment_info() -> None:
    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)

    try:
        import pennylane as qml
        pennylane_version = qml.__version__
    except ImportError:
        pennylane_version = "NOT INSTALLED"

    print(f"PyTorch version   : {torch.__version__}")
    print(f"PennyLane version : {pennylane_version}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name          : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU name          : N/A (no CUDA device)")
    print("=" * 78)
    print()


# =============================================================================
# Peak CPU memory (best-effort; resource module is POSIX-only)
# =============================================================================

def _peak_cpu_memory_mb() -> Optional[float]:
    try:
        import resource
        # ru_maxrss is KB on Linux, bytes on macOS.
        peak_kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return peak_kb_or_bytes / (1024 * 1024)
        return peak_kb_or_bytes / 1024
    except Exception:
        return None


# =============================================================================
# Single-qubit-count benchmark (runs in its own process when invoked via
# --_internal-single-run, or in-process as a fallback)
# =============================================================================

def _run_single_qubit_benchmark(
    num_qubits: int,
    batch_size: int,
    opt_steps: int,
    seed: int,
) -> QubitBenchmarkResult:
    result = QubitBenchmarkResult(num_qubits=num_qubits)

    try:
        # Import here (not at module top) so that each subprocess
        # invocation gets a clean import of configs/models, and so
        # in-process fallback mode can still import successfully even
        # if this script is executed from a slightly different CWD.
        sys.path.insert(0, str(Path(__file__).resolve().parent))

        from configs import config
        from models.model_factory import get_model

        # ---- Override ONLY VQC_NUM_QUBITS for this run. Everything
        # else (VQC_DEPTH, VQC_SIMULATOR, VQC_DIFF_METHOD, dataset,
        # split, optimizer, lr, weight decay, batch size used for real
        # training, etc.) is read from the existing config unchanged,
        # per the non-negotiable controls in the research spec.
        config.VQC_NUM_QUBITS = num_qubits

        torch.manual_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Model construction ----
        t0 = time.perf_counter()
        model = get_model("proposed_phn")  # -> ProposedModel(use_quantum_branch=True)
        model = model.to(device)
        model.train()
        construction_time = time.perf_counter() - t0
        result.construction_time_s = construction_time

        # Sanity: confirm we actually built the PHN model with a VQC
        # branch, not the separate `phn` model or the plain `proposed`
        # model without a quantum branch.
        if not hasattr(model, "vqc_branch"):
            raise RuntimeError(
                "get_model('proposed_phn') did not produce a model with "
                "a 'vqc_branch' attribute -- refusing to benchmark the "
                "wrong model."
            )
        if model.vqc_branch.num_qubits != num_qubits:
            raise RuntimeError(
                f"model.vqc_branch.num_qubits={model.vqc_branch.num_qubits} "
                f"does not match requested num_qubits={num_qubits} -- "
                "config override did not take effect as expected."
            )

        result.total_param_count = sum(p.numel() for p in model.parameters())
        result.vqc_param_count = sum(
            p.numel() for p in model.vqc_branch.parameters()
        )
        result.projection_shape = str(
            tuple(model.vqc_branch.projection.weight.shape)
        )

        # ---- Synthetic batch: (batch, LOOKBACK, NUM_FEATURES) ----
        x = torch.randn(
            batch_size, config.LOOKBACK, config.NUM_FEATURES, device=device
        )
        y_true = torch.randn(
            batch_size,
            config.HORIZON_TO_OUTPUT_DIM[config.ACTIVE_HORIZON],
            device=device,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        loss_fn = torch.nn.MSELoss()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # ---- Warm-up ----
        for _ in range(WARMUP_ITERS):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y_true)
            loss.backward()
            optimizer.step()
        _sync()

        # ---- Timed forward-only ----
        forward_times = []
        with torch.no_grad():
            for _ in range(TIMED_ITERS):
                _sync()
                t0 = time.perf_counter()
                _ = model(x)
                _sync()
                forward_times.append((time.perf_counter() - t0) * 1000.0)

        # ---- Timed forward+backward, and full step ----
        backward_times = []
        step_times = []
        last_out = None
        last_loss = None
        for _ in range(TIMED_ITERS):
            optimizer.zero_grad(set_to_none=True)

            _sync()
            t_step0 = time.perf_counter()

            out = model(x)
            loss = loss_fn(out, y_true)

            _sync()
            t_bwd0 = time.perf_counter()
            loss.backward()
            _sync()
            backward_times.append((time.perf_counter() - t_bwd0) * 1000.0)

            optimizer.step()
            _sync()
            step_times.append((time.perf_counter() - t_step0) * 1000.0)

            last_out = out
            last_loss = loss

        def _mean_std(vals):
            t = torch.tensor(vals)
            return t.mean().item(), t.std(unbiased=False).item()

        result.forward_ms_mean, result.forward_ms_std = _mean_std(forward_times)
        result.backward_ms_mean, result.backward_ms_std = _mean_std(backward_times)
        result.step_ms_mean, result.step_ms_std = _mean_std(step_times)

        result.output_shape = str(tuple(last_out.shape))
        result.output_finite = bool(torch.isfinite(last_out).all().item())

        grad_finite = True
        any_grad = False
        for p in model.parameters():
            if p.grad is not None:
                any_grad = True
                if not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break
        result.gradient_finite = grad_finite and any_grad

        result.optimizer_step_succeeded = True  # we already ran .step() above without raising

        if torch.cuda.is_available():
            result.peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            )
        result.peak_cpu_memory_mb = _peak_cpu_memory_mb()

        # -------------------------------------------------------------
        # PRELIMINARY OPTIMIZATION SIGNAL ONLY.
        #
        # This runs a SMALL number of additional optimizer steps on the
        # SAME synthetic random batch. It exists ONLY to check that:
        #   - gradients remain finite over several steps,
        #   - loss does not explode / diverge to NaN or Inf,
        #   - the larger circuit is numerically trainable at all,
        #   - runtime does not suddenly blow up step-to-step.
        #
        # It is explicitly NOT a predictive-performance measurement.
        # A few steps on one synthetic random batch says NOTHING about
        # final RMSE/MAE/R²/nRMSE on the real dataset, and this script
        # must never be read as implying otherwise.
        # -------------------------------------------------------------
        if opt_steps > 0:
            prelim_losses = []
            prelim_finite = True
            for _ in range(opt_steps):
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = loss_fn(out, y_true)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                prelim_losses.append(loss_val)
                if not (torch.isfinite(out).all() and torch.isfinite(loss)):
                    prelim_finite = False
            result.prelim_opt_losses = prelim_losses
            result.prelim_opt_finite_throughout = prelim_finite

        # -------------------------------------------------------------
        # Rough full-training runtime ESTIMATE.
        #
        # Uses the ACTUAL step/epoch counts from configs/config.py:
        #   - config.NUM_EPOCHS (upper bound on epochs; early stopping
        #     may end training sooner -- this is a ceiling, not a
        #     prediction)
        #   - config.BATCH_SIZE (the REAL training batch size, 256 by
        #     default) -- NOTE this differs from this smoke test's
        #     small batch_size (default 4), used here only for
        #     feasibility probing given 10-qubit statevector cost.
        #   - config.NUM_RUNS (5 repeated runs per the existing
        #     multi-run protocol)
        #
        # This is a coarse order-of-magnitude estimate only:
        #   - it assumes a fixed, unknown number of training examples
        #     (train-set size is not available to this script without
        #     loading the actual dataset artifacts), so it reports
        #     PER-EPOCH time assuming (dataset_size / BATCH_SIZE) steps
        #     and leaves total-epoch scaling to the reader, rather than
        #     fabricating a training-set size;
        #   - step time at BATCH_SIZE=256 is NOT assumed to equal
        #     step time at this smoke test's batch_size — VQC
        #     execution is CPU-bound and per-sample serial in several
        #     PennyLane execution paths, so this script reports the
        #     measured per-sample step time (step_ms / batch_size) and
        #     extrapolates LINEARLY IN BATCH SIZE ONLY as a labeled
        #     approximation, while explicitly NOT assuming the
        #     per-qubit-count scaling itself is linear (that is the
        #     entire quantity being measured).
        # -------------------------------------------------------------
        per_sample_step_ms = result.step_ms_mean / batch_size
        result.real_batch_step_ms_estimate = per_sample_step_ms * config.BATCH_SIZE
        # steps-per-epoch is intentionally left uncomputed here: it
        # depends on the real training-set size, which this script does
        # not load (see module docstring / print_runtime_estimates).
        result.estimated_full_training_hours = None

        result.status = "PASS"

    except Exception as exc:  # noqa: BLE001 - smoke test must capture and report, not crash the sweep
        result.status = "FAIL"
        result.failure_reason = f"{type(exc).__name__}: {exc}"

    return result


# =============================================================================
# Subprocess wrapper (isolation between qubit counts)
# =============================================================================

def _run_in_subprocess(
    num_qubits: int, batch_size: int, opt_steps: int, seed: int
) -> QubitBenchmarkResult:
    """
    Runs `_run_single_qubit_benchmark` in a fresh `python` subprocess
    by re-invoking this same script with an internal flag, and reads
    back a JSON-serialized result over stdout. Falls back to raising
    (caller catches and switches to in-process mode) if subprocess
    execution fails outright (e.g. sandboxing prevents it).
    """
    script_path = str(Path(__file__).resolve())
    cmd = [
        sys.executable,
        script_path,
        "--_internal-single-run",
        "--_internal-num-qubits", str(num_qubits),
        "--batch-size", str(batch_size),
        "--opt-steps", str(opt_steps),
        "--_internal-seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        result = QubitBenchmarkResult(num_qubits=num_qubits, status="FAIL")
        result.failure_reason = (
            f"Subprocess exited with code {proc.returncode}. "
            f"stderr(last 2000 chars): {proc.stderr[-2000:]}"
        )
        return result

    # The subprocess prints exactly one JSON line prefixed with
    # "RESULT_JSON:" -- extract and parse it.
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            payload = json.loads(line[len("RESULT_JSON:"):])
            return QubitBenchmarkResult(**payload)

    result = QubitBenchmarkResult(num_qubits=num_qubits, status="FAIL")
    result.failure_reason = (
        "Subprocess completed but no RESULT_JSON line was found in "
        f"stdout. stdout(last 1000 chars): {proc.stdout[-1000:]}"
    )
    return result


# =============================================================================
# Reporting
# =============================================================================

def _fmt(val, suffix="", nd=2):
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        return f"{val:.{nd}f}{suffix}"
    return f"{val}{suffix}"


def print_timing_table(results: list[QubitBenchmarkResult]) -> None:
    print()
    print("TIMING / PARAMETER TABLE")
    print(
        f"{'Qubits':>6} | {'Params':>10} | {'VQC Params':>10} | "
        f"{'Forward ms':>12} | {'Backward ms':>12} | {'Step ms':>10}"
    )
    print("-" * 78)
    for r in results:
        if r.status != "PASS":
            print(f"{r.num_qubits:>6} | {'FAILED':>10} | {'-':>10} | "
                  f"{'-':>12} | {'-':>12} | {'-':>10}")
            continue
        fwd = f"{r.forward_ms_mean:.2f}±{r.forward_ms_std:.2f}"
        bwd = f"{r.backward_ms_mean:.2f}±{r.backward_ms_std:.2f}"
        step = f"{r.step_ms_mean:.2f}"
        print(
            f"{r.num_qubits:>6} | {r.total_param_count:>10} | "
            f"{r.vqc_param_count:>10} | {fwd:>12} | {bwd:>12} | {step:>10}"
        )


def print_status_table(results: list[QubitBenchmarkResult]) -> None:
    print()
    print("STATUS TABLE")
    print(
        f"{'Qubits':>6} | {'GPU Mem (MB)':>13} | {'Output Shape':>14} | "
        f"{'Grads Finite':>12} | {'Status':>8}"
    )
    print("-" * 78)
    for r in results:
        gpu_mem = _fmt(r.peak_gpu_memory_mb, nd=1)
        out_shape = r.output_shape or "-"
        grads = _fmt(r.gradient_finite)
        print(
            f"{r.num_qubits:>6} | {gpu_mem:>13} | {out_shape:>14} | "
            f"{grads:>12} | {r.status:>8}"
        )
        if r.status == "FAIL":
            print(f"         -> reason: {r.failure_reason}")


def print_runtime_estimates(
    results: list[QubitBenchmarkResult], config
) -> None:
    print()
    print("APPROXIMATE FULL-TRAINING RUNTIME (per run, ceiling estimate)")
    print(
        "NOTE: this is a coarse extrapolation from a tiny synthetic batch. "
        "It assumes linear scaling in BATCH SIZE ONLY (per-sample step "
        "time x real BATCH_SIZE), NOT linear scaling in qubit count -- "
        "qubit-count-vs-runtime scaling is exactly what this table is "
        "meant to reveal, not assume. Uses the pipeline's ACTUAL "
        f"NUM_EPOCHS={config.NUM_EPOCHS} as an upper bound (early "
        f"stopping, patience={config.EARLY_STOPPING_PATIENCE}, may end "
        f"training sooner) and NUM_RUNS={config.NUM_RUNS} repeated runs. "
        "Steps-per-epoch is NOT computed here since this script does not "
        "load the real dataset artifacts; multiply the printed "
        "per-epoch-per-step estimate by your actual (train_set_size / "
        f"BATCH_SIZE={config.BATCH_SIZE}) to get a per-epoch time."
    )
    print()
    print(f"{'Qubits':>6} | {'Est. ms/step @ real batch size':>32} | {'Status':>8}")
    print("-" * 78)
    for r in results:
        if r.status != "PASS":
            print(f"{r.num_qubits:>6} | {'-':>32} | {r.status:>8}")
            continue
        est = r.real_batch_step_ms_estimate
        est_str = f"{est:.1f} ms" if est is not None else "N/A"
        print(f"{r.num_qubits:>6} | {est_str:>32} | {r.status:>8}")


def print_preliminary_optimization_signal(
    results: list[QubitBenchmarkResult],
) -> None:
    any_ran = any(r.prelim_opt_losses for r in results)
    if not any_ran:
        return
    print()
    print("=" * 78)
    print("PRELIMINARY OPTIMIZATION SIGNAL ONLY -- NOT PREDICTIVE PERFORMANCE")
    print("=" * 78)
    print(
        "The values below come from a handful of optimizer steps on ONE "
        "small synthetic random batch. They indicate ONLY whether "
        "gradients stayed finite and whether loss moved in a stable "
        "direction on that batch. They must NOT be interpreted as, or "
        "reported as, an estimate of final RMSE / MAE / R² / nRMSE. Only "
        "a full training run on the real dataset can produce those."
    )
    print()
    for r in results:
        if not r.prelim_opt_losses:
            continue
        losses_str = ", ".join(f"{v:.6f}" for v in r.prelim_opt_losses)
        print(
            f"  num_qubits={r.num_qubits}: losses=[{losses_str}] "
            f"finite_throughout={r.prelim_opt_finite_throughout}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computational-feasibility smoke test for PHN VQC "
        "qubit-count scaling (candidates 3-10, against the existing "
        "2-qubit proposed_phn baseline)."
    )
    parser.add_argument(
        "--qubits", type=int, nargs="+", default=CANDIDATE_QUBITS,
        help="Qubit counts to benchmark (default: 3 4 5 6 7 8 9 10).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Synthetic batch size for the smoke test only (default: 4). "
        "This is NOT config.BATCH_SIZE (used for real training); it is "
        "deliberately small because the 10-qubit statevector may be "
        "expensive.",
    )
    parser.add_argument(
        "--opt-steps", type=int, default=5,
        help="Number of PRELIMINARY OPTIMIZATION SIGNAL steps to run "
        "after timing (default: 5; pass 0 to disable). This is NOT a "
        "predictive-performance measurement -- see module docstring.",
    )
    parser.add_argument(
        "--no-subprocess", action="store_true",
        help="Force in-process (single Python process) benchmarking "
        "instead of one fresh subprocess per qubit count. Subprocess "
        "isolation is preferred (avoids CUDA-memory/QNode-state leakage "
        "between qubit counts) but this flag allows a fallback.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for all benchmarked configurations (default: "
        "configs.config.RANDOM_SEED, so the smoke test uses the same "
        "seed protocol as the rest of the pipeline). The SAME seed is "
        "used for every qubit count so any behavior difference between "
        "them is not an artifact of a different starting point.",
    )

    # Internal flags used to re-invoke this script as a subprocess for a
    # single qubit count. Not intended for direct use.
    parser.add_argument("--_internal-single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_internal-num-qubits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_internal-seed", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ---- Internal single-run subprocess entry point ----
    if args._internal_single_run:
        result = _run_single_qubit_benchmark(
            num_qubits=args._internal_num_qubits,
            batch_size=args.batch_size,
            opt_steps=args.opt_steps,
            seed=args._internal_seed,
        )
        payload = asdict(result)
        print("RESULT_JSON:" + json.dumps(payload))
        return

    # ---- Normal (top-level) entry point ----
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from configs import config as project_config

    seed = args.seed if args.seed is not None else project_config.RANDOM_SEED

    print_environment_info()

    print(f"Benchmarking qubit counts: {args.qubits}")
    print(f"Synthetic batch size      : {args.batch_size}")
    print(f"Warm-up iterations        : {WARMUP_ITERS}")
    print(f"Timed iterations          : {TIMED_ITERS}")
    print(f"Preliminary opt steps     : {args.opt_steps}")
    print(f"Random seed (all configs) : {seed}")
    print(f"Isolation mode            : "
          f"{'in-process (forced)' if args.no_subprocess else 'subprocess per qubit count'}")
    print()

    results: list[QubitBenchmarkResult] = []

    for nq in args.qubits:
        print(f"--- num_qubits={nq} " + "-" * 40)
        if args.no_subprocess:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = _run_single_qubit_benchmark(
                num_qubits=nq,
                batch_size=args.batch_size,
                opt_steps=args.opt_steps,
                seed=seed,
            )
        else:
            try:
                result = _run_in_subprocess(
                    num_qubits=nq,
                    batch_size=args.batch_size,
                    opt_steps=args.opt_steps,
                    seed=seed,
                )
            except Exception as exc:  # subprocess launch itself failed
                print(
                    f"  Subprocess launch failed ({exc}); "
                    "falling back to in-process for this qubit count."
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = _run_single_qubit_benchmark(
                    num_qubits=nq,
                    batch_size=args.batch_size,
                    opt_steps=args.opt_steps,
                    seed=seed,
                )

        results.append(result)

        if result.status == "PASS":
            print(
                f"  PASS  construction={result.construction_time_s:.3f}s  "
                f"params={result.total_param_count}  "
                f"vqc_params={result.vqc_param_count}  "
                f"forward={result.forward_ms_mean:.2f}ms  "
                f"backward={result.backward_ms_mean:.2f}ms  "
                f"step={result.step_ms_mean:.2f}ms"
            )
        else:
            print(f"  FAIL  reason={result.failure_reason}")
        print()

    print_timing_table(results)
    print_status_table(results)
    print_runtime_estimates(results, project_config)
    print_preliminary_optimization_signal(results)

    print()
    print("=" * 78)
    print("SMOKE TEST COMPLETE. No training was started.")
    print("Review PASS/FAIL, memory, and timing above before deciding")
    print("which qubit configuration(s), if any, are worth training")
    print("overnight on the real dataset.")
    print("=" * 78)


if __name__ == "__main__":
    main()