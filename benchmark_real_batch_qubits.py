"""
benchmark_real_batch_qubits.py

Apples-to-apples computational-feasibility benchmark for proposed_phn
at the ACTUAL TRAINING BATCH SIZE (256), across qubit counts
{2 (optional), 5, 10}, holding VQC_DEPTH=2 fixed.

WHY THIS SCRIPT EXISTS (do not skip this section)
--------------------------------------------------
An earlier smoke test (smoke_test_qubits.py) measured step time at a
tiny synthetic batch size (4) and extrapolated LINEARLY to the real
training batch size (256) to produce an order-of-magnitude estimate.
That extrapolation assumed per-sample cost is constant regardless of
batch size. It is NOT reliable here, for a specific, inspected reason:

    models/vqc_branch.py's VQCBranch executes its QNode
    (`simulator="default.qubit"`, `diff_method="backprop"`) entirely
    on CPU. default.qubit has NO CUDA execution path. The rest of the
    classical backbone (DCNN / Residual BiLSTM / Scalar Gated Fusion /
    MLPHead) runs on GPU. Every single forward call therefore crosses
    a GPU->CPU device boundary (for `angles`, right before the QNode
    call) and a CPU->GPU boundary (for the QNode's output, right
    after), both autograd-tracked `.to()` calls. VQCBranch.weights
    (the variational parameters) is permanently CPU-pinned via a
    `_apply()` override and never participates in these per-call
    transfers, but the per-batch `angles` tensor and the QNode's
    output do, on every forward call.

    This means the VQC portion of the step is CPU-bound, single
    -threaded (per PennyLane's default.qubit + backprop execution
    path), and runs synchronously relative to the GPU work around it.
    Whether its cost scales linearly, super-linearly, or is dominated
    by fixed per-call overhead as batch size grows from 4 to 256 is
    exactly the open question this script exists to answer -- it must
    NOT be assumed away by extrapolation, and it must NOT be measured
    with CUDA-event timing alone, since CUDA events cannot see
    CPU-side wall-clock cost. This script times with `time.perf_counter()`
    wall-clock around the full step, in addition to synchronizing CUDA
    at the correct points, precisely because of this CPU/GPU split.

WHAT THIS SCRIPT DOES
----------------------
For each qubit count in {5, 10} (and optionally 2), using the REAL
`get_model("proposed_phn_Nq")` factory path (models/model_factory.py,
unmodified) and the REAL VQCBranch/ProposedModel implementation
(unmodified), this script:

  1. Constructs a fresh model via get_model(), exactly as main.py does,
     with config.VQC_NUM_QUBITS overridden ONLY for the duration of
     that run's subprocess (config.VQC_DEPTH is NOT touched; it stays
     at whatever configs/config.py already has it set to -- 2, per the
     current baseline -- and is only read, never written, by this
     script).
  2. Moves the model to config's selected device exactly as main.py's
     `_select_device()` + `model.to(device)` does (XLA -> CUDA -> CPU
     fallback order), which is what causes the classical backbone to
     land on GPU (if available) while VQCBranch.weights stays pinned
     to CPU via its own `_apply()` override -- this script does not
     reimplement or alter that device-pinning logic in any way, it
     only constructs the model and lets the model's own `_apply`
     override do what it already does.
  3. Runs several warm-up iterations, then several TIMED iterations,
     on a synthetic batch of shape (256, 24, 7) -- BATCH_SIZE=256,
     LOOKBACK=24, NUM_FEATURES=7, matching configs/config.py and the
     real training input shape exactly.
  4. Measures, separately: construction time, forward-pass time,
     backward-pass time, optimizer-step time, complete
     forward+loss+backward+step time, iterations/sec, peak GPU memory,
     best-effort peak CPU memory, output shape, gradient finiteness,
     and whether a full optimizer step succeeds.
  5. Times BOTH with CUDA-synchronized wall-clock (`torch.cuda.synchronize()`
     immediately before/after each timed region, when CUDA is
     available) AND with plain wall-clock throughout, since the VQC's
     CPU-bound region is invisible to CUDA events alone.
  6. Prints a compact comparison table plus the 10q/5q slowdown factor
     computed from measured complete-step time (not estimated).

WHAT THIS SCRIPT DOES NOT DO
------------------------------
  - Does NOT load the real dataset (synthetic random input only).
  - Does NOT train a model or produce any predictive-performance claim.
  - Does NOT modify config.VQC_DEPTH (read-only; whatever is currently
    in configs/config.py is what every qubit count in this sweep uses).
  - Does NOT modify main.py, trainer.py, proposed_model.py,
    vqc_branch.py, configs/config.py, model_factory.py, or any
    training/evaluation code, or touch any existing experiment/
    evaluation artifacts.
  - Does NOT attempt to isolate "classical-only" vs "full-model" timing.
    Cleanly isolating the VQC's contribution would require either (a)
    constructing a second, VQC-less variant of the forward pass, which
    is not something VQCBranch/ProposedModel expose as a supported
    code path without invasive changes, or (b) hooking submodule
    forward calls in a way that could itself perturb the very CPU/GPU
    transfer timing this script exists to measure accurately. Per the
    request, this optional isolation is skipped rather than
    approximated. What this script CAN and DOES do instead: reports
    forward and backward time separately, so the two dominant
    CPU-bound crossings (GPU->CPU before the QNode, CPU->GPU after)
    are at least visible in aggregate, even though they are not
    separated from the classical backbone's own GPU time within
    forward()/backward().
  - Does NOT draw a feasibility conclusion for you. It reports the
    measured numbers and the 10q/5q ratio; interpretation against the
    previously-observed ~18-21 it/s (5q) vs ~2.73 it/s (10q) training
    numbers is left to the reader, per the request.

SUBPROCESS ISOLATION
----------------------
Each qubit count runs in a fresh `python` subprocess by default (same
mechanism as smoke_test_qubits.py / smoke_test_depth.py), so PennyLane
device/QNode state, CUDA memory, and any process-level caching cannot
leak between qubit counts. Falls back to in-process measurement (with
explicit CUDA cache clearing and GC between configs) if subprocess
execution is unavailable, and labels results accordingly. If one
qubit count's subprocess fails, the script continues to the next.

USAGE
-----
    python benchmark_real_batch_qubits.py
    python benchmark_real_batch_qubits.py --qubits 5 10
    python benchmark_real_batch_qubits.py --qubits 2 5 10
    python benchmark_real_batch_qubits.py --batch-size 256   # default
    python benchmark_real_batch_qubits.py --warmup 3 --timed-iters 10
    python benchmark_real_batch_qubits.py --no-subprocess
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
# `configs/`, `models/`, `main.py`, etc.), exactly like main.py,
# smoke_test_qubits.py, and smoke_test_depth.py. It imports
# `configs.config` and `models.model_factory` exactly as the rest of
# the pipeline does, and constructs models ONLY via
# `models.model_factory.get_model("proposed_phn_{n}q")`, the same
# entry point main.py uses -- never a hand-built model.
# ---------------------------------------------------------------------------

DEFAULT_QUBITS = [5, 10]
REAL_BATCH_SIZE = 256

DEFAULT_WARMUP_ITERS = 3
DEFAULT_TIMED_ITERS = 10

# Maps a benchmarked qubit count to the model-name alias already
# registered in models/model_factory.py. This script does not invent
# new aliases; if a requested qubit count has no corresponding alias
# in the factory, the run fails loudly with that exact reason rather
# than silently falling back to constructing ProposedModel directly.
_QUBITS_TO_MODEL_ALIAS = {
    2: "proposed_phn",       # baseline alias; also matches proposed_phn_2q semantics
    5: "proposed_phn_5q",
    10: "proposed_phn_10q",
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RealBatchBenchmarkResult:
    num_qubits: int
    model_alias: str
    batch_size: int
    status: str = "PENDING"  # PASS / FAIL
    failure_reason: Optional[str] = None

    device: Optional[str] = None
    vqc_depth_used: Optional[int] = None

    construction_time_s: Optional[float] = None
    total_param_count: Optional[int] = None
    vqc_param_count: Optional[int] = None

    # Wall-clock (time.perf_counter), CUDA-synchronized at region
    # boundaries when CUDA is available. This is the PRIMARY timing
    # source, since VQCBranch's QNode is CPU-bound and invisible to
    # CUDA events alone (see module docstring).
    forward_ms_mean: Optional[float] = None
    forward_ms_std: Optional[float] = None
    backward_ms_mean: Optional[float] = None
    backward_ms_std: Optional[float] = None
    optimizer_step_ms_mean: Optional[float] = None
    optimizer_step_ms_std: Optional[float] = None
    full_step_ms_mean: Optional[float] = None
    full_step_ms_std: Optional[float] = None

    iterations_per_sec: Optional[float] = None

    peak_gpu_memory_mb: Optional[float] = None
    peak_cpu_memory_mb: Optional[float] = None

    output_shape: Optional[str] = None
    output_finite: Optional[bool] = None
    gradient_finite: Optional[bool] = None
    optimizer_step_succeeded: Optional[bool] = None

    # Per-timed-iteration raw full-step wall-clock samples (ms), kept
    # for transparency / manual std-dev sanity-checking, and so a
    # reader can see run-to-run jitter directly rather than only the
    # summary mean/std.
    full_step_ms_samples: list = field(default_factory=list)


# =============================================================================
# Environment info
# =============================================================================

def print_environment_info() -> None:
    print("=" * 88)
    print("ENVIRONMENT")
    print("=" * 88)

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
    try:
        import os
        print(f"CPU count         : {os.cpu_count()}")
    except Exception:
        pass
    print("=" * 88)
    print()


# =============================================================================
# Peak CPU memory (best-effort; resource module is POSIX-only)
# =============================================================================

def _peak_cpu_memory_mb() -> Optional[float]:
    try:
        import resource
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

def _run_single_benchmark(
    num_qubits: int,
    batch_size: int,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> RealBatchBenchmarkResult:
    model_alias = _QUBITS_TO_MODEL_ALIAS.get(num_qubits)
    result = RealBatchBenchmarkResult(
        num_qubits=num_qubits,
        model_alias=model_alias or f"<no alias registered for {num_qubits} qubits>",
        batch_size=batch_size,
    )

    if model_alias is None:
        result.status = "FAIL"
        result.failure_reason = (
            f"No model_factory.py alias is registered for num_qubits="
            f"{num_qubits} in this benchmark script's "
            f"_QUBITS_TO_MODEL_ALIAS mapping ({sorted(_QUBITS_TO_MODEL_ALIAS)} "
            "supported). Refusing to hand-construct ProposedModel "
            "directly, since that would bypass the real "
            "models/model_factory.py routing this benchmark is meant "
            "to exercise."
        )
        return result

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

        from configs import config
        from models.model_factory import get_model

        # ---- Override ONLY VQC_NUM_QUBITS for this run, to select
        # the qubit count under test. VQC_DEPTH is intentionally NEVER
        # written here -- it is read from whatever configs/config.py
        # already has it set to, per the request ("Do not test depth
        # in this benchmark" / "Do NOT modify configs/config.py"; this
        # script mutates the in-memory config object for its own
        # subprocess only, exactly as smoke_test_qubits.py and
        # smoke_test_depth.py already do, and never writes the file).
        config.VQC_NUM_QUBITS = num_qubits
        result.vqc_depth_used = config.VQC_DEPTH

        torch.manual_seed(seed)

        # ---- Device selection: mirrors main.py's _select_device()
        # exactly (XLA -> CUDA -> CPU), since main.py's model.to(device)
        # is the real behavior this benchmark must reproduce. This
        # script does not alter or special-case that logic in any way;
        # VQCBranch's own _apply() override is what keeps its
        # `weights` CPU-pinned regardless of what device() resolves to
        # here.
        device = None
        try:
            import importlib
            xm = importlib.import_module("torch_xla.core.xla_model")
            device = xm.xla_device()
        except Exception:
            pass
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result.device = str(device)

        # ---- Model construction via the REAL factory path ----
        t0 = time.perf_counter()
        model = get_model(model_alias)  # -> ProposedModel(use_quantum_branch=True)
        model = model.to(device)
        model.train()
        construction_time = time.perf_counter() - t0
        result.construction_time_s = construction_time

        if not hasattr(model, "vqc_branch"):
            raise RuntimeError(
                f"get_model('{model_alias}') did not produce a model "
                "with a 'vqc_branch' attribute -- refusing to benchmark "
                "the wrong model."
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

        # ---- Synthetic batch: (256, LOOKBACK, NUM_FEATURES), matching
        # the REAL training batch size and input shape exactly. ----
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

        cuda_active = torch.cuda.is_available() and str(device).startswith("cuda")
        if cuda_active:
            torch.cuda.reset_peak_memory_stats(device)

        def _sync():
            # CUDA-synchronize when relevant, so GPU work is actually
            # complete before a wall-clock boundary is read. This is
            # necessary but NOT sufficient on its own: it makes GPU
            # timing accurate, but the VQC's CPU-bound QNode work is
            # captured correctly regardless, because it is synchronous
            # Python-thread work that perf_counter() already sees in
            # real time (there is no async CPU dispatch to wait on).
            if cuda_active:
                torch.cuda.synchronize()

        # ---- Warm-up ----
        for _ in range(warmup_iters):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y_true)
            loss.backward()
            optimizer.step()
        _sync()

        # ---- Timed iterations: forward, backward, optimizer-step,
        # and full-step, each measured with perf_counter() wall-clock,
        # synchronizing CUDA at the correct boundary each time. ----
        forward_times = []
        backward_times = []
        opt_step_times = []
        full_step_times = []

        last_out = None

        for _ in range(timed_iters):
            optimizer.zero_grad(set_to_none=True)

            _sync()
            t_full0 = time.perf_counter()

            # ---- forward ----
            t_fwd0 = time.perf_counter()
            out = model(x)
            _sync()
            forward_times.append((time.perf_counter() - t_fwd0) * 1000.0)

            loss = loss_fn(out, y_true)

            # ---- backward ----
            t_bwd0 = time.perf_counter()
            loss.backward()
            _sync()
            backward_times.append((time.perf_counter() - t_bwd0) * 1000.0)

            # ---- optimizer step ----
            t_opt0 = time.perf_counter()
            optimizer.step()
            _sync()
            opt_step_times.append((time.perf_counter() - t_opt0) * 1000.0)

            full_step_times.append((time.perf_counter() - t_full0) * 1000.0)

            last_out = out

        def _mean_std(vals):
            t = torch.tensor(vals)
            return t.mean().item(), t.std(unbiased=False).item()

        result.forward_ms_mean, result.forward_ms_std = _mean_std(forward_times)
        result.backward_ms_mean, result.backward_ms_std = _mean_std(backward_times)
        result.optimizer_step_ms_mean, result.optimizer_step_ms_std = _mean_std(opt_step_times)
        result.full_step_ms_mean, result.full_step_ms_std = _mean_std(full_step_times)
        result.full_step_ms_samples = full_step_times

        result.iterations_per_sec = (
            1000.0 / result.full_step_ms_mean if result.full_step_ms_mean else None
        )

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

        result.optimizer_step_succeeded = True

        if cuda_active:
            result.peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            )
        result.peak_cpu_memory_mb = _peak_cpu_memory_mb()

        result.status = "PASS"

    except Exception as exc:  # noqa: BLE001 - benchmark must capture and report, not crash the sweep
        result.status = "FAIL"
        result.failure_reason = f"{type(exc).__name__}: {exc}"

    return result


# =============================================================================
# Subprocess wrapper (isolation between qubit counts)
# =============================================================================

def _run_in_subprocess(
    num_qubits: int,
    batch_size: int,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
) -> RealBatchBenchmarkResult:
    script_path = str(Path(__file__).resolve())
    cmd = [
        sys.executable,
        script_path,
        "--_internal-single-run",
        "--_internal-num-qubits", str(num_qubits),
        "--batch-size", str(batch_size),
        "--warmup", str(warmup_iters),
        "--timed-iters", str(timed_iters),
        "--_internal-seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        model_alias = _QUBITS_TO_MODEL_ALIAS.get(num_qubits, "<unregistered>")
        result = RealBatchBenchmarkResult(
            num_qubits=num_qubits,
            model_alias=model_alias,
            batch_size=batch_size,
            status="FAIL",
        )
        result.failure_reason = (
            f"Subprocess exited with code {proc.returncode}. "
            f"stderr(last 2000 chars): {proc.stderr[-2000:]}"
        )
        return result

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            payload = json.loads(line[len("RESULT_JSON:"):])
            return RealBatchBenchmarkResult(**payload)

    model_alias = _QUBITS_TO_MODEL_ALIAS.get(num_qubits, "<unregistered>")
    result = RealBatchBenchmarkResult(
        num_qubits=num_qubits,
        model_alias=model_alias,
        batch_size=batch_size,
        status="FAIL",
    )
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


def print_comparison_table(results: list[RealBatchBenchmarkResult]) -> None:
    print()
    print("REAL-BATCH-SIZE (256) BENCHMARK TABLE")
    header = (
        f"{'Qubits':>6} | {'Batch':>5} | {'Forward ms':>14} | "
        f"{'Backward ms':>14} | {'Step ms':>12} | {'Iter/s':>8} | "
        f"{'GPU MB':>8} | {'Status':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if r.status != "PASS":
            print(
                f"{r.num_qubits:>6} | {r.batch_size:>5} | {'-':>14} | "
                f"{'-':>14} | {'-':>12} | {'-':>8} | {'-':>8} | {'FAIL':>7}"
            )
            continue
        fwd = f"{r.forward_ms_mean:.2f}±{r.forward_ms_std:.2f}"
        bwd = f"{r.backward_ms_mean:.2f}±{r.backward_ms_std:.2f}"
        step = f"{r.full_step_ms_mean:.2f}±{r.full_step_ms_std:.2f}"
        iters = f"{r.iterations_per_sec:.2f}" if r.iterations_per_sec else "N/A"
        gpu_mem = _fmt(r.peak_gpu_memory_mb, nd=1)
        print(
            f"{r.num_qubits:>6} | {r.batch_size:>5} | {fwd:>14} | "
            f"{bwd:>14} | {step:>12} | {iters:>8} | {gpu_mem:>8} | {'PASS':>7}"
        )


def print_status_and_detail_table(results: list[RealBatchBenchmarkResult]) -> None:
    print()
    print("STATUS / DETAIL TABLE")
    print(
        f"{'Qubits':>6} | {'Alias':>18} | {'Device':>8} | {'Depth':>5} | "
        f"{'Params':>9} | {'VQC Params':>10} | {'Grads Finite':>12} | {'Opt Step ms':>12}"
    )
    print("-" * 100)
    for r in results:
        if r.status != "PASS":
            print(f"{r.num_qubits:>6} | {r.model_alias:>18} | FAILED -> {r.failure_reason}")
            continue
        opt_step = f"{r.optimizer_step_ms_mean:.2f}±{r.optimizer_step_ms_std:.2f}"
        print(
            f"{r.num_qubits:>6} | {r.model_alias:>18} | {str(r.device):>8} | "
            f"{_fmt(r.vqc_depth_used):>5} | {_fmt(r.total_param_count):>9} | "
            f"{_fmt(r.vqc_param_count):>10} | {_fmt(r.gradient_finite):>12} | {opt_step:>12}"
        )


def print_slowdown_analysis(results: list[RealBatchBenchmarkResult]) -> None:
    by_qubits = {r.num_qubits: r for r in results if r.status == "PASS"}

    print()
    print("=" * 88)
    print("MEASURED SLOWDOWN (from complete-step wall-clock time, NOT estimated)")
    print("=" * 88)

    if 5 not in by_qubits or 10 not in by_qubits:
        print(
            "Cannot compute the 10q/5q slowdown factor: both the 5-qubit "
            "and 10-qubit runs must have PASSed. Current PASS set: "
            f"{sorted(by_qubits.keys())}."
        )
        return

    r5 = by_qubits[5]
    r10 = by_qubits[10]

    slowdown = r10.full_step_ms_mean / r5.full_step_ms_mean
    it_s_5 = r5.iterations_per_sec
    it_s_10 = r10.iterations_per_sec

    print(f"5q  : {r5.full_step_ms_mean:.2f} ms/step  ({it_s_5:.2f} it/s)")
    print(f"10q : {r10.full_step_ms_mean:.2f} ms/step  ({it_s_10:.2f} it/s)")
    print(f"slowdown (10q step-time / 5q step-time) = {slowdown:.3f}x")
    print()
    print(
        "For reference, the previously observed REAL TRAINING numbers were "
        "approximately 5q ~18-21 it/s and 10q ~2.73 it/s "
        f"(ratio ~{18/2.73:.2f}x to ~{21/2.73:.2f}x, using the reported range)."
    )
    print(
        "This script reports the measured numbers above; it does not "
        "itself decide whether they confirm, contradict, or fall outside "
        "that previously observed range -- compare them directly."
    )

    if 2 in by_qubits:
        r2 = by_qubits[2]
        print()
        print(
            f"2q  : {r2.full_step_ms_mean:.2f} ms/step  "
            f"({r2.iterations_per_sec:.2f} it/s)  "
            f"[10q/2q slowdown = {r10.full_step_ms_mean / r2.full_step_ms_mean:.3f}x]"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apples-to-apples real-batch-size (256) computational "
        "benchmark for proposed_phn across qubit counts, using the REAL "
        "model_factory.py / ProposedModel / VQCBranch implementation."
    )
    parser.add_argument(
        "--qubits", type=int, nargs="+", default=DEFAULT_QUBITS,
        help=f"Qubit counts to benchmark (default: {DEFAULT_QUBITS}). "
        f"Supported: {sorted(_QUBITS_TO_MODEL_ALIAS)} (each must have a "
        "registered model_factory.py alias).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=REAL_BATCH_SIZE,
        help=f"Batch size for the benchmark (default: {REAL_BATCH_SIZE}, "
        "the REAL training batch size from configs/config.py's "
        "BATCH_SIZE -- not read dynamically from config, since the "
        "explicit purpose here is to test the actual value used for "
        "training; pass a different value only to deliberately "
        "compare against a non-default batch size).",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP_ITERS,
        help=f"Warm-up iterations before timing (default: {DEFAULT_WARMUP_ITERS}).",
    )
    parser.add_argument(
        "--timed-iters", type=int, default=DEFAULT_TIMED_ITERS,
        help=f"Timed iterations to average over (default: {DEFAULT_TIMED_ITERS}).",
    )
    parser.add_argument(
        "--no-subprocess", action="store_true",
        help="Force in-process (single Python process) benchmarking "
        "instead of one fresh subprocess per qubit count.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for all benchmarked configurations (default: "
        "configs.config.RANDOM_SEED). The SAME seed is used for every "
        "qubit count.",
    )

    # Internal flags for subprocess re-invocation. Not intended for
    # direct use.
    parser.add_argument("--_internal-single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_internal-num-qubits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_internal-seed", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._internal_single_run:
        result = _run_single_benchmark(
            num_qubits=args._internal_num_qubits,
            batch_size=args.batch_size,
            warmup_iters=args.warmup,
            timed_iters=args.timed_iters,
            seed=args._internal_seed,
        )
        payload = asdict(result)
        print("RESULT_JSON:" + json.dumps(payload))
        return

    # ---- Normal (top-level) entry point ----
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from configs import config as project_config

    seed = args.seed if args.seed is not None else project_config.RANDOM_SEED

    unsupported = [q for q in args.qubits if q not in _QUBITS_TO_MODEL_ALIAS]
    if unsupported:
        print(
            f"WARNING: the following requested qubit counts have no "
            f"registered model_factory.py alias in this script and will "
            f"be skipped/reported as FAIL: {unsupported}. Supported: "
            f"{sorted(_QUBITS_TO_MODEL_ALIAS)}."
        )

    print_environment_info()

    print(f"Benchmarking qubit counts : {args.qubits}")
    print(f"Batch size                : {args.batch_size}  (real training batch size)")
    print(f"Warm-up iterations        : {args.warmup}")
    print(f"Timed iterations          : {args.timed_iters}")
    print(f"Random seed (all configs) : {seed}")
    print(f"VQC_DEPTH                 : read from configs/config.py, NOT "
          f"overridden by this script (currently "
          f"{getattr(project_config, 'VQC_DEPTH', 'N/A')})")
    print(f"Isolation mode             : "
          f"{'in-process (forced)' if args.no_subprocess else 'subprocess per qubit count'}")
    print()

    results: list[RealBatchBenchmarkResult] = []

    for nq in args.qubits:
        alias = _QUBITS_TO_MODEL_ALIAS.get(nq, "<unregistered>")
        print(f"--- num_qubits={nq} (alias={alias}) " + "-" * 30)
        if args.no_subprocess:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = _run_single_benchmark(
                num_qubits=nq,
                batch_size=args.batch_size,
                warmup_iters=args.warmup,
                timed_iters=args.timed_iters,
                seed=seed,
            )
        else:
            try:
                result = _run_in_subprocess(
                    num_qubits=nq,
                    batch_size=args.batch_size,
                    warmup_iters=args.warmup,
                    timed_iters=args.timed_iters,
                    seed=seed,
                )
            except Exception as exc:
                print(
                    f"  Subprocess launch failed ({exc}); "
                    "falling back to in-process for this qubit count."
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = _run_single_benchmark(
                    num_qubits=nq,
                    batch_size=args.batch_size,
                    warmup_iters=args.warmup,
                    timed_iters=args.timed_iters,
                    seed=seed,
                )

        results.append(result)

        if result.status == "PASS":
            print(
                f"  PASS  construction={result.construction_time_s:.3f}s  "
                f"device={result.device}  "
                f"forward={result.forward_ms_mean:.2f}ms  "
                f"backward={result.backward_ms_mean:.2f}ms  "
                f"full_step={result.full_step_ms_mean:.2f}ms  "
                f"({result.iterations_per_sec:.2f} it/s)"
            )
        else:
            print(f"  FAIL  reason={result.failure_reason}")
        print()

    print_comparison_table(results)
    print_status_and_detail_table(results)
    print_slowdown_analysis(results)

    print()
    print("=" * 88)
    print("BENCHMARK COMPLETE. No training was started. No dataset was loaded.")
    print("This is a computational-feasibility measurement only; it makes")
    print("no claim about forecasting performance (RMSE/MAE/R^2/nRMSE).")
    print("=" * 88)


if __name__ == "__main__":
    main()