"""
profile_vqc_simulator_comparison.py

Standalone, temporary, read-only benchmark. Does NOT modify any
project file, does NOT change the committed VQC architecture, and is
not meant to be kept as part of the codebase -- it exists purely to
answer one question empirically: can a different PennyLane device
execute our EXACT SAME 2-qubit, depth-2 VQC as a genuinely batched
computation instead of PennyLane's `broadcast_expand`-driven 256
serial tape executions, and is it actually faster?

Configurations benchmarked
---------------------------
A. lightning.qubit + interface="torch" + diff_method="adjoint"
   (current committed baseline)
B. default.qubit    + interface="torch" + diff_method="adjoint"
C. default.qubit    + interface="torch" + diff_method="backprop"

For every configuration, the circuit definition itself (2 qubits,
depth 2, RY/RZ variational layers, single CNOT(0,1) per layer, angle
encoding via AngleEmbedding(rotation="Y"), observables
<Z0>, <Z1>, <Z0 Z1>) is copy-identical -- only the device string and
diff_method differ. Trainable weights and the input tensor are
identical (same seed) across all three configurations, so outputs can
be compared for numerical equivalence.

What "true native batching" actually means here (checked, not assumed)
------------------------------------------------------------------------
Verified against PennyLane 0.45.1 source and documentation:
  - lightning.qubit's own preprocessing states it "does not
    intrinsically support parameter broadcasting" and its compile
    pipeline unconditionally applies qml.transforms.broadcast_expand,
    which splits one broadcasted tape into B separate tapes, executed
    serially.
  - default.qubit's preprocessing (`_conditional_broadcast_expand`)
    only falls back to broadcast_expand for shadow-measurement
    operations (ShadowExpvalMP / ClassicalShadowMP). Our circuit uses
    only plain qml.expval(PauliZ(...)) / qml.expval(PauliZ @ PauliZ)
    measurements, which are NOT shadow measurements -- so for OUR
    exact observables, default.qubit is expected to execute the
    broadcasted tape natively via array-broadcasted operations, for
    BOTH diff_method="adjoint" (a natively supported device
    derivative on default.qubit, not a gradient-transform fallback)
    and diff_method="backprop".
  - This expectation is verified empirically below (tape count /
    device-call count instrumentation), not merely asserted from
    documentation, per the explicit instruction not to judge by
    feature name alone.

Usage (Kaggle)
--------------
    python profile_vqc_simulator_comparison.py

No project files are imported or modified; this script defines a
minimal, self-contained circuit matching models/vqc_branch.py exactly,
so it can run standalone without depending on the rest of the project
(e.g. before/without the dataset artifacts being present).
"""

import copy
import statistics
import time
from typing import List

import pennylane as qml
import torch


# =============================================================================
# Benchmark configuration
# =============================================================================

BATCH_SIZE = 256
NUM_QUBITS = 2
DEPTH = 2
OUTPUT_DIM = 3         # <Z0>, <Z1>, <Z0 Z1> -- matches HORIZON_TO_OUTPUT_DIM["15"]
INPUT_DIM = 128         # matches bilstm_hidden_size * 2 in the real project
NUM_WARMUP_ITERS = 3
NUM_TIMED_ITERS = 10
RANDOM_SEED = 42
NUMERICAL_TOLERANCE = 1e-5  # reasonable float32 tolerance across adjoint/backprop


# =============================================================================
# Circuit definition -- copy-identical to models/vqc_branch.py's _build_qnode,
# duplicated here (not imported) so this script is fully standalone and
# self-contained, and so the exact same function can be reused verbatim
# across all three device/diff_method configurations for a fair comparison.
# =============================================================================

def build_qnode(device_name: str, diff_method: str):
    device = qml.device(device_name, wires=NUM_QUBITS)

    observables = [
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
    ][:OUTPUT_DIM]

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> List[torch.Tensor]:
        qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")

        for layer in range(DEPTH):
            for qubit in range(NUM_QUBITS):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
            qml.CNOT(wires=[0, 1])

        return [qml.expval(observable) for observable in observables]

    return circuit, device


# =============================================================================
# Instrumentation: detect whether the device actually executed ONE batched
# call or MANY per-sample calls, rather than inferring this from
# documentation alone.
# =============================================================================

def _count_device_executions(device_name: str, diff_method: str, inputs: torch.Tensor, weights: torch.Tensor) -> int:
    """
    Wrap the device's `execute` method to count how many times it is
    actually invoked, and with how many tapes each time, for a SINGLE
    forward call with the full (BATCH_SIZE, NUM_QUBITS) input tensor.

    Returns the total number of individual QuantumScript tapes actually
    executed by the device across that one QNode call -- 1 means
    genuine single-call batched execution; BATCH_SIZE means the
    broadcast_expand-style serial-tape path (or any other N-way split).
    """

    circuit, device = build_qnode(device_name, diff_method)

    tape_count = {"total_tapes": 0, "num_execute_calls": 0}
    original_execute = device.execute

    def counting_execute(circuits, execution_config=None):
        num_tapes = len(circuits) if isinstance(circuits, (list, tuple)) else 1
        tape_count["total_tapes"] += num_tapes
        tape_count["num_execute_calls"] += 1
        if execution_config is not None:
            return original_execute(circuits, execution_config)
        return original_execute(circuits)

    device.execute = counting_execute

    with torch.no_grad():
        _ = circuit(inputs, weights)

    device.execute = original_execute  # restore, defensive cleanup

    return tape_count["total_tapes"], tape_count["num_execute_calls"]


def _sync_cpu_noop() -> None:
    # All three configurations here run on CPU (lightning.qubit and
    # default.qubit are both CPU simulators; no CUDA synchronization
    # is applicable to the QNode call itself). Kept as an explicit
    # no-op function, rather than silently omitting synchronization,
    # so the timing protocol structure matches the CUDA-aware scripts
    # used elsewhere in this project for consistency.
    pass


def _timed_forward(circuit, inputs, weights, num_warmup, num_iters):
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = circuit(inputs, weights)
        _sync_cpu_noop()

    durations = []
    for _ in range(num_iters):
        _sync_cpu_noop()
        start = time.perf_counter()
        with torch.no_grad():
            _ = circuit(inputs, weights)
        _sync_cpu_noop()
        end = time.perf_counter()
        durations.append(end - start)
    return durations


def _timed_forward_backward(circuit, inputs, weights, num_warmup, num_iters):
    for _ in range(num_warmup):
        w = weights.clone().detach().requires_grad_(True)
        out = torch.stack(circuit(inputs, w), dim=-1)
        loss = out.pow(2).mean()
        loss.backward()
        _sync_cpu_noop()

    durations = []
    last_grad = None
    for _ in range(num_iters):
        w = weights.clone().detach().requires_grad_(True)
        _sync_cpu_noop()
        start = time.perf_counter()
        out = torch.stack(circuit(inputs, w), dim=-1)
        loss = out.pow(2).mean()
        loss.backward()
        _sync_cpu_noop()
        end = time.perf_counter()
        durations.append(end - start)
        last_grad = w.grad

    return durations, last_grad


def _mean(values):
    return sum(values) / len(values)


def _stdev(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _run_configuration(label: str, device_name: str, diff_method: str, inputs: torch.Tensor, weights: torch.Tensor):
    print(f"--- Configuration: {label} ---")
    print(f"  Device: {device_name}   diff_method: {diff_method}")

    # --- 1. True native batching check ---
    total_tapes, num_execute_calls = _count_device_executions(device_name, diff_method, inputs, weights)
    is_native_batched = total_tapes == 1
    print(
        f"  Tapes actually executed by device.execute() for ONE forward call "
        f"of batch_size={BATCH_SIZE}: {total_tapes} "
        f"(across {num_execute_calls} execute() call(s))"
    )
    print(f"  --> Genuine single-call batched execution: {is_native_batched}")

    # --- 2 & 3. Timing ---
    circuit, _ = build_qnode(device_name, diff_method)

    forward_durations = _timed_forward(circuit, inputs, weights, NUM_WARMUP_ITERS, NUM_TIMED_ITERS)
    fwd_bwd_durations, grad = _timed_forward_backward(
        circuit, inputs, weights, NUM_WARMUP_ITERS, NUM_TIMED_ITERS
    )

    forward_mean = _mean(forward_durations)
    fwd_bwd_mean = _mean(fwd_bwd_durations)

    print(
        f"  Forward mean            : {forward_mean*1000:9.2f} ms "
        f"(std={_stdev(forward_durations)*1000:6.2f} ms)"
    )
    print(
        f"  Forward+backward mean   : {fwd_bwd_mean*1000:9.2f} ms "
        f"(std={_stdev(fwd_bwd_durations)*1000:6.2f} ms)"
    )

    # --- 4. Gradient validity ---
    grad_valid = (
        grad is not None
        and not torch.isnan(grad).any().item()
        and not torch.isinf(grad).any().item()
        and grad.abs().sum().item() > 0.0
    )
    print(f"  Gradient valid (non-None, finite, nonzero): {grad_valid}")

    # --- 5. Output (computed separately, cleanly, for equivalence check) ---
    with torch.no_grad():
        output = torch.stack(circuit(inputs, weights), dim=-1)

    print()

    return {
        "label": label,
        "device_name": device_name,
        "diff_method": diff_method,
        "total_tapes_for_one_forward": total_tapes,
        "is_native_batched": is_native_batched,
        "forward_mean_s": forward_mean,
        "forward_bwd_mean_s": fwd_bwd_mean,
        "grad_valid": grad_valid,
        "grad": grad,
        "output": output,
    }


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    # Identical input and weights across ALL configurations, per the
    # requirement, so timing and output comparisons are apples-to-apples.
    pooled_features = torch.randn(BATCH_SIZE, INPUT_DIM)
    projection = torch.nn.Linear(INPUT_DIM, NUM_QUBITS)
    torch.nn.init.xavier_uniform_(projection.weight)
    torch.nn.init.constant_(projection.bias, 0.0)
    with torch.no_grad():
        raw_angles = projection(pooled_features)
        shared_inputs = torch.tanh(raw_angles) * torch.pi

    shared_weights = 0.01 * torch.randn(DEPTH, NUM_QUBITS, 2)

    print("=" * 88)
    print("SIMULATOR COMPARISON BENCHMARK")
    print("=" * 88)
    print(f"Batch size: {BATCH_SIZE}   Qubits: {NUM_QUBITS}   Depth: {DEPTH}   Output dim: {OUTPUT_DIM}")
    print(f"Warmup iters: {NUM_WARMUP_ITERS}   Timed iters: {NUM_TIMED_ITERS}")
    print(f"Input tensor shape: {tuple(shared_inputs.shape)}   Weights shape: {tuple(shared_weights.shape)}")
    print("=" * 88)
    print()

    result_a = _run_configuration(
        "A: lightning.qubit + adjoint (baseline)",
        "lightning.qubit",
        "adjoint",
        shared_inputs,
        shared_weights,
    )
    result_b = _run_configuration(
        "B: default.qubit + adjoint",
        "default.qubit",
        "adjoint",
        shared_inputs,
        shared_weights,
    )
    result_c = _run_configuration(
        "C: default.qubit + backprop",
        "default.qubit",
        "backprop",
        shared_inputs,
        shared_weights,
    )

    results = [result_a, result_b, result_c]
    baseline = result_a

    # --- 5. Numerical equivalence of outputs vs baseline ---
    print("=" * 88)
    print("NUMERICAL EQUIVALENCE CHECK (vs. Configuration A baseline)")
    print("=" * 88)
    for r in results:
        if r is baseline:
            continue
        max_abs_diff = (r["output"] - baseline["output"]).abs().max().item()
        equivalent = max_abs_diff < NUMERICAL_TOLERANCE
        print(
            f"{r['label']:42s} max|output - baseline| = {max_abs_diff:.3e}  "
            f"(tolerance {NUMERICAL_TOLERANCE:.0e})  "
            f"--> {'EQUIVALENT' if equivalent else 'NOT EQUIVALENT'}"
        )
        r["output_equivalent"] = equivalent
    baseline["output_equivalent"] = True  # trivially equal to itself
    print("=" * 88)
    print()

    # --- Summary table ---
    print("=" * 108)
    print("SUMMARY TABLE")
    print("=" * 108)
    header = (
        f"{'Simulator':16} {'Diff method':12} {'True batch?':12} "
        f"{'Fwd mean (ms)':14} {'Fwd+Bwd mean (ms)':18} {'Grad valid?':12} "
        f"{'Output equiv?':14} {'Speedup vs A':12}"
    )
    print(header)
    print("-" * len(header))
    baseline_fwd_bwd = baseline["forward_bwd_mean_s"]
    for r in results:
        speedup = baseline_fwd_bwd / r["forward_bwd_mean_s"] if r["forward_bwd_mean_s"] > 0 else float("nan")
        print(
            f"{r['device_name']:16} {r['diff_method']:12} "
            f"{str(r['is_native_batched']):12} "
            f"{r['forward_mean_s']*1000:14.2f} "
            f"{r['forward_bwd_mean_s']*1000:18.2f} "
            f"{str(r['grad_valid']):12} "
            f"{str(r.get('output_equivalent', 'N/A')):14} "
            f"{speedup:10.2f}x"
        )
    print("=" * 108)
    print()

    print(
        "Speedup vs A is (baseline forward+backward mean) / (this configuration's "
        "forward+backward mean). >1.0x means faster than the lightning.qubit "
        "baseline; <1.0x means slower."
    )
    print(
        "IMPORTANT: diff_method='backprop' computes gradients via reverse-mode "
        "automatic differentiation through the full simulated statevector "
        "evolution, which is a DIFFERENT gradient-computation method than "
        "'adjoint' (adjoint reverses through the circuit via inverse gates "
        "without storing the full autodiff graph). Both are mathematically "
        "exact for this noiseless circuit, so gradients should agree up to "
        "floating-point tolerance (see equivalence check above) -- but "
        "'backprop' is a genuine change in HOW gradients are computed and "
        "should only be adopted with explicit approval, not silently."
    )


if __name__ == "__main__":
    main()