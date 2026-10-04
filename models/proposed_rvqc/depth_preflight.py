"""Lightweight depth checks on train/validation only; not a scientific pilot."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from .batch_order import permutation
from .data import ROOT, VAL_END, reconstruct
from .depth_branch import DepthResidualBranch, depth_state
from .depth_pilot import PILOT_ROOT, angle_analysis
from .frozen_baseline import FrozenBaseline
from .residual_branch import FAMILIES, FrozenResidualModel, ResidualBranch, canonical_state


def checks(processed_csv, run):
    csv_path = Path(processed_csv).resolve()
    original_open = Path.open
    bytes_read, lines_read = 0, 0

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            nonlocal bytes_read, lines_read
            chunk = self.stream.read(size)
            bytes_read += len(chunk)
            lines_read += chunk.count(b"\n")
            assert lines_read <= VAL_END + 1
            if lines_read == VAL_END + 1:
                assert chunk.endswith(b"\n")
            return chunk

    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return BoundedReader(stream) if path.resolve() == csv_path else stream

    with patch.object(Path, "open", guarded_open):
        datasets, _, _, provenance = reconstruct(csv_path)
    assert lines_read == VAL_END + 1 and provenance["test_rows_read"] == 0
    seed = 41 + run
    rng_before = torch.get_rng_state().clone()
    states = [depth_state(seed, blocks) for blocks in (1, 2, 3, 4)]
    for blocks, state in enumerate(states, 1):
        assert state["quantum_angles"].shape == (blocks, 6, 2)
        for key in ("projection.weight", "projection.bias", "readout.weight", "readout.bias"):
            assert torch.equal(state[key], states[0][key])
            assert not torch.equal(state[key], depth_state(seed + 1, blocks)[key])
        assert torch.equal(state["quantum_angles"], states[-1]["quantum_angles"][:blocks])
        assert not torch.equal(state["quantum_angles"], depth_state(seed + 1, blocks)["quantum_angles"])
    assert all(torch.equal(v, states[1][k]) for k, v in canonical_state(seed).items())
    branches = [(blocks, family, DepthResidualBranch(family, states[blocks - 1], blocks))
                for blocks in (1, 2, 3, 4) for family in FAMILIES]
    assert torch.equal(rng_before, torch.get_rng_state())
    baseline = FrozenBaseline(run)
    before = {k: v.clone() for k, v in baseline.state_dict().items()}
    assert not any(p.requires_grad for p in baseline.parameters()) and not baseline.training
    inputs = torch.stack([datasets["train"][i][0] for i in range(4)])
    target = torch.stack([datasets["train"][i][1] for i in range(4)])
    pooled, prediction = baseline(inputs)
    old = ResidualBranch("proposed_rvqc", canonical_state(seed))
    default = DepthResidualBranch("proposed_rvqc", depth_state(seed))
    assert torch.equal(old(pooled), default(pooled))
    counts, residual_counts = [], []
    for blocks, family, branch in branches:
        tape = branch.circuit.construct((torch.zeros(4, 6), branch.quantum_angles), {})
        expected_operations = [(name, [wire]) for wire in range(6) for name in ("RY", "RZ", "RY")]
        expected_operations += [("CNOT", [wire, (wire + 1) % 6]) for wire in range(6)]
        assert [(op.name, list(op.wires)) for op in tape.operations] == expected_operations * blocks
        initial = branch.quantum_angles.detach().clone()
        expected = blocks * 12 if family == "proposed_rvqc" else 0
        assert (branch.quantum_angles.numel() if branch.quantum_angles.requires_grad else 0) == expected
        counts.append(expected)
        residual_counts.append(sum(p.numel() for p in branch.parameters() if p.requires_grad))
        optimizer = torch.optim.Adam([p for p in branch.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-5)
        nn.functional.mse_loss(branch.forward_cached(pooled, prediction), target).backward()
        for parameter in (branch.projection.weight, branch.projection.bias, branch.readout.weight, branch.readout.bias):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        if expected:
            gradient = branch.quantum_angles.grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        else:
            assert branch.quantum_angles.grad is None
        optimizer.step()
        if not expected:
            assert torch.equal(initial, branch.quantum_angles)
        assert all(torch.equal(v, baseline.state_dict()[k]) for k, v in before.items())
    hashes = []
    for epoch in range(100):
        left = permutation(len(datasets["train"]), seed, epoch)
        for blocks in (1, 2, 3, 4):
            depth_state(seed, blocks)
            assert torch.equal(left, permutation(len(datasets["train"]), seed, epoch))
        hashes.append(hashlib.sha256(left.numpy().tobytes()).hexdigest())
    assert not torch.equal(permutation(len(datasets["train"]), seed, 0),
                           permutation(len(datasets["train"]), seed + 1, 0))
    official = FrozenResidualModel(run, "proposed_rvqc")
    checkpoint_path = ROOT / f"experiments/proposed_rvqc/horizon_15/run_{run}/checkpoints/best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    official.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert all(torch.equal(v, official.frozen.state_dict()[k]) for k, v in before.items())
    default.load_state_dict(official.residual.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(official(inputs), default.forward_cached(pooled, prediction))
    analysis = angle_analysis(torch.zeros(2, 6, 2), torch.full((2, 6, 2), 0.25))
    assert analysis["mean_abs_change"] == 0.25 and len(analysis["per_block"]) == 2
    return {"run": run, "seed": seed, "quantum_trainable_counts": counts,
            "residual_trainable_counts": residual_counts,
            "train_window_count": len(datasets["train"]),
            "validation_window_count": len(datasets["validation"]),
            "source_rows_read": VAL_END, "test_rows_read": 0, "source_bytes_read": bytes_read,
            "nested_angles_and_identical_common_initialization": True,
            "global_torch_rng_preserved": True,
            "all_100_complete_epoch_permutations_sha256": hashes,
            "different_run_initializations_and_orders": True,
            "finite_nonzero_gradients": True, "frozen_angles_unchanged": True,
            "baseline_frozen_and_state_unchanged": True,
            "default_two_block_predictions_exact": True,
            "exact_rotation_order_and_cnot_ring_all_depths": True,
            "official_checkpoint_strict_load_and_exact_inference": True,
            "full_historical_validation_gate": "Deferred to real pilot startup; this is a lightweight preflight."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = checks(args.processed_csv, args.run)
    destination = PILOT_ROOT / f"_smoke/run_{args.run}/preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Depth preflight passed; no test rows read.\nReport: {destination}")


if __name__ == "__main__":
    main()
