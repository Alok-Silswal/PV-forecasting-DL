"""Mandatory validation gate; residual checks follow only if it passes."""

import argparse
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import Subset

from .data import ROOT, reconstruct
from .frozen_baseline import FrozenBaseline, validation_gate


def residual_checks(datasets, feature_scaler, target_scaler, provenance, baseline, gate, run):
    from .batch_order import batches, permutation, prove_paired_order
    from .cache import CachedSplit, cache_identity, verify_cache, write_cache
    from .residual_branch import FAMILIES, FrozenResidualModel, canonical_state

    seed = 41 + run
    checks = {}
    for split, dataset in datasets.items():
        first_x, first_y = dataset[0]
        last_x, last_y = dataset[len(dataset) - 1]
        assert torch.equal(first_x, dataset.features[:24])
        assert torch.equal(first_y, dataset.target[24:27])
        assert torch.equal(last_x, dataset.features[-27:-3])
        assert torch.equal(last_y, dataset.target[-3:])
        checks[split + "_window_count"] = len(dataset)
    checks["split_boundary_windows_exact"] = True
    rng_before = torch.get_rng_state().clone()
    state = canonical_state(seed)
    models = [FrozenResidualModel(run, family, state) for family in FAMILIES]
    assert torch.equal(rng_before, torch.get_rng_state())
    checks["model_construction_preserves_global_torch_rng"] = True
    assert all(torch.equal(value, models[1].residual.state_dict()[key])
               for key, value in models[0].residual.state_dict().items())
    assert state["quantum_angles"].shape == (2, 6, 2)
    assert state["quantum_angles"].abs().max() <= 0.1
    checks["identical_projection_readout_quantum_initialization"] = True
    checks["quantum_trainable_counts"] = [sum(p.numel() for p in [m.residual.quantum_angles] if p.requires_grad) for m in models]
    assert checks["quantum_trainable_counts"] == [24, 0]
    checks["residual_trainable_counts"] = [sum(p.numel() for p in m.residual.parameters() if p.requires_grad) for m in models]
    assert checks["residual_trainable_counts"] == [819, 795]
    assert all(sum(p.numel() for p in m.frozen.parameters() if p.requires_grad) == 0 for m in models)
    for model in models:
        model.train()
        assert not any(module.training for module in model.frozen.modules())
    checks["baseline_frozen_and_eval_even_after_train"] = True
    hashes = prove_paired_order(len(datasets["train"]), seed)
    checks["complete_epoch_permutation_sha256"] = hashes
    left_batches = batches(len(datasets["train"]), seed, 0)
    right_batches = batches(len(datasets["train"]), seed, 0)
    assert len(left_batches) == len(right_batches)
    assert all(torch.equal(left, right) for left, right in zip(left_batches, right_batches))
    checks["batch_count"] = len(left_batches)
    checks["final_batch_size"] = len(left_batches[-1])
    for other_run in range(1, 6):
        other_seed = 41 + other_run
        if other_seed == seed:
            continue
        other = canonical_state(other_seed)
        assert all(not torch.equal(value, other[key]) for key, value in state.items())
        assert not torch.equal(permutation(len(datasets["train"]), seed, 0),
                               permutation(len(datasets["train"]), other_seed, 0))
    checks["different_run_initializations_and_orders"] = True
    for other_run in range(1, 6):
        FrozenBaseline(other_run)
    checks["all_five_checkpoints_load_strictly"] = True
    before = [{k: v.clone() for k, v in m.frozen.state_dict().items()} for m in models]
    quantum_before = models[1].residual.quantum_angles.detach().clone()
    directory = ROOT / f"experiments/proposed_rvqc/_smoke/horizon_15/run_{run}"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        smoke_data = {split: Subset(dataset, list(range(8))) for split, dataset in datasets.items()}
        identity = cache_identity(provenance, baseline, run, gate, "smoke")
        cache_dir = Path(temporary) / "cache"
        write_cache(cache_dir, smoke_data, baseline, identity, feature_scaler, target_scaler)
        verify_cache(cache_dir, identity)
        for split in smoke_data:
            cached = CachedSplit(cache_dir, split, scope="smoke")
            pooled, prediction, target = cached[torch.arange(8)]
            inputs = torch.stack([smoke_data[split][i][0] for i in range(8)])
            direct_pooled, direct_prediction = baseline(inputs)
            torch.testing.assert_close(pooled, direct_pooled, rtol=0, atol=0)
            torch.testing.assert_close(prediction, direct_prediction, rtol=0, atol=0)
            assert torch.equal(target, torch.stack([smoke_data[split][i][1] for i in range(8)]))
            with torch.no_grad():
                torch.testing.assert_close(direct_prediction, baseline.baseline(inputs), rtol=0, atol=0)
        checks["cache_equals_direct_and_extraction_preserves_predictions"] = True
        train_cache = CachedSplit(cache_dir, "train", scope="smoke")
        pooled, prediction, target = train_cache[torch.arange(8)]
        with torch.no_grad():
            assert torch.equal(models[0].forward_cached(pooled, prediction), models[1].forward_cached(pooled, prediction))
        schedules = []
        for index, model in enumerate(models):
            optimizer = torch.optim.Adam([p for p in model.residual.parameters() if p.requires_grad],
                                         lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model.forward_cached(pooled, prediction), target)
            loss.backward()
            for layer in [model.residual.projection, model.residual.readout]:
                for parameter in layer.parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    assert parameter.grad.abs().sum() > 0
            angles = model.residual.quantum_angles
            if index == 0:
                assert angles.grad is not None and torch.isfinite(angles.grad).all() and angles.grad.abs().sum() > 0
            else:
                assert angles.grad is None
            optimizer.step()
            schedule = [optimizer.param_groups[0]["lr"]]
            scheduler.step()
            schedule.append(optimizer.param_groups[0]["lr"])
            for _ in range(99):
                optimizer.zero_grad(set_to_none=True)
                optimizer.step()
                scheduler.step()
                schedule.append(optimizer.param_groups[0]["lr"])
            schedules.append(schedule)
            assert all(torch.equal(value, model.frozen.state_dict()[key]) for key, value in before[index].items())
            assert all(p.grad is None for p in model.frozen.parameters())
            model.eval()
            with torch.no_grad():
                expected = model(inputs)
            checkpoint = Path(temporary) / f"{model.residual.family}.pt"
            torch.save({"epoch": 0, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(), "best_val_loss": float(loss.detach())}, checkpoint)
            restored = FrozenResidualModel(run, model.residual.family, state).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"], strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
        assert schedules[0] == schedules[1]
        assert abs(schedules[0][-1] - 1e-6) < 1e-15
        assert torch.equal(quantum_before, models[1].residual.quantum_angles)
        assert not torch.equal(state["quantum_angles"], models[0].residual.quantum_angles)
    checks.update({"projection_readout_gradients_both": True, "trainable_quantum_gradients": True,
                   "baseline_state_unchanged_after_step": True, "frozen_quantum_angles_unchanged": True,
                   "checkpoint_reload_exact_predictions": True, "identical_cosine_schedule": True,
                   "test_rows_read": provenance["test_rows_read"]})
    assert checks["test_rows_read"] == 0
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse-gate", type=Path, help="Reuse only an exactly matching full-validation gate report.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    datasets, feature_scaler, target_scaler, provenance = reconstruct(args.processed_csv)
    print("Reconstructed train/validation; no test rows read.", flush=True)
    baseline = FrozenBaseline(args.run)
    if args.reuse_gate:
        previous = json.loads(args.reuse_gate.read_text())
        gate = previous["validation_gate"]
        from .data import file_hash
        history = ROOT / f"experiments/proposed/horizon_15/run_{args.run}/history.json"
        if (previous["provenance"] != provenance or not gate["passed"] or
                gate["checkpoint_sha256"] != baseline.sha256 or gate["history_sha256"] != file_hash(history)):
            raise SystemExit("STOP: previous validation gate provenance does not match.")
    else:
        gate = validation_gate(baseline, datasets["validation"], args.run)
    report = {"provenance": provenance, "validation_gate": gate,
              "baseline_trainable_parameters": sum(p.numel() for p in baseline.parameters() if p.requires_grad)}
    directory = ROOT / f"experiments/proposed_rvqc/_smoke/horizon_15/run_{args.run}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(gate, indent=2), flush=True)
    if not gate["passed"]:
        raise SystemExit("STOP: reconstructed validation does not reproduce the Proposed checkpoint.")
    report["residual_checks"] = residual_checks(datasets, feature_scaler, target_scaler, provenance, baseline, gate, args.run)
    report["passed"] = True
    (directory / "preflight.json").write_text(json.dumps(report, indent=2))
    print("PASS: all mandatory residual preflight checks.", flush=True)


if __name__ == "__main__":
    main()
