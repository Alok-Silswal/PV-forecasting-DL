"""Epoch order is independent of every model-construction RNG."""

import hashlib
import torch


def permutation(size, run_seed, epoch):
    if size < 1 or epoch not in range(100):
        raise ValueError("Expected a nonempty dataset and an epoch in 0..99.")
    generator = torch.Generator(device="cpu").manual_seed(run_seed + 20000 + epoch)
    return torch.randperm(size, generator=generator)


def batches(size, run_seed, epoch):
    return permutation(size, run_seed, epoch).split(256)


def prove_paired_order(size, run_seed):
    hashes = []
    for epoch in range(100):
        left = permutation(size, run_seed, epoch)
        # Model construction/global RNG consumption must not affect this result.
        with torch.random.fork_rng(devices=[]):
            torch.rand(113)
            right = permutation(size, run_seed, epoch)
        if not torch.equal(left, right):
            raise AssertionError(f"Different complete permutations at epoch {epoch}.")
        if not torch.equal(torch.sort(left).values, torch.arange(size)):
            raise AssertionError("Permutation does not cover each index exactly once.")
        hashes.append(hashlib.sha256(left.numpy().tobytes()).hexdigest())
    return hashes
