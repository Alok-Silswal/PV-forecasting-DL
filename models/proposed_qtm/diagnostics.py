"""Optional batch aggregates; normal training performs no diagnostic logging."""

import math

import torch
from torch import Tensor


@torch.no_grad()
def summarize(weights: Tensor, q: Tensor, theta: Tensor, gamma: Tensor) -> dict:
    probabilities = weights / 24
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(1)
    return {
        "temporal_weight_mean": weights.mean(0).cpu().tolist(),
        "temporal_weight_std": weights.std(0, unbiased=False).cpu().tolist(),
        "temporal_weight_entropy_mean": entropy.mean().item(),
        "temporal_weight_entropy_normalized": (entropy.mean() / math.log(24)).item(),
        "q_mean": q.mean(0).cpu().tolist(), "q_std": q.std(0, unbiased=False).cpu().tolist(),
        "gamma": gamma.item(), "theta_min": theta.min().item(), "theta_max": theta.max().item(),
        "theta_edge_fraction": ((theta < 0.05) | (theta > math.pi - 0.05)).float().mean().item(),
        "edge_margin_radians": 0.05, "samples": len(weights),
    }
