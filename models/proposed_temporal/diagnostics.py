"""Optional held-out diagnostics; no architecture-selection decisions."""

import math

import torch


@torch.no_grad()
def feature_diagnostics(model, pooled, moments, seed=0, edge_margin=0.05):
    if model.arm == "A":
        raise ValueError("Arm A has no augmentation features.")
    mode = model.training
    model.eval()
    try:
        theta, q = model.augmentation(moments)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(q), generator=generator).to(q.device)
        prediction = model.head(torch.cat((pooled, q), dim=1))
        permuted = model.head(torch.cat((pooled, q[indices]), dim=1))
        report = {
            "q_mean": q.mean(0).cpu().tolist(), "q_std": q.std(0, unbiased=False).cpu().tolist(),
            "theta_mean": theta.mean(0).cpu().tolist(),
            "theta_std": theta.std(0, unbiased=False).cpu().tolist(),
            "theta_min": theta.amin(0).cpu().tolist(), "theta_max": theta.amax(0).cpu().tolist(),
            "theta_near_edge_fraction": ((theta < edge_margin) | (theta > math.pi - edge_margin)).float().mean().item(),
            "edge_margin_radians": edge_margin,
            "permuted_q_prediction_rmse": (prediction - permuted).square().mean().sqrt().item(),
        }
        if model.arm in ("D", "E"):
            report["quantum_parameter_movement_l2"] = model.transform.parameter_movement()
        return report
    finally:
        model.train(mode)
