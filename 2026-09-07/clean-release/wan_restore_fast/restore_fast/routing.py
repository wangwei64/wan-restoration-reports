"""Mutually exclusive S/L/R ownership; one adaptive, calibrated latent-only route."""

import cv2, numpy as np, torch
from torch.nn import functional as F


def clean_subject(probability, high=0.60, low=0.40, min_pixels=4):
    raw = probability.detach().float().cpu().numpy()[0, 0]
    out = np.zeros_like(raw)
    for t, p in enumerate(raw):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (p >= low).astype(np.uint8), 8
        )
        for i in range(1, count):
            area = labels == i
            if stats[i, cv2.CC_STAT_AREA] >= min_pixels and bool(
                np.any(p[area] >= high)
            ):
                out[t, area] = 1
    return torch.from_numpy(out)[None, None].to(probability.device)


def route(
    z10, z9, expected, saliency, heat, subject_probability, calibration, tolerance=0.010
):
    grid = (z10.shape[-3], z10.shape[-2] // 2, z10.shape[-1] // 2)
    saliency = saliency.reshape(1, 1, *grid).float()
    prob = F.avg_pool3d(subject_probability.float(), (1, 2, 2), (1, 2, 2)).clamp(
        0.005, 0.995
    )
    attn = saliency.clamp(0.005, 0.995)
    probability = torch.sigmoid(1.5 * torch.logit(prob) + 0.5 * torch.logit(attn))
    binary = clean_subject(probability)
    error = F.avg_pool3d(expected.float(), (1, 2, 2), (1, 2, 2))
    knots = torch.tensor(calibration["predicted_error_knots"], device=error.device)
    values = torch.tensor(calibration["calibrated_error"], device=error.device)
    upper = torch.searchsorted(knots, error.flatten()).clamp(1, len(knots) - 1)
    lower = upper - 1
    fraction = (
        (error.flatten() - knots[lower]) / (knots[upper] - knots[lower]).clamp_min(1e-8)
    ).clamp(0, 1)
    error = (values[lower] + fraction * (values[upper] - values[lower])).view_as(error)
    drift = F.avg_pool3d((z10 - z9).abs().mean(1, keepdim=True), (1, 2, 2), (1, 2, 2))
    motion = (drift / drift.mean().clamp_min(1e-6)).clamp(0, 3)
    detail = F.avg_pool3d(heat.float(), (1, 2, 2), (1, 2, 2)).clamp(0, 1)
    threshold = (
        tolerance
        * (1 - 0.25 * binary)
        * (1 + 0.10 * (1 - detail))
        / (1 + 0.20 * motion + 0.15 * detail)
    )
    motion_floor = float(calibration.get("routing_motion_floor", 0.003))
    risk = error + motion_floor * motion
    normalized = risk / threshold.clamp_min(1e-6)
    normalized = torch.maximum(
        normalized, 0.75 * F.max_pool3d(normalized, (3, 3, 3), 1, 1)
    )
    token_mask = (normalized > 1).float()
    edge = F.max_pool3d(binary, (1, 3, 3), 1, (0, 1, 1)) - (
        -F.max_pool3d(-binary, (1, 3, 3), 1, (0, 1, 1))
    )
    token_mask = torch.maximum(token_mask, ((edge > 0) & (normalized > 0.75)).float())
    expand = lambda x: x.repeat_interleave(2, -2).repeat_interleave(2, -1)
    active = expand(token_mask)
    semantic = expand(binary)
    info = {
        "strategy": "local restoration-error threshold; no top-k; no coverage cap",
        "calibration_enabled": True,
        "quality_tolerance": tolerance,
        "motion_error_floor": motion_floor,
        "threshold_min": float(threshold.min()),
        "threshold_max": float(threshold.max()),
        "threshold_mean": float(threshold.mean()),
        "subject_area": float(binary.mean()),
        "actual_token_coverage": float(token_mask.mean()),
        "subject_tokens_restored_fraction": float(
            ((1 - token_mask) * binary).sum() / binary.sum().clamp_min(1)
        ),
        "subject_probability_source": "offline distilled geometry + automatic prompt attention",
        "mutually_exclusive": True,
    }
    extra = {
        "subject_binary": semantic,
        "subject_probability": expand(probability),
        "routing_threshold": expand(threshold),
        "routing_risk": expand(risk),
        "routing_normalized_risk": expand(normalized),
        "routing_confidence": expand(torch.sigmoid((1 - normalized) / 0.25)),
    }
    return active, token_mask, semantic, info, extra
