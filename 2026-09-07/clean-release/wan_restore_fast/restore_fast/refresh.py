"""Restoration-relative continuous refresh demand. One policy, no fixed tiers."""

import torch
from torch.nn import functional as F
from .integrator import TokenUniPC


class RefinementState:
    def __init__(
        self,
        warm,
        scheduler,
        post10,
        endpoint,
        heat,
        routing,
        token_mask,
        subject_indices,
        low_indices,
        budget=2.0,
    ):
        self.core = TokenUniPC(warm, scheduler, post10)
        self.budget = budget
        self.token_shape = token_mask.shape
        self.eligible = token_mask.flatten().bool()
        self.subject_indices = subject_indices
        self.low_indices = low_indices
        self.risk = routing["routing_normalized_risk"][..., ::2, ::2].flatten().float()
        pool = lambda x: F.avg_pool3d(x.float(), (1, 2, 2), (1, 2, 2)).flatten()
        self.importance = (
            1 + 3 * pool(routing["subject_probability"]) + 0.5 * pool(heat).clamp(0, 1)
        )
        self.prior = 0.25 * pool(routing["latent_risk"]).clamp(0, 1)
        self.restored = self.core.pack(endpoint)
        self.repair_delta = torch.zeros_like(self.restored)
        self.previous_repair_delta = torch.zeros_like(self.restored)
        self.repair_i0 = self.core.i0.clone()
        self.repair_i1 = self.core.i1.clone()
        self.counts = torch.zeros(
            len(self.eligible), device=post10.device, dtype=torch.int16
        )
        self.evidence = torch.zeros_like(self.counts, dtype=torch.float32)
        self.innovation_rate = torch.zeros_like(self.evidence)
        self.correction_surprise = torch.zeros_like(self.evidence)
        self.masks = []
        self.scores = []
        self.steps = []

    @staticmethod
    def norm(x):
        return x.square().mean(-1).sqrt()

    def select(self, idx):
        c = self.core
        h = (c.lam(c.sigmas[idx]) - c.lam(c.sigmas[c.i0])).clamp_min(0)
        magnitude = self.norm(self.repair_delta)
        scale = magnitude + self.prior + 0.02
        neighbor = F.max_pool3d(
            (self.correction_surprise * torch.exp(-h)).reshape(self.token_shape),
            (3, 3, 3),
            1,
            1,
        ).flatten()
        remaining_prior = self.risk.pow(1) / (1 + self.evidence)
        score = (
            self.importance
            * h.square()
            * (remaining_prior + self.innovation_rate / scale + neighbor / scale)
        )
        due = self.eligible & (score >= self.budget)
        if idx == 49:
            due = self.eligible.clone()
        self.counts += due.to(torch.int16)
        self.masks.append(due)
        self.scores.append(score)
        self.steps.append(
            {
                "step": idx + 1,
                "computed_tokens": int(due.sum()),
                "subject_computed_tokens": int(due[self.subject_indices].sum()),
                "low_computed_tokens": int(due[self.low_indices].sum()),
                "wan_calls": int(bool(due.any())),
                "boundary_refresh": idx == 49,
            }
        )
        return due

    def update(self, flow, latent, idx, indices):
        c = self.core
        _, _, h = c.refresh(flow, latent, idx, indices)
        old = self.repair_delta[indices]
        previous = self.previous_repair_delta[indices]
        interval = (
            c.lambdas[self.repair_i0[indices]] - c.lambdas[self.repair_i1[indices]]
        )[:, None]
        actual = c.m0[indices] - self.restored[indices]
        innovation = actual - (old + h * (old - previous) / interval)
        change = actual - old
        self.previous_repair_delta[indices] = old
        self.repair_delta[indices] = actual
        self.repair_i1[indices] = self.repair_i0[indices]
        self.repair_i0[indices] = idx
        inv = self.norm(innovation)
        blend = 1 - torch.exp(-h.flatten())
        self.innovation_rate[indices] = (1 - blend) * self.innovation_rate[
            indices
        ] + blend * inv / h.flatten().square()
        self.correction_surprise[indices] = inv
        self.evidence[indices] += h.flatten() * torch.exp(
            -self.norm(change) / (self.prior[indices] + 0.02)
        )

    def dense(self, sigma):
        return self.core.unpack(self.core.dense(sigma))

    def tensors(self):
        expand = (
            lambda x: x.reshape(self.token_shape)
            .repeat_interleave(2, -2)
            .repeat_interleave(2, -1)
        )
        t, h, w = self.core.grid
        return {
            "refresh_count": expand(self.counts),
            "refinement_importance": expand(self.importance),
            "refinement_prior": expand(self.prior),
            "refinement_stability_evidence": expand(self.evidence),
            "refresh_events": torch.stack(self.masks).reshape(1, 40, t, h, w),
            "refinement_scores": torch.stack(self.scores).reshape(1, 40, t, h, w),
        }

    def report(self):
        active = self.counts[self.eligible]
        counts, number = torch.unique(active, return_counts=True)
        return {
            "refinement_budget": self.budget,
            "refinement_policy": "restoration_debt",
            "fixed_refresh_tiers": False,
            "fixed_refresh_schedule": False,
            "regional_compute_steps": self.steps,
            "active_token_refresh_histogram": {
                int(k): int(v) for k, v in zip(counts.tolist(), number.tolist())
            },
            "active_refresh_min": int(active.min()) if len(active) else 0,
            "active_refresh_max": int(active.max()) if len(active) else 0,
            "restored_tokens_wan_refreshes": 0,
            "native_step10_bridge_max_error": self.core.bridge_error,
            "force_final_fusion_refresh": True,
        }
