"""Vectorized UniPC-2 histories with independent observation times per patch.

Only real model evaluations enter the two clean-output histories. The equations
are the bh2/predict_x0 order-2 specialization of Wan's FlowUniPC scheduler.
"""

import copy
import torch

_COEFFICIENTS = {}


def coefficients(sigmas, device):
    key = (tuple(float(s) for s in sigmas), str(device))
    if key in _COEFFICIENTS:
        return _COEFFICIENTS[key]
    lam = torch.stack([torch.log(1 - s) - torch.log(s) for s in sigmas])
    n = len(sigmas)
    factor = torch.zeros(n, n)
    weight = torch.zeros(n, n)
    rk = torch.ones(n, n, n)
    rhs = torch.zeros(n, n, 2)
    valid = []
    for source in range(9, n - 1):
        for target in range(source + 1, n - 1):
            h = lam[target] - lam[source]
            bh = torch.expm1(-h)
            phi2 = bh / (-h) - 1
            factor[target, source] = sigmas[target] / sigmas[source]
            weight[target, source] = (1 - sigmas[target]) * bh
            rk[target, source, 8:source] = (lam[8:source] - lam[source]) / h
            rhs[target, source, 0] = phi2 / bh
            rhs[target, source, 1] = (phi2 / (-h) - 0.5) * 2 / bh
            valid.extend((target, source, prev) for prev in range(8, source))
    # ATen implements CUDA tensor / CPU scalar by multiplying the scalar's
    # FP32 reciprocal. A vector divide rounds differently at rare BF16 ties.
    # Precompute the same CPU inverse for independent token histories.
    inverse_rk = rk.reciprocal().to(device)
    factor, weight, rk, rhs = [x.to(device) for x in (factor, weight, rk, rhs)]
    ids = torch.tensor(valid, device=device)
    t, s, p = ids.unbind(1)
    matrices = torch.ones((len(ids), 2, 2), device=device)
    matrices[:, 1, 0] = rk[t, s, p]
    values = torch.linalg.solve(matrices, rhs[t, s])
    rho = torch.zeros(n, n, n, 2, device=device)
    rho[t, s, p] = values
    _COEFFICIENTS[key] = (factor, weight, inverse_rk, rho)
    return _COEFFICIENTS[key]


class TokenUniPC:
    def __init__(self, warm, scheduler, post10):
        saved, sample, flow = warm
        low = copy.deepcopy(saved)
        assert (
            low.config.solver_order == 2
            and low.config.solver_type == "bh2"
            and low.predict_x0
        )
        assert (
            not low.config.thresholding
            and not low.disable_corrector
            and not low.solver_p
        )
        self.shape = post10.shape
        self.grid = (post10.shape[2], post10.shape[3] // 2, post10.shape[4] // 2)
        endpoint = low.step(flow, low.timesteps[9], sample, return_dict=False)[0]
        assert torch.equal(endpoint, post10)
        self.sigmas = scheduler.sigmas.float().to(post10.device)
        # Match the original scheduler's scalar CPU lambda coefficients, including
        # its subtraction order; low-noise extrapolation amplifies a few ulps.
        self.lambdas = torch.stack(
            [torch.log(1 - s) - torch.log(s) for s in scheduler.sigmas]
        ).to(post10.device)
        self.sigma_indices = {float(s): i for i, s in enumerate(scheduler.sigmas)}
        self.factors, self.weights, self.inverse_rks, self.rhos = coefficients(
            scheduler.sigmas, post10.device
        )
        self.x = self.pack(low.last_sample)
        self.m0 = self.pack(low.model_outputs[-1])
        self.m1 = self.pack(low.model_outputs[-2])
        n = len(self.x)
        self.i0 = torch.full((n,), 9, device=post10.device, dtype=torch.long)
        self.i1 = torch.full_like(self.i0, 8)
        self.bridge_error = float(
            (self.unpack(self.dense(self.sigmas[10])) - post10).abs().max()
        )
        assert self.bridge_error < 1e-5, self.bridge_error

    def pack(self, x):
        b, c, t, h, w = x.shape
        assert b == 1
        return (
            x.reshape(c, t, h // 2, 2, w // 2, 2)
            .permute(1, 2, 4, 0, 3, 5)
            .reshape(-1, c * 4)
            .float()
        )

    def unpack(self, x):
        b, c, t, h, w = self.shape
        return (
            x.reshape(t, h // 2, w // 2, c, 2, 2)
            .permute(3, 0, 1, 4, 2, 5)
            .reshape(self.shape)
        )

    @staticmethod
    def lam(sigma):
        return torch.log(1 - sigma) - torch.log(sigma)

    def slope(self):
        gap = (self.lambdas[self.i0] - self.lambdas[self.i1])[:, None]
        return (self.m0 - self.m1) / gap

    def dense(self, target):
        if float(target) == 0:
            return self.m0
        target_idx = self.sigma_indices[float(target)]
        inverse_rk = self.inverse_rks[target_idx, self.i0, self.i1, None]
        d1 = (self.m1 - self.m0) * inverse_rk
        weight = self.weights[target_idx, self.i0, None]
        base = self.factors[target_idx, self.i0, None] * self.x - weight * self.m0
        pred = (
            (0.5 * d1.bfloat16()).bfloat16()
            if torch.is_autocast_enabled()
            else 0.5 * d1
        )
        # The original CPU scalar times BF16 contraction retains BF16 output;
        # a vector coefficient would otherwise promote it to FP32.
        correction = (
            (weight * pred.float()).bfloat16()
            if torch.is_autocast_enabled()
            else weight * pred
        )
        return base - correction

    def refresh(self, flow, latent, idx, indices):
        # Unselected predictions can be zero-filled by sparse Wan; never consume them.
        mt = self.pack(latent - self.sigmas[idx] * flow).index_select(0, indices)
        st = self.sigmas[idx]
        h = self.lambdas[idx] - self.lambdas[self.i0[indices], None]
        inverse_rk = self.inverse_rks[idx, self.i0[indices], self.i1[indices], None]
        rho = self.rhos[idx, self.i0[indices], self.i1[indices]]
        rho0 = rho[:, 0, None]
        rho1 = rho[:, 1, None]
        m0 = self.m0[indices]
        m1 = self.m1[indices]
        weight = self.weights[idx, self.i0[indices], None]
        base = self.factors[idx, self.i0[indices], None] * self.x[indices] - weight * m0
        d1 = (m1 - m0) * inverse_rk
        corr = (
            (rho0.bfloat16() * d1.bfloat16()).bfloat16()
            if torch.is_autocast_enabled()
            else rho0 * d1
        )
        corrected = base - weight * (corr + rho1 * (mt - m0))
        predicted = m0 + h * (m0 - m1) / (
            self.lambdas[self.i0[indices], None] - self.lambdas[self.i1[indices], None]
        )
        innovation = mt - predicted
        change = mt - m0
        self.x.index_copy_(0, indices, corrected)
        self.m1.index_copy_(0, indices, m0)
        self.m0.index_copy_(0, indices, mt)
        self.i1[indices] = self.i0[indices]
        self.i0[indices] = idx
        return innovation, change, h
