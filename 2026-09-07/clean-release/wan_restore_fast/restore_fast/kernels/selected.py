"""Gather selected tokens before the first block and scatter only the final head."""

import torch, triton
import triton.language as tl


@triton.jit
def _selected_norm(
    X, I, S, T, Y, D: tl.constexpr, L, FULL, EPS: tl.constexpr, K: tl.constexpr
):
    row = tl.program_id(0)
    c = tl.arange(0, K)
    valid = c < D
    batch = row // L
    token = tl.load(I + row % L)
    x = tl.load(X + (batch * FULL + token) * D + c, valid, 0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    v = tl.where(valid, x - mean, 0)
    var = tl.sum(v * v, 0) / D
    z = (x - mean) * tl.rsqrt(var + EPS)
    z = z.to(X.dtype.element_ty).to(tl.float32)
    offset = batch * D + c
    result = z * (1 + tl.load(S + offset, valid, 0)) + tl.load(T + offset, valid, 0)
    tl.store(Y + row * D + c, result, valid)


@triton.jit
def _selected_residual(X, I, Y, G, O, N, D: tl.constexpr, L, FULL, K: tl.constexpr):
    i = tl.program_id(0) * K + tl.arange(0, K)
    valid = i < N
    row = i // D
    batch = row // L
    token = tl.load(I + row % L, valid, 0)
    x = tl.load(X + (batch * FULL + token) * D + i % D, valid, 0).to(tl.float32)
    y = tl.load(Y + i, valid, 0).to(tl.float32)
    g = tl.load(G + batch * D + i % D, valid, 0)
    tl.store(O + i, x + y * g, valid)


def selected_norm(x, indices, module, scale, shift):
    b, full, d = x.shape
    l = len(indices)
    out = torch.empty((b, l, d), device=x.device, dtype=torch.bfloat16)
    _selected_norm[(b * l,)](
        x,
        indices,
        scale.contiguous(),
        shift.contiguous(),
        out,
        d,
        l,
        full,
        module.eps,
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return out


def selected_residual(x, indices, y, gate):
    out = torch.empty_like(y, dtype=torch.float32)
    _selected_residual[(triton.cdiv(out.numel(), 1024),)](
        x,
        indices,
        y,
        gate.contiguous(),
        out,
        out.numel(),
        x.shape[-1],
        len(indices),
        x.shape[1],
        1024,
        enable_fp_fusion=False,
    )
    return out
