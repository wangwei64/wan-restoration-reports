"""Selected-path norm, modulation, residual and normalized RoPE primitives."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(
    X,
    W,
    B,
    S,
    T,
    Y,
    N,
    D: tl.constexpr,
    L,
    EPS: tl.constexpr,
    RMS: tl.constexpr,
    AFFINE: tl.constexpr,
    MOD: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, K)
    mask = c < D
    x = tl.load(X + row * D + c, mask, 0).to(tl.float32)
    if RMS:
        mean2 = tl.sum(x * x, 0) / D
        z = x * tl.rsqrt(mean2 + EPS)
        z = z.to(X.dtype.element_ty).to(tl.float32)
        z = z * tl.load(W + c, mask, 0).to(tl.float32)
    else:
        mean = tl.sum(x, 0) / D
        v = tl.where(mask, x - mean, 0)
        var = tl.sum(v * v, 0) / D
        z = (x - mean) * tl.rsqrt(var + EPS)
        if AFFINE:
            z = z * tl.load(W + c, mask, 0) + tl.load(B + c, mask, 0)
        z = z.to(X.dtype.element_ty).to(tl.float32)
        if MOD:
            offset = (row // L) * D + c
            z = z * (1 + tl.load(S + offset, mask, 0)) + tl.load(T + offset, mask, 0)
    tl.store(Y + row * D + c, z, mask)


@triton.jit
def _rms_rope(
    X,
    W,
    F,
    I,
    Y,
    D: tl.constexpr,
    L,
    EPS: tl.constexpr,
    HEAD: tl.constexpr,
    SELECTED: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, K)
    mask = c < D
    x = tl.load(X + row * D + c, mask, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    z = (x * inv).to(X.dtype.element_ty).to(tl.float32) * tl.load(W + c, mask, 0)
    neighbor = tl.load(X + row * D + (c ^ 1), mask, 0).to(tl.float32)
    other = (
        (neighbor * inv).to(X.dtype.element_ty).to(tl.float32)
        * tl.load(W + (c ^ 1), mask, 0)
    ).to(tl.float64)
    z = z.to(tl.float64)
    token = row % L
    if SELECTED:
        token = tl.load(I + token)
    pos = token * HEAD + (c % HEAD // 2) * 2
    re = tl.load(F + pos, mask, 0)
    im = tl.load(F + pos + 1, mask, 0)
    sign = tl.where(c % 2 == 0, -1.0, 1.0).to(tl.float64)
    result = (z * re + sign * other * im).to(tl.float32)
    tl.store(Y + row * D + c, result, mask)


@triton.jit
def _residual(X, Y, G, O, N, D: tl.constexpr, L, GATE: tl.constexpr, K: tl.constexpr):
    i = tl.program_id(0) * K + tl.arange(0, K)
    m = i < N
    x = tl.load(X + i, m, 0).to(tl.float32)
    y = tl.load(Y + i, m, 0).to(tl.float32)
    if GATE:
        y = y * tl.load(G + (i // (L * D)) * D + i % D, m, 0)
    tl.store(O + i, x + y, m)


def normalize(x, module, scale=None, shift=None, rms=False):
    x = x.contiguous()
    d = x.shape[-1]
    n = x.numel() // d
    # RMS intermediate is used by rotary math and retains FP32 output.
    y = torch.empty_like(x, dtype=torch.float32 if rms else torch.bfloat16)
    affine = getattr(module, "elementwise_affine", False)
    w = module.weight if rms or affine else x
    b = module.bias if affine else x
    # Modulations are views with a six-way stride. Make the small vectors contiguous.
    scale = scale.contiguous() if scale is not None else None
    shift = shift.contiguous() if shift is not None else None
    _norm[(n,)](
        x,
        w,
        b,
        scale if scale is not None else x,
        shift if shift is not None else x,
        y,
        n,
        d,
        x.shape[1],
        module.eps,
        rms,
        affine,
        scale is not None,
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return y


def residual(x, y, gate=None):
    o = torch.empty_like(x, dtype=torch.float32)
    g = gate.contiguous() if gate is not None else x
    _residual[(triton.cdiv(x.numel(), 1024),)](
        x,
        y,
        g,
        o,
        x.numel(),
        x.shape[-1],
        x.shape[1],
        gate is not None,
        1024,
        enable_fp_fusion=False,
    )
    return o


_freq_cache = {}


def rotary_frequencies(head_dim, grid_sizes, freqs):
    grid = tuple(grid_sizes[0].tolist())
    key = (grid, head_dim, freqs.data_ptr())
    if key not in _freq_cache:
        f, h, w = grid
        c = head_dim // 2
        ft, fh, fw = freqs.split([c - 2 * (c // 3), c // 3, c // 3], 1)
        full = torch.cat(
            [
                ft[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                fh[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                fw[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            -1,
        ).reshape(-1, c)
        _freq_cache[key] = torch.view_as_real(full).contiguous()
    return _freq_cache[key]


def normalized_rotary(x, norm, head_dim, grid_sizes, freqs, indices=None):
    x = x.contiguous()
    d = x.shape[-1]
    y = torch.empty_like(x, dtype=torch.bfloat16)
    table = rotary_frequencies(head_dim, grid_sizes, freqs)
    _rms_rope[(x.numel() // d,)](
        x,
        norm.weight,
        table,
        indices if indices is not None else x,
        y,
        d,
        x.shape[1],
        norm.eps,
        head_dim,
        indices is not None,
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return y.view(*x.shape[:2], d // head_dim, head_dim)
