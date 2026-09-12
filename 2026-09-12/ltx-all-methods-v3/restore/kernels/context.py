"""Materialize a restore-first K/V trajectory and update sparse corrections.

The two fixed anchors are the Step10 observation and the known restored
endpoint. Only eligible live tokens acquire a new measured residual. This
operator keeps the existing BF16 interpolation/add/sub rounding boundaries.
It does not change query selection, attention, or restoration approximation.
"""

import torch, triton
import triton.language as tl


@triton.jit
def _restore_context(
    KL,
    VL,
    KR,
    VR,
    DK,
    DV,
    K,
    V,
    MAP,
    OK,
    OV,
    N,
    A,
    Q,
    D: tl.constexpr,
    SIZE,
    F,
    FIRST,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < SIZE
    channel = i % D
    token = (i // D) % N
    batch = i // (N * D)
    query = tl.load(MAP + token, valid, -1)
    live = valid & (query >= 0)
    kr = tl.load(KR + i, valid, 0).to(tl.float32)
    kl = tl.load(KL + i, valid, 0).to(tl.float32)
    vr = tl.load(VR + i, valid, 0).to(tl.float32)
    vl = tl.load(VL + i, valid, 0).to(tl.float32)
    # Match ATen scalar lerp's numerically stable branch and BF16 store.
    ak = (
        tl.where(F < 0.5, tl.fma(F, kl - kr, kr), tl.fma(-(1.0 - F), kl - kr, kl))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    av = (
        tl.where(F < 0.5, tl.fma(F, vl - vr, vr), tl.fma(-(1.0 - F), vl - vr, vl))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    residual_offset = (batch * A + token) * D + channel
    skipped = valid & (token < A) & ~live
    dk = tl.load(DK + residual_offset, skipped, 0).to(tl.float32)
    dv = tl.load(DV + residual_offset, skipped, 0).to(tl.float32)
    live_offset = (batch * Q + query) * D + channel
    k = tl.load(K + live_offset, live, 0).to(tl.float32)
    v = tl.load(V + live_offset, live, 0).to(tl.float32)
    tl.store(OK + i, tl.where(live, k, ak + dk), valid)
    tl.store(OV + i, tl.where(live, v, av + dv), valid)
    update = live & (token < A) & (query >= FIRST)
    tl.store(DK + residual_offset, k - ak, update)
    tl.store(DV + residual_offset, v - av, update)


def materialize_restoration_context(
    kl, vl, kr, vr, dk, dv, k, v, query_map, out_k, out_v, fraction, first=0
):
    b, n, h, c = kl.shape
    d = h * c
    _restore_context[(triton.cdiv(kl.numel(), 1024),)](
        kl,
        vl,
        kr,
        vr,
        dk,
        dv,
        k,
        v,
        query_map,
        out_k,
        out_v,
        n,
        dk.shape[1],
        k.shape[1],
        d,
        kl.numel(),
        fraction,
        first,
        1024,
        enable_fp_fusion=False,
    )


def query_inverse(positions, total):
    result = torch.full((total,), -1, device=positions.device, dtype=torch.int32)
    result.index_copy_(
        0,
        positions,
        torch.arange(len(positions), device=positions.device, dtype=torch.int32),
    )
    return result
