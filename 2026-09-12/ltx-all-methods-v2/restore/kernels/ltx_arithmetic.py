"""Wan-style elementwise fusion preserving LTX's BF16 rounding boundaries."""
import torch,triton
import triton.language as tl

@triton.jit
def _rope(X,C,S,Y,N,D:tl.constexpr,FB:tl.constexpr,SIZE,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<SIZE
    channel=i%D;row=i//D;batch=row//N;token=row%N
    table=((batch if FB else 0)*N+token)*D+channel
    x=tl.load(X+i,valid,0).to(tl.float32)
    neighbor=tl.load(X+(i-channel)+(channel^1),valid,0).to(tl.float32)
    rotated=tl.where(channel%2==0,-neighbor,neighbor)
    cosine=tl.load(C+table,valid,0).to(tl.float32);sine=tl.load(S+table,valid,0).to(tl.float32)
    a=(x*cosine).to(tl.bfloat16).to(tl.float32)
    b=(rotated*sine).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,a+b,valid)

def rope(x,freqs):
    assert x.dtype==torch.bfloat16 and x.is_contiguous()
    c,s=freqs;assert c.dtype==x.dtype and c.is_contiguous() and s.is_contiguous()
    assert c.shape[0] in (1,x.shape[0]) and c.shape[1:]==x.shape[1:]
    out=torch.empty_like(x)
    _rope[(triton.cdiv(x.numel(),1024),)](x,c,s,out,x.shape[1],x.shape[2],c.shape[0]!=1,x.numel(),1024,enable_fp_fusion=False)
    return out

@triton.jit
def _residual(X,Y,G,O,N,D:tl.constexpr,GN:tl.constexpr,SIZE,GATE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<SIZE
    x=tl.load(X+i,valid,0).to(tl.float32);y=tl.load(Y+i,valid,0).to(tl.float32)
    if GATE:
        row=i//D;batch=row//N;token=row%N
        gi=(batch*GN+(token if GN!=1 else 0))*D+i%D
        gate=tl.load(G+gi,valid,0).to(tl.float32)
        y=(y*gate).to(tl.bfloat16).to(tl.float32)
    tl.store(O+i,x+y,valid)

def residual(x,y,gate=None):
    assert x.dtype==y.dtype==torch.bfloat16
    x=x.contiguous();y=y.contiguous();out=torch.empty_like(x)
    g=gate.contiguous() if gate is not None else x
    _residual[(triton.cdiv(x.numel(),1024),)](x,y,g,out,x.shape[1],x.shape[2],g.shape[1],x.numel(),gate is not None,1024,enable_fp_fusion=False)
    return out

@triton.jit
def _modulate(X,S,T,O,N,D:tl.constexpr,SN:tl.constexpr,SIZE,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<SIZE
    row=i//D;batch=row//N;token=row%N
    si=(batch*SN+(token if SN!=1 else 0))*D+i%D
    x=tl.load(X+i,valid,0).to(tl.float32)
    scale=tl.load(S+si,valid,0).to(tl.float32);shift=tl.load(T+si,valid,0).to(tl.float32)
    factor=(1+scale).to(tl.bfloat16).to(tl.float32)
    scaled=(x*factor).to(tl.bfloat16).to(tl.float32)
    tl.store(O+i,scaled+shift,valid)

def modulate(x,scale,shift):
    assert x.dtype==scale.dtype==shift.dtype==torch.bfloat16
    x=x.contiguous();scale=scale.contiguous();shift=shift.contiguous();out=torch.empty_like(x)
    _modulate[(triton.cdiv(x.numel(),1024),)](x,scale,shift,out,x.shape[1],x.shape[2],scale.shape[1],x.numel(),1024,enable_fp_fusion=False)
    return out


@triton.jit
def _rms(X,W,B,Y,D:tl.constexpr,EPS:tl.constexpr,AFFINE:tl.constexpr,BIAS:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,K);valid=c<D
    x=tl.load(X+row*D+c,valid,0).to(tl.float32)
    variance=tl.sum(x*x,0)/D
    # Diffusers RMSNorm first computes in FP32, casts before BF16 affine,
    # and rounds weight multiplication before an optional bias addition.
    z=(x*tl.rsqrt(variance+EPS)).to(tl.bfloat16).to(tl.float32)
    if AFFINE:
        z=(z*tl.load(W+c,valid,0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    if BIAS:z=z+tl.load(B+c,valid,0).to(tl.float32)
    tl.store(Y+row*D+c,z,valid)


def rms_approximate_reduction(x,module):
    assert x.dtype==torch.bfloat16
    assert module.weight is None or module.weight.dtype==x.dtype
    assert module.bias is None or module.bias.dtype==x.dtype
    x=x.contiguous();out=torch.empty_like(x);d=x.shape[-1]
    assert tuple(module.dim)==(d,)
    _rms[(x.numel()//d,)](x,module.weight if module.weight is not None else x,module.bias if module.bias is not None else x,out,d,module.eps,module.weight is not None,module.bias is not None,triton.next_power_of_2(d),enable_fp_fusion=False)
    return out


@triton.jit
def _square_fp32(X,Y,N,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    x=tl.load(X+i,valid,0).to(tl.float32)
    tl.store(Y+i,x*x,valid)


@triton.jit
def _rms_apply(X,R,W,B,Y,N,D:tl.constexpr,AFFINE:tl.constexpr,BIAS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    x=tl.load(X+i,valid,0).to(tl.float32);inv=tl.load(R+i//D,valid,0)
    z=(x*inv).to(tl.bfloat16).to(tl.float32)
    if AFFINE:z=(z*tl.load(W+i%D,valid,0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    if BIAS:z=z+tl.load(B+i%D,valid,0).to(tl.float32)
    tl.store(Y+i,z,valid)


def rms(x,module):
    """Fuse pointwise work but retain ATen's exact FP32 reduction/rsqrt.

    The single-kernel reduction above changed rare BF16 ties and failed the
    complete sampling guard. This path keeps the original reduction order.
    """
    assert x.dtype==torch.bfloat16
    assert module.weight is None or module.weight.dtype==x.dtype
    assert module.bias is None or module.bias.dtype==x.dtype
    x=x.contiguous();out=torch.empty_like(x);d=x.shape[-1]
    assert tuple(module.dim)==(d,)
    squared=torch.empty_like(x,dtype=torch.float32)
    _square_fp32[(triton.cdiv(x.numel(),1024),)](x,squared,x.numel(),1024,enable_fp_fusion=False)
    inv=torch.rsqrt(squared.mean(-1,keepdim=True)+module.eps)
    _rms_apply[(triton.cdiv(x.numel(),1024),)](x,inv,module.weight if module.weight is not None else x,module.bias if module.bias is not None else x,out,x.numel(),d,module.weight is not None,module.bias is not None,1024,enable_fp_fusion=False)
    return out
