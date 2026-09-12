"""LTX STG prefix sharing, without changing the configured guidance formula."""
import torch
from .kernels.ltx_arithmetic import rope,residual,modulate,rms,rms_approximate_reduction


class FusedLTXRMS:
    """Wan-style RMS pointwise fusion with the native ATen reduction.

    Preserve LTX's BF16 affine rounding and the native FP32 reduction/rsqrt.
    An optional predicate enables a single-kernel reduction after warmup only.
    That opt-in path changes FP32 reduction order and requires video validation.
    """
    def __init__(self,model,allow_single_kernel=None):
        from diffusers.models.normalization import RMSNorm
        self.original=[]
        for module in model.modules():
            if not isinstance(module,RMSNorm):continue
            old=module.forward;self.original.append((module,old))
            def run(x,m=module,original=old):
                if x.dtype!=torch.bfloat16 or m.training:return original(x)
                if allow_single_kernel is not None and allow_single_kernel():
                    return rms_approximate_reduction(x,m)
                return rms(x,m)
            module.forward=run
        assert self.original

    def close(self):
        for module,original in self.original:module.forward=original
        self.original.clear()


class SharedSTGPrefix:
    """The positive and STG branches are identical before perturbation block 19.

    Hooks retain the official transformer implementation and expose its original
    three-branch interface. The self-attention caches inside the prefix therefore
    also store only two unique branches. No sharing occurs after perturbation.
    """
    def __init__(self, model, first_perturbed_block=19):
        self.handles=[]
        self.first=first_perturbed_block
        self.checked=False
        self.enabled=True
        self.handles.append(model.register_forward_pre_hook(self.validate,with_kwargs=True))
        for i,block in enumerate(model.transformer_blocks[:first_perturbed_block]):
            self.handles.append(block.register_forward_pre_hook(self.before,with_kwargs=True))
            if i==first_perturbed_block-1:
                self.handles.append(block.register_forward_hook(self.after))

    def validate(self,module,args,kwargs):
        if self.checked:return
        x=args[0] if args else kwargs['hidden_states']
        assert x.shape[0]==3 and torch.equal(x[1],x[2])
        for name in ('encoder_hidden_states','timestep','encoder_attention_mask'):
            value=kwargs.get(name)
            if value is not None:assert value.shape[0]==3 and torch.equal(value[1],value[2]),name
        mask=kwargs['skip_layer_mask']
        assert bool((mask[:self.first]==1).all())
        assert float(mask[self.first,2])==0
        self.checked=True

    @staticmethod
    def trim(value):
        if torch.is_tensor(value) and value.ndim and value.shape[0]==3:return value[:2]
        if isinstance(value,tuple):return tuple(SharedSTGPrefix.trim(v) for v in value)
        return value

    def before(self,module,args,kwargs):
        if not self.enabled:return args,kwargs
        assert args[0].shape[0] in (2,3)
        return (args[0][:2],*args[1:]),{k:self.trim(v) for k,v in kwargs.items()}

    def after(self,module,args,output):
        if not self.enabled:return output
        return torch.cat((output,output[1:2]),dim=0)

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()


class FusedLTXArithmetic:
    """Same official block equations, with Wan-style elementwise fusion.

    RMS normalization, GEMMs, attention, guidance and the scheduler stay native.
    The kernels preserve the individual BF16 stores between multiply/add stages.
    """
    def __init__(self,model):
        self.original=[]
        for block in model.transformer_blocks:
            assert block.adaptive_norm=='single_scale_shift'
            original=block.forward;old_rope=block.attn1.apply_rotary_emb
            self.original.append((block,original,old_rope))
            block.forward=self.forward(block,original)
            block.attn1.apply_rotary_emb=lambda x,f,old=old_rope:rope(x,f) if x.dtype==torch.bfloat16 else old(x,f)

    @staticmethod
    def forward(block,original):
        def run(hidden_states,freqs_cis=None,attention_mask=None,encoder_hidden_states=None,
                encoder_attention_mask=None,timestep=None,cross_attention_kwargs=None,
                class_labels=None,skip_layer_mask=None,skip_layer_strategy=None):
            if hidden_states.dtype!=torch.bfloat16 or block.training:
                return original(hidden_states,freqs_cis,attention_mask,encoder_hidden_states,encoder_attention_mask,timestep,cross_attention_kwargs,class_labels,skip_layer_mask,skip_layer_strategy)
            assert hidden_states.ndim==3 and hidden_states.shape[1]>1
            assert class_labels is None and not cross_attention_kwargs
            from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
            assert skip_layer_strategy in (None,SkipLayerStrategy.AttentionValues)
            b=hidden_states.shape[0]
            values=block.scale_shift_table[None,None]+timestep.reshape(b,timestep.shape[1],6,-1)
            shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=values.unbind(dim=2)
            x=modulate(block.norm1(hidden_states),scale_msa,shift_msa)
            y=block.attn1(x,freqs_cis=freqs_cis,attention_mask=attention_mask,skip_layer_mask=skip_layer_mask,skip_layer_strategy=skip_layer_strategy)
            hidden_states=residual(hidden_states,y,gate_msa)
            y=block.attn2(hidden_states,freqs_cis=freqs_cis,encoder_hidden_states=encoder_hidden_states,attention_mask=encoder_attention_mask)
            hidden_states=hidden_states+y
            x=modulate(block.norm2(hidden_states),scale_mlp,shift_mlp)
            return residual(hidden_states,block.ff(x),gate_mlp)
        return run

    def close(self):
        for block,forward,old_rope in self.original:
            block.forward=forward;block.attn1.apply_rotary_emb=old_rope
        self.original.clear()
