"""Training-free native LTX 0.9.6 adapters. See ADAPTATION.md for provenance.

Full-compute paths call the untouched frozen native forward. All native CFG/STG
branches share a cache decision but retain independent residuals in the batch.
"""
import math
import numpy as np
import torch
from ltx_video.models.transformers.transformer3d import Transformer3DModelOutput


def packed(value, kwargs):
    return Transformer3DModelOutput(sample=value) if kwargs.get('return_dict', True) else (value,)


class NaviCacheLTX:
    """Wan official raw-input/raw-output NaviCache, batched for native LTX."""
    def __init__(self, model, threshold, steps=50, warmup_steps=10):
        assert not model.training and threshold >= 0
        self.model=model; self.threshold=threshold; self.steps=steps; self.warmup=warmup_steps
        self.original=model.forward; model.forward=self.forward
        self.i=0; self.accum=0.; self.ratio=0.; self.uncertainty=1.
        self.previous_input=self.exact_input=self.exact_output=self.residual=None
        self.events=[]

    def forward(self, *args, **kwargs):
        assert self.i < self.steps
        raw=(args[0] if args else kwargs['hidden_states']).clone()
        assert raw.shape[0] == 3
        # Positive branch is 1 in native CFG/STG (Wan invokes it first separately).
        x=raw[1]
        forced=self.i < self.warmup or self.i == self.steps-1 or self.threshold == 0
        increment=None
        if forced:
            full=True; self.accum=0.
        else:
            drift=(x-self.previous_input).abs().mean()
            norm=self.exact_output.abs().mean()
            increment=float(self.ratio*(drift/norm))
            assert math.isfinite(increment)
            self.accum+=increment
            full=self.accum>=self.threshold
        before=self.accum
        self.previous_input=x.clone()
        if full:
            self.accum=0.
            output=self.original(*args,**kwargs)
            y=(output[0] if isinstance(output,tuple) else output.sample)
            if self.exact_output is not None:
                dy=(y[1]-self.exact_output).abs().mean()
                dx=(x-self.exact_input).abs().mean()
                z=dy/(dx+1e-8)
                if self.i < self.warmup:
                    self.ratio=z; self.uncertainty=1.
                else:
                    self.uncertainty+=.05
                    gain=self.uncertainty/(self.uncertainty+.05+1e-8)
                    self.ratio=self.ratio+gain*(z-self.ratio)
                    self.uncertainty=(1-gain)*self.uncertainty
            self.exact_input=x.clone(); self.exact_output=y[1].clone()
            self.residual=y-raw
        else:
            output=packed(raw+self.residual,kwargs)
        self.events.append({'step':self.i+1,'full_compute':full,'forced':forced,
            'predicted_increment':increment,'accumulated_before_reset':before,
            'state_ratio':float(self.ratio),'uncertainty':self.uncertainty})
        self.i+=1
        if self.i==self.steps:self.clear()
        return output

    def clear(self):
        self.previous_input=self.exact_input=self.exact_output=self.residual=None

    def report(self):
        assert self.i==self.steps
        full=sum(e['full_compute'] for e in self.events)
        return {'algorithm':'NaviCache official Wan algorithm, native LTX adaptation',
            'threshold':self.threshold,'process_noise':.05,'measurement_noise':.05,
            'full_warmup_steps':self.warmup,'final_step_full':True,
            'full_transformer_steps':full,'cached_transformer_steps':self.steps-full,
            'cache_trace':self.events,'residual_boundary':'raw transformer input to final transformer output',
            'restoration_network_used':False,'custom_hardware_fusions':False}

    def close(self):
        self.model.forward=self.original; self.clear()


class _SenHit(Exception):pass


class SenCacheLTX:
    """Official LTX SenCache residual boundary and sensitivity gate, native hooks."""
    SCALE=1015.9685034488028
    def __init__(self,model,threshold,sensitivity_path,steps=50,warmup_steps=10,max_skips=4):
        assert not model.training and threshold>=0
        self.model=model; self.threshold=threshold; self.steps=steps; self.warmup=warmup_steps; self.K=max_skips
        data=np.load(sensitivity_path)
        self.times=data['timesteps'];self.jx=data['J_x_norm'];self.jt=data['J_t_norm']
        assert all(np.isfinite(a).all() for a in [self.times,self.jx,self.jt])
        self.original=model.forward;model.forward=self.forward
        self.handles=[model.transformer_blocks[0].register_forward_pre_hook(self.before_block,with_kwargs=True),
            model.proj_out.register_forward_pre_hook(self.before_projection)]
        self.i=0;self.skips=0;self.cached_z=self.cached_t=self.residual=self.projected=None
        self.jindex=None;self.replaying=False;self.events=[]

    def before_block(self,block,args,kwargs):
        self.projected=args[0] if args else kwargs['hidden_states']
        if not self.full:raise _SenHit()

    def before_projection(self,module,args):
        if not self.replaying:self.residual=(args[0]-self.projected).detach()

    def forward(self,*args,**kwargs):
        assert self.i < self.steps
        raw=args[0] if args else kwargs['hidden_states']
        assert raw.shape[0]==3
        # Native time is sigma in [0,1]. Diffusers/SenCache lookup uses [0,1000].
        time_value=kwargs['timestep'][1].reshape(-1)[0].float()*1000.
        z=raw[1]
        forced=self.i<self.warmup or self.i==self.steps-1 or self.threshold==0
        error=dx=dt=None
        self.full=True
        if not forced and self.cached_z is not None:
            # FP32 accumulation avoids BF16 norm overflow; native latent is BF16.
            dx=float(torch.norm((z-self.cached_z).float()))
            dt=float((time_value-self.cached_t).abs())
            error=float(self.jx[self.jindex]*dx+self.jt[self.jindex]*dt)
            self.full=not(error < self.threshold*self.SCALE and self.skips < self.K)
        self.skips=0 if self.full else self.skips+1
        try:
            result=self.original(*args,**kwargs)
        except _SenHit:
            self.replaying=True
            try: result=packed(self.model.proj_out(self.projected+self.residual),kwargs)
            finally:self.replaying=False
        if self.full:
            self.cached_z=z.detach().clone();self.cached_t=time_value.detach().clone()
            self.jindex=int(np.argmin(np.abs(self.times-float(time_value))))
        self.events.append({'step':self.i+1,'full_compute':self.full,'forced':forced,
            'sensitivity_error':error,'threshold_scaled':self.threshold*self.SCALE,
            'delta_z_norm':dx,'delta_t':dt,'lookup_index':self.jindex,
            'lookup_t':float(self.times[self.jindex]),'actual_t':float(time_value),'consecutive_skips':self.skips})
        self.i+=1;self.projected=None
        if self.i==self.steps:self.clear()
        return result

    def clear(self):
        self.cached_z=self.cached_t=self.residual=self.projected=None

    def report(self):
        assert self.i==self.steps
        full=sum(e['full_compute'] for e in self.events)
        return {'algorithm':'SenCache official LTX algorithm, native 0.9.6 adaptation',
            'threshold':self.threshold,'threshold_scaling_factor':self.SCALE,'max_consecutive_skips':self.K,
            'sensitivity_source':'official published sensitivity_ltx.npz, unchanged (0.9.1)',
            'full_warmup_steps':self.warmup,'final_step_full':True,
            'full_transformer_steps':full,'cached_transformer_steps':self.steps-full,'cache_trace':self.events,
            'residual_boundary':'post output norm/modulation, pre output projection',
            'restoration_network_used':False,'custom_hardware_fusions':False}

    def close(self):
        self.model.forward=self.original
        for h in self.handles:h.remove()
        self.handles.clear();self.clear()
