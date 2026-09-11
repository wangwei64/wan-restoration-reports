"""Wan restoration/refresh semantics attached to the official LTX sampling loop."""
import math
import torch
from .sparse import SparseLTX
from .hardware import SharedSTGPrefix,FusedLTXArithmetic,FusedLTXRMS
from .policy import route,AdaptiveRefresh

class RestorationController:
    def __init__(self,restorer=None,warmup_ratio=.2,tolerance=.12,budget=.2,blend=1.,calibration=None,force_full=False,deduplicate=False,fused_context=False,fused_arithmetic=False,share_warmup=False,continuous_history=False,fused_rms=False,wan_evidence=False,history_integration='midpoint',rms_after_warmup_fast=False):
        self.restorer=restorer;self.ratio=warmup_ratio;self.tolerance=tolerance;self.budget=budget
        self.continuous_history=continuous_history;self.share_warmup=share_warmup;self.fused_arithmetic=fused_arithmetic;self.arithmetic=None
        self.fused_context=fused_context;self.deduplicate=deduplicate;self.hardware=None
        self.blend=blend;self.calibration=calibration or {};self.force_full=force_full
        self.fused_rms=fused_rms;self.rms=None
        self.wan_evidence=wan_evidence
        self.history_integration=history_integration
        self.rms_after_warmup_fast=rms_after_warmup_fast
        assert not rms_after_warmup_fast or fused_rms

    def volume(self,x):return x.transpose(1,2).reshape(x.shape[0],x.shape[-1],*self.shape)
    def tokens(self,x):return x.flatten(2).transpose(1,2)

    def begin(self,pipe,original_step,shape,steps):
        self.pipe=pipe;self.original_step=original_step;self.shape=shape;self.steps=steps
        self.warmup=math.ceil(steps*self.ratio);assert 2<=self.warmup<steps
        self.arithmetic=FusedLTXArithmetic(pipe.transformer) if self.fused_arithmetic else None
        allow_single_kernel=(lambda:self.i>=self.warmup) if self.rms_after_warmup_fast else None
        self.rms=FusedLTXRMS(pipe.transformer,allow_single_kernel=allow_single_kernel) if self.fused_rms else None
        self.hardware=SharedSTGPrefix(pipe.transformer) if self.deduplicate else None
        if self.hardware is not None:self.hardware.enabled=self.share_warmup
        self.sparse=SparseLTX(pipe.transformer,fused_context=self.fused_context,shared_prefix=self.deduplicate);self.original_forward=pipe.transformer.forward
        self.i=0;self.cache_bytes=0;self.previous=None;self.policy=None;self.endpoint=None
        self.minimum_query_padding_tokens=0
        pipe.transformer.forward=self.forward

    def forward(self,hidden_states,indices_grid,**kwargs):
        c=self.sparse;self.forward_kwargs=dict(kwargs);self.coords=indices_grid
        if self.i<self.warmup:
            c.mode='capture' if self.i==self.warmup-1 else 'native';c.slot='left'
            result=self.original_forward(hidden_states,indices_grid,**kwargs)
            self.branch_left=result[0].clone()
            return result
        due=self.policy.select(self.i);self.due=due
        if not bool(due.any()):return (self.branch_left,)
        ids=torch.nonzero(due).flatten()
        if ids.numel()==1:
            # The upstream block squeezes dimension 1, so a single-query batch
            # loses its token axis. Compute one adjacent support query too and
            # count it as real work, while keeping static ownership unchanged.
            extra=(int(ids[0])+1)%hidden_states.shape[1]
            due[extra]=True;self.policy.counts[extra]+=1;self.minimum_query_padding_tokens+=1
            self.policy.events[-1].update(active_tokens=2,active_ratio=2/hidden_states.shape[1])
            ids=torch.nonzero(due).flatten()
        c.indices=ids;c.mode='sparse'
        if self.fused_context:
            from .kernels.context import query_inverse
            c.context_query_map=query_inverse(ids,hidden_states.shape[1])
        if self.i==self.steps-1 and ids.numel()==hidden_states.shape[1]:c.mode='native'
        left=self.sigmas[self.warmup-1];right=self.sigmas[-2]
        c.fraction=max(0.,min(1.,(self.sigmas[self.i]-right)/(left-right)))
        selected_kwargs=dict(kwargs)
        if selected_kwargs['timestep'].shape[1]>1:selected_kwargs['timestep']=selected_kwargs['timestep'][:,ids]
        result=self.original_forward(hidden_states[:,ids],indices_grid[:,:,ids],**selected_kwargs)[0]
        base=torch.lerp(self.branch_right,self.branch_left,c.fraction)
        self.branch_delta[:,ids]=result-base[:,ids]
        output=base+self.branch_delta;output[:,ids]=result
        return (output,)

    def step(self,latents,noise_pred,current_timestep,conditioning_mask,t,extra_step_kwargs,t_eps=1e-6,stochastic_sampling=False):
        assert conditioning_mask is None and not stochastic_sampling,'This release supports deterministic text-to-video.'
        i=self.i;x=self.volume(latents);v=self.volume(noise_pred)
        if i<self.warmup:
            clean=x.float()-float(t)*v.float()
            if i==self.warmup-2:self.previous=clean.clone()
            result=self.original_step(latents,noise_pred,current_timestep,conditioning_mask,t,extra_step_kwargs,t_eps,stochastic_sampling)
            if i==self.warmup-1:
                self.sigmas=self.pipe.scheduler.timesteps.cpu().tolist()+[0.]
                if self.restorer is None:
                    restored=clean;error=torch.ones_like(clean[:,:1]);heat=error*.5;risk=error
                else:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        restored,confidence,aux=self.restorer(clean,float(t))
                    error=aux['expected_rgb_error'] if 'expected_rgb_error' in aux else aux['expected_latent_error'];heat=aux['heat'];risk=aux['latent_risk']
                self.endpoint=clean+self.blend*(restored.float()-clean)
                saliency=self.sparse.saliency(latents.shape[1],x.device)
                eligible,routing=route(clean,self.previous,error,heat,saliency,self.tolerance,self.calibration,risk)
                if self.force_full:eligible[:]=True
                self.policy=AdaptiveRefresh({'steps':self.steps,'sigmas':self.sigmas},self.warmup,self.endpoint,v,eligible,routing,0 if self.force_full else self.budget,
                                            warm_clean=clean,previous_clean=self.previous,wan_evidence=self.wan_evidence,history_integration=self.history_integration)
                self.ownership=self.tokens(self.policy.latent_mask(eligible))
                self.post=result.clone()
                if self.continuous_history:self.policy.initialize_bridge(self.volume(result))
                if bool(eligible.any()):
                    if self.hardware is not None:self.hardware.enabled=True
                    c=self.sparse;c.mode='capture';c.slot='right'
                    kw=dict(self.forward_kwargs);kw['timestep']=torch.full_like(kw['timestep'],self.sigmas[-2])
                    endpoint_tokens=self.tokens(self.endpoint).to(self.pipe.transformer.dtype)
                    self.branch_right=self.original_forward(endpoint_tokens.repeat(3,1,1),self.coords,**kw)[0]
                    self.branch_delta=torch.zeros_like(self.branch_left)
        else:
            if bool(self.due.any()):self.policy.update(x,v,i,self.due)
            result=self.original_step(latents,self.tokens(self.policy.velocity),current_timestep,conditioning_mask,t,extra_step_kwargs,t_eps,stochastic_sampling)
            if self.continuous_history:result=self.tokens(self.policy.continuous(self.volume(result),i,self.due))
            endpoint=self.tokens(self.endpoint)
            path=endpoint+(self.post-endpoint)*(self.sigmas[i+1]/self.sigmas[self.warmup])
            result=torch.where(self.ownership,result,path.to(result.dtype))
        self.i+=1
        return result

    def release_cache(self):
        self.cache_bytes=self.sparse.bytes();self.sparse.reset()
        for name in ('branch_left','branch_right','branch_delta','forward_kwargs','coords'):
            if hasattr(self,name):delattr(self,name)
        torch.cuda.empty_cache()

    def report(self):
        return {'warmup':self.warmup,'warmup_ratio':self.ratio,'warmup_actual_sigma':self.sigmas[self.warmup-1],
                'tolerance':self.tolerance,'restoration_blend':self.blend,'cache_precision':'bfloat16',
                'cache_gib':self.cache_bytes/2**30,'endpoint_full_evaluations':1 if bool(self.policy.eligible.any()) else 0,
                'checkpoint':getattr(self.restorer,'checkpoint_path',None),
                'checkpoint_step':getattr(self.restorer,'checkpoint_step',None),'calibration':self.calibration,
                'checkpoint_sha256':getattr(self.restorer,'checkpoint_sha256',None),
                'quality_training_steps':getattr(self.restorer,'quality_training_steps',0),
                'algorithm_version':'ltx128-iteration2',
                'continuous_reconstruction':f'two real x0 observations, log-SNR trend integrated with {self.history_integration}, exact native Euler bridge' if self.continuous_history else 'last real velocity',
                'shared_stg_prefix_blocks':19 if self.deduplicate else 0,
                'stg_sharing_during_warmup':bool(self.deduplicate and self.share_warmup),
                'fused_wan_context_kernel':self.fused_context,
                'fused_ltx_elementwise_arithmetic':self.fused_arithmetic,
                'fused_ltx_rms_normalization':self.fused_rms,
                'rms_reduction_after_warmup':'single-kernel FP32 reduction; not bit-exact to native' if self.rms_after_warmup_fast else 'native ATen FP32 reduction',
                'rms_reduction_warmup_and_initial_anchors':'native ATen FP32 reduction',
                'history_initialization':'last two real native warmup clean predictions',
                'minimum_query_padding_tokens':self.minimum_query_padding_tokens,
                'cache_scope':'current request only: last warmup, restored endpoint, last real residual',
                'guidance_approximation':'Global CFG/STG rescaling uses reconstructed branch outputs for unselected tokens',
                'force_full_regression':self.force_full,**self.policy.report()}

    def close(self):
        self.pipe.transformer.forward=self.original_forward
        self.sparse.close()
        if self.hardware is not None:self.hardware.close()
        if self.arithmetic is not None:self.arithmetic.close()
        if self.rms is not None:self.rms.close()
