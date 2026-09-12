"""Restoration-relative routing and adaptive refresh on the native LTX RF timeline."""
import math
import torch
from torch.nn import functional as F


def pool(x): return x.float()
def expand(x): return x


def route(clean,previous,error,heat,saliency,tolerance,calibration,latent_risk=None):
    shape=(1,1,clean.shape[2],clean.shape[3],clean.shape[4])
    detail=pool(heat).clamp(0,1)
    semantic=saliency.reshape(shape).clamp(0,1)
    drift=pool((clean-previous).abs().mean(1,keepdim=True))
    motion=(drift/drift.mean().clamp_min(1e-6)).clamp(0,3)
    if 'predicted_error_knots' in calibration:
        knots=torch.tensor(calibration['predicted_error_knots'],device=error.device)
        values=torch.tensor(calibration['calibrated_error'],device=error.device)
        upper=torch.searchsorted(knots,error.flatten().float()).clamp(1,len(knots)-1);lower=upper-1
        fraction=((error.flatten()-knots[lower])/(knots[upper]-knots[lower]).clamp_min(1e-8)).clamp(0,1)
        expected=(values[lower]+fraction*(values[upper]-values[lower])).view_as(error)
    else:
        expected=pool(error)*calibration.get('error_multiplier',1.0)+calibration.get('error_offset',0.0)
    threshold=tolerance*(1-.25*semantic)/(1+.15*detail+.15*motion)
    risk=(expected+calibration.get('routing_motion_floor',.003)*motion)/threshold.clamp_min(1e-6)
    risk=torch.maximum(risk,.75*F.max_pool3d(risk,3,1,1))
    eligible=(risk>1).flatten()
    return eligible,dict(risk=risk.flatten(),heat=detail.flatten(),saliency=semantic.flatten(),
                         motion=motion.flatten(),grid=shape,threshold=threshold,
                         predicted_error=expected,
                         latent_risk=(pool(latent_risk).flatten() if latent_risk is not None else risk.flatten().clamp(0,1)))


class AdaptiveRefresh:
    def __init__(self,request,warmup,endpoint,initial_velocity,eligible,routing,budget,warm_clean=None,previous_clean=None,wan_evidence=False,history_integration='midpoint'):
        self.request=request;self.warmup=warmup;self.endpoint=endpoint;self.velocity=initial_velocity.clone()
        self.eligible=eligible;self.routing=routing;self.budget=budget
        self.wan_evidence=wan_evidence
        assert history_integration in ('midpoint','native_euler')
        self.history_integration=history_integration
        n=eligible.numel();device=eligible.device
        self.last=torch.full((n,),warmup-1,device=device,dtype=torch.long)
        self.previous=self.last.clone()
        self.delta=self.pack(endpoint).new_zeros(n,endpoint.shape[1])
        self.previous_delta=self.delta.clone()
        # Both observations already exist from full warmup. Zero-initializing
        # this history would misclassify the restorer correction itself as a new
        # surprise at the first refresh, and trigger needless dense evaluations.
        if warm_clean is not None:self.delta=self.pack(warm_clean-endpoint).clone()
        if previous_clean is not None:
            self.previous_delta=self.pack(previous_clean-endpoint).clone()
            self.previous.fill_(warmup-2)
        else:self.previous_delta=self.delta.clone()
        self.innovation=torch.zeros(n,device=device);self.evidence=torch.zeros(n,device=device)
        self.surprise=torch.zeros(n,device=device);self.counts=torch.zeros(n,device=device,dtype=torch.int32)
        self.lambdas=torch.tensor([math.log(max(1-s,1e-6))-math.log(max(s,1e-6)) for s in request['sigmas']],device=device)
        self.events=[]

    @staticmethod
    def pack(x):
        b,c,t,h,w=x.shape
        assert b==1
        return x.permute(0,2,3,4,1).reshape(-1,c)

    def latent_mask(self,token_mask):
        return expand(token_mask.reshape(self.routing['grid']))

    def initialize_bridge(self,post_warmup):
        # Wan reconstructs skipped states continuously from two real x0
        # observations. LTX's official solver is Euler, so retain its exact next
        # point as the bridge instead of importing Wan's UniPC corrector wholesale.
        self.bridge=self.pack(post_warmup).clone()
        self.sigma_tensor=torch.tensor(self.request['sigmas'],device=self.bridge.device)
        if self.history_integration=='native_euler':
            # Integrate the two-real-x0 linear trend on the actual native Euler
            # grid. The first real Euler bridge remains unchanged. This avoids
            # evaluating the trend half a step ahead of LTX's left-point rule.
            sigmas=self.request['sigmas'];count=len(sigmas)
            lambdas=self.lambdas.cpu().tolist()
            moments=torch.zeros(count,count,dtype=torch.float32)
            for last in range(self.warmup-1,count-2):
                moment=0.
                for j in range(last+1,count-1):
                    ratio=sigmas[j+1]/sigmas[j]
                    moment=ratio*moment+(1-ratio)*(lambdas[j]-lambdas[last])
                    moments[j+1,last]=moment
            self.euler_moments=moments.to(self.bridge.device)

    def continuous(self,native_next,i,due):
        """Two-real-observation reconstruction, exact on every real Euler step.

        A real evaluation retains the official Euler next state. Beyond that
        bridge, the Wan x0/log-SNR linear trend supplies the same bh2 midpoint
        predictor principle. Reconstructed states never enter the observations.
        """
        packed=self.pack(native_next)
        self.bridge[due]=packed[due]
        skipped=self.eligible & ~due
        if not bool(skipped.any()) or i==self.request['steps']-1:return native_next
        ids=torch.nonzero(skipped).flatten();last=self.last[ids];previous=self.previous[ids]
        anchor=last+1;target=i+1
        ratio=(self.sigma_tensor[target]/self.sigma_tensor[anchor])[:,None]
        interval=(self.lambdas[last]-self.lambdas[previous]).clamp_min(1e-6)[:,None]
        slope=(self.delta[ids]-self.previous_delta[ids])/interval
        clean=self.pack(self.endpoint)[ids]+self.delta[ids]
        if self.history_integration=='native_euler':
            predicted=ratio*self.bridge[ids]+(1-ratio)*clean+self.euler_moments[target,last,None]*slope
        else:
            midpoint=(.5*(self.lambdas[target]+self.lambdas[anchor])-self.lambdas[last])[:,None]
            predicted=ratio*self.bridge[ids]+(1-ratio)*(clean+midpoint*slope)
        packed[ids]=predicted
        b,c,t,h,w=native_next.shape
        return packed.reshape(b,t,h,w,c).permute(0,4,1,2,3).contiguous()

    def select(self,i):
        if self.budget<=0:
            due=self.eligible.clone()
        else:
            h=(self.lambdas[i]-self.lambdas[self.last]).clamp_min(0)
            magnitude=self.delta.square().mean(-1).sqrt()
            scale=magnitude+.25*self.routing['latent_risk'].clamp(0,1)+.02
            neighbor=F.max_pool3d((self.surprise*torch.exp(-h)).reshape(self.routing['grid']),3,1,1).flatten()
            importance=1+3*self.routing['saliency']+.5*self.routing['heat']
            score=importance*h.square()*(self.routing['risk']/(1+self.evidence)+self.innovation/scale+neighbor/scale)
            due=self.eligible & (score>=self.budget)
        if i==self.request['steps']-1: due=self.eligible.clone()
        self.counts+=due.to(torch.int32)
        self.events.append({'step':i+1,'active_tokens':int(due.sum()),'active_ratio':float(due.float().mean())})
        return due

    def update(self,x,velocity,i,due):
        ids=torch.nonzero(due).flatten()
        sigma=self.request['sigmas'][i]
        observed=self.pack(x.float()-sigma*velocity.float()-self.endpoint)[ids]
        old=self.delta[ids]
        h=(self.lambdas[i]-self.lambdas[self.last[ids]]).clamp_min(1e-5)
        interval=(self.lambdas[self.last[ids]]-self.lambdas[self.previous[ids]]).clamp_min(1e-5)
        slope=(old-self.previous_delta[ids])/interval[:,None]
        # Do not invent a trend before two distinct real model observations.
        slope=torch.where((self.last[ids]!=self.previous[ids])[:,None],slope,torch.zeros_like(slope))
        surprise=(observed-(old+h[:,None]*slope)).square().mean(-1).sqrt()
        blend=1-torch.exp(-h)
        self.innovation[ids]=(1-blend)*self.innovation[ids]+blend*surprise/h.square()
        self.surprise[ids]=surprise
        change=(observed-old).square().mean(-1).sqrt()
        evidence_scale=.25*self.routing['latent_risk'][ids].clamp(0,1)+.02 if self.wan_evidence else .1
        self.evidence[ids]+=h*torch.exp(-change/evidence_scale)
        self.previous_delta[ids]=old;self.delta[ids]=observed
        self.previous[ids]=self.last[ids];self.last[ids]=i
        self.velocity=torch.where(self.latent_mask(due),velocity,self.velocity)

    def report(self):
        return {'refresh_events':self.events,'eligible_ratio':float(self.eligible.float().mean()),
                'routing_expected_error_mean':float(self.routing['predicted_error'].mean()),
                'routing_expected_error_quantiles':torch.quantile(self.routing['predicted_error'].float(),torch.tensor([.1,.5,.9],device=self.endpoint.device)).tolist(),
                'routing_threshold_mean':float(self.routing['threshold'].mean()),
                'stability_evidence_normalization':'Wan per-token latent risk prior + 0.02' if self.wan_evidence else 'legacy LTX fixed 0.1',
                'mean_active_ratio':sum(s['active_ratio'] for s in self.events)/max(1,len(self.events)),
                'budget':self.budget,'solver':('native LTX RF Euler at real observations; two-real-x0/log-SNR continuous reconstruction beyond each exact Euler bridge' if hasattr(self,'bridge') else 'native LTX RF Euler with last real velocity for skipped tokens'),
                'history_integration':self.history_integration,
                'pseudo_observations_stored':False}
