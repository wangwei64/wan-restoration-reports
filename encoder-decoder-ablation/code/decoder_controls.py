"""S x C x A factorial under one nonzero-exit protocol and fixed compute code."""
from pathlib import Path
import sys,json,hashlib,copy,time,traceback
R=Path(__file__).resolve().parent;E=R.parent
sys.path[:0]=[str(R),str(E/'wan_paper_experiments_20260912'),'/root/autodl-tmp/wan_restore_fast']
from common import save,verify_frozen,gpu_pids,Guard
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from restore_fast.pipeline import RestorationPipeline
from restore_fast.config import GenerationConfig
from restore_fast.routing import clean_subject
from restore_fast.integrator import TokenUniPC
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def status(stage,**kw):save(R/'generation_status.json',dict(stage=stage,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw))

@torch.inference_mode()
def prefix(pipe,c,state_R,context_R,kv_align=False):
    exit_step=50;inside=True
    runner=pipe.wan;sparse=pipe.controller;sparse.reset();runner.set_request(c['prompt'],c['seed']);runner.encode_prompt();sparse.configure_prompt(runner,c['prompt'])
    latent=runner.initial_latents();scheduler=runner.scheduler();ref=load_file(str(Path(c['full_directory'])/'outputs.safetensors'))
    calls=[]
    hook=pipe.restorer.register_forward_pre_hook(lambda *_:calls.append(True))
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for idx in range(10):
                if idx==9:sparse.mode='capture';sparse.slot='left';sparse.observe=True
                flow=runner.predict(latent,idx,scheduler,sparse);clean=runner.clean_estimate(latent,flow,scheduler.sigmas[idx])
                if idx==9:warm=(copy.deepcopy(scheduler),latent.clone(),flow.clone())
                latent=runner.step(scheduler,flow,scheduler.timesteps[idx],latent)
                if idx==8:z9=clean.clone()
            sparse.observe=False;z10=clean.clone();post10=latent.clone()
            P=z10+(z10-z9)*((0-scheduler.sigmas[9])/(scheduler.sigmas[9]-scheduler.sigmas[8])).to(z10)
            f,h=pipe.features(z10)
            learned=None
            if state_R or context_R:
                learned,_,_=pipe.restorer(z10,f,h);learned=learned.float()
                assert torch.equal(learned[0].cpu(),ref['restored'])
            assert len(calls)==int(state_R or context_R)
            endpoint=learned if state_R else P
            context=learned if context_R else P
            probability=pipe.subject_head(z10,f,h).sigmoid();saliency=sparse.subject(probability)
            grid=(z10.shape[-3],z10.shape[-2]//2,z10.shape[-1]//2)
            prob=F.avg_pool3d(probability.float(),(1,2,2),(1,2,2)).clamp(.005,.995)
            attn=saliency.reshape(1,1,*grid).float().clamp(.005,.995)
            fused=torch.sigmoid(1.5*torch.logit(prob)+.5*torch.logit(attn))
            semantic=clean_subject(fused).repeat_interleave(2,-2).repeat_interleave(2,-1)
            active=ref['active'][None].to(z10);token_mask=active[...,::2,::2]
            checks=dict(entry9_exact=torch.equal(z9[0].cpu(),ref['x0_step9']),entry10_exact=torch.equal(z10[0].cpu(),ref['x0_step10']),
                semantic_exact=torch.equal(semantic[0].cpu(),ref['subject_binary']),
                probability_exact=torch.equal(fused.repeat_interleave(2,-2).repeat_interleave(2,-1)[0].cpu(),ref['subject_probability']),
                learned_G_call_count=len(calls)==int(state_R or context_R))
            assert all(checks.values()),checks
            sparse.partition(token_mask,semantic);sparse.enable_batched_cfg()
            core=TokenUniPC(warm,scheduler,post10);warm=None
            assert len(sparse.indices)>0
            sparse.mode='capture';sparse.slot='right';runner.predict(context,49,scheduler,sparse);sparse.mode='sparse'
            trace=[]
            if not inside: latent=active*post10+(1-active)*endpoint
            for idx in range(10,exit_step):
                due=token_mask.flatten().bool() if idx==exit_step-1 else ref['refresh_events'][idx-10].flatten().to(runner.device).bool()
                assert not bool((due&~token_mask.flatten().bool()).any())
                sparse.select(due)
                if len(sparse.indices):
                    flow=runner.predict(latent,idx,scheduler,sparse)
                    core.refresh(flow,latent,idx,sparse.indices)
                next_latent=core.unpack(core.dense(scheduler.sigmas[idx+1]))
                aligned=endpoint+(post10-endpoint)*(scheduler.sigmas[idx+1]/scheduler.sigmas[10])
                latent=active*next_latent+(1-active)*(aligned if inside else endpoint)
                trace.append(dict(step=idx+1,count=int(due.sum()),mask_sha256=digest(due),boundary_refresh=idx==exit_step-1))
            latent=active*next_latent+(1-active)*aligned
            raw_exit=active*next_latent+(1-active)*endpoint
            candidate_diff=float(torch.where(active.bool(),(raw_exit-latent).abs(),0.).max())
            expected=(1-active)*(post10-endpoint)*(scheduler.sigmas[exit_step]/scheduler.sigmas[10])
            formula_error=float(((latent-raw_exit)-expected).abs().max())
            assert candidate_diff==0 and formula_error<1e-5 and 0<=float(scheduler.sigmas[exit_step])<float(scheduler.sigmas[10])
            checkpoints=dict(aligned_exit=latent,raw_exit=raw_exit,active=active,endpoint=endpoint,post10=post10)
            replay=None
            if exit_step==30 and inside:
                original=load_file(str(E/'wan_decoder_alignment_20260915/prefix'/c['key']/'checkpoint.safetensors'))
                replay={k:torch.equal(v[0].cpu(),original[k]) for k,v in checkpoints.items()}
                assert all(replay.values()),replay
    finally:hook.remove()
    N=token_mask.numel();C=int(token_mask.sum())
    audit=dict(checks=checks,key=c['key'],state_R=state_R,context_R=context_R,G_calls=len(calls),exit_step=exit_step,inside_alignment=inside,
        sigma_x=float(scheduler.sigmas[10]),sigma_y=float(scheduler.sigmas[exit_step]),total_tokens=N,candidate_count=C,affected_tokens=N-C,
        candidate_exit_max_difference=candidate_diff,exit_formula_max_error=formula_error,shared_prefix=True,
        full_prefix_replay=replay,per_step=trace,actual_50_step_token_counts=[N]*10+[x['count'] for x in trace]+[N]*(50-exit_step),
        history_endpoint_sha256=digest(P),state_endpoint_sha256=digest(endpoint),context_endpoint_sha256=digest(context))
    d=R/'prefix'/c['key']/f'S{int(state_R)}_C{int(context_R)}_A{int(kv_align)}';d.mkdir(parents=True)
    save_file({k:v[0].detach().cpu().contiguous() for k,v in checkpoints.items()},str(d/'checkpoint.safetensors'));save(d/'audit.json',audit)
    sparse.reset();return latent.clone(),raw_exit.clone(),audit



import contextlib
sys.path.insert(0,'/root/autodl-tmp/Wan2.1')
import restore_fast.wan.controller as controller_module
@contextlib.contextmanager
def kv_mode(enabled):
    original=controller_module.materialize_restoration_context
    audit=dict(enabled=enabled,calls=0,requested_min=1.,requested_max=0.,effective_min=1.,effective_max=0.)
    def wrapped(*args,**kw):
        assert not kw and len(args)==13
        values=list(args);requested=float(values[11]);values[11]=requested if enabled else 0.
        audit['calls']+=1
        audit['requested_min']=min(audit['requested_min'],requested);audit['requested_max']=max(audit['requested_max'],requested)
        audit['effective_min']=min(audit['effective_min'],values[11]);audit['effective_max']=max(audit['effective_max'],values[11])
        return original(*values)
    controller_module.materialize_restoration_context=wrapped
    try:yield audit
    finally:controller_module.materialize_restoration_context=original
