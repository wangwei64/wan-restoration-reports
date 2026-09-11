"""ProfilingDiT equations 8/9/14/15, native LTX block-list residual caching."""
import torch


def groups(indices):
    result=[]
    for i in sorted(set(indices)):
        if result and i==result[-1][-1]+1:result[-1].append(i)
        else:result.append([i])
    return result


def refresh_schedule(steps=50,warmup=10,max_interval=12,min_interval=3):
    out=set(range(warmup));i=warmup
    while i<steps:
        out.add(i)
        interval=round(max_interval-(max_interval-min_interval)*(i-warmup)/(steps-1-warmup))
        i+=max(1,interval)
    out.add(steps-1)
    return sorted(out)


class ProfilingDiTLTX:
    def __init__(self,model,background_blocks,steps=50,warmup_steps=10,max_interval=12,min_interval=3):
        assert not model.training
        self.model=model;self.steps=steps;self.i=0;self.background=sorted(background_blocks)
        assert all(0<=x<len(model.transformer_blocks) for x in self.background)
        self.groups=groups(self.background);self.ends={g[0]:g[-1] for g in self.groups}
        self.end_set={g[-1] for g in self.groups};self.bg=set(self.background)
        self.schedule=refresh_schedule(steps,warmup_steps,max_interval,min_interval)
        self.refresh=set(self.schedule);self.original=model.forward;self.block_originals=[]
        self.delta={};self.start_hidden=None;self.events=[];self.block_computed=0
        for idx,block in enumerate(model.transformer_blocks):
            original=block.forward;self.block_originals.append(original)
            def wrapped(hidden_states,*args,_idx=idx,_original=original,**kwargs):
                if self.i in self.refresh or _idx not in self.bg:
                    if _idx in self.ends:self.start_hidden=hidden_states.clone()
                    out=_original(hidden_states,*args,**kwargs);self.block_computed+=1
                    if _idx in self.end_set:
                        self.delta[_idx]=(out-self.start_hidden).detach();self.start_hidden=None
                    return out
                if _idx in self.end_set:return hidden_states+self.delta[_idx]
                return hidden_states
            block.forward=wrapped
        model.forward=self.forward

    def forward(self,*args,**kwargs):
        assert self.i<self.steps;self.block_computed=0
        out=self.original(*args,**kwargs)
        self.events.append({'step':self.i+1,'full_compute':self.block_computed==len(self.block_originals),
            'computed_blocks':self.block_computed,'cached_blocks':len(self.block_originals)-self.block_computed})
        self.i+=1
        if self.i==self.steps:self.delta.clear()
        return out

    def report(self):
        assert self.i==self.steps
        total=len(self.block_originals)*self.steps;computed=sum(e['computed_blocks'] for e in self.events)
        return {'algorithm':'ProfilingDiT paper-based LTX reproduction','background_blocks':self.background,
            'block_groups':self.groups,'refresh_steps_zero_based':self.schedule,
            'full_transformer_steps':sum(e['full_compute'] for e in self.events),
            'cached_transformer_steps':0,'equivalent_full_steps':computed/len(self.block_originals),
            'computed_block_calls':computed,'cached_block_calls':total-computed,'total_block_calls':total,
            'transformer_compute_ratio':computed/total,'cache_trace':self.events,
            'full_warmup_steps':10,'final_step_full':True,'restoration_network_used':False,'custom_hardware_fusions':False,
            'residual_boundary':'sum of residuals across each consecutive background block group'}

    def close(self):
        self.model.forward=self.original
        for b,f in zip(self.model.transformer_blocks,self.block_originals):b.forward=f
        self.delta.clear();self.start_hidden=None
