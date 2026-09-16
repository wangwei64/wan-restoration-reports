"""Oldest-first refresh with a single binary cue only breaking age ties.

The neutral comparator is Uniform Coverage from the previous fixed pilot.
It differs only in whether equally old tokens are region-prioritized or random.
"""
import contextlib,hashlib
import torch
from torch.nn import functional as F
def digest(x):return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def select(age,region,eligible,n,rng):
 ids=eligible.nonzero().flatten();assert 0<=n<=len(ids) and bool(torch.isfinite(age[ids]).all())
 shuffled=ids[torch.randperm(len(ids),generator=rng,device=ids.device)]
 regional=shuffled[torch.argsort(region[shuffled].to(torch.int8),descending=True,stable=True)]
 order=regional[torch.argsort(age[regional],descending=True,stable=True)]
 due=torch.zeros_like(eligible);due[order[:n]]=True
 cut=age[order[n-1]] if n else None;above=eligible&(age>cut) if n else torch.zeros_like(eligible);at=eligible&(age==cut) if n else torch.zeros_like(eligible)
 assert int(due.sum())==n and int(above.sum())<=n<=int(above.sum()+at.sum()) and not bool((above&~due).any())
 # At the age threshold, prefer the region without displacing any older token.
 assert int((due&at&region).sum())==min(n-int(above.sum()),int((at&region).sum()))
 info=dict(threshold=float(cut) if n else None,above_threshold=int(above.sum()),at_threshold=int(at.sum()),selected_at_threshold=n-int(above.sum()),inside_selected=int((due&region).sum()),outside_selected=int((due&~region).sum()),preferred_region_at_age_threshold=int((due&at&region).sum()))
 return due,age.clone(),info
def make_state(base,arm,ref,seed,trace):
 kind,offset=arm.rsplit('_',1);assert kind in ['subject_tiebreak','detail_tiebreak'] and int(offset) in [101,202,303]
 class State(base):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);ids=self.eligible.nonzero().flatten();heat=F.avg_pool3d(args[4].float(),(1,2,2),(1,2,2)).flatten().clamp(0,1)
   subject=ref['subject_binary'][...,::2,::2].flatten().to(self.eligible.device).bool()&self.eligible;count=int(subject.sum());assert 0<count<len(ids)
   detail=torch.zeros_like(subject);order=ids[torch.argsort(heat[ids],descending=True,stable=True)];detail[order[:count]]=True
   self.region=subject if kind=='subject_tiebreak' else detail
   assert torch.equal(self.eligible.cpu(),ref['active'][...,::2,::2].flatten().bool())
   self.rng=torch.Generator(device=self.eligible.device).manual_seed(seed+int(offset))
   trace.update(policy=kind,ordering='age first, binary cue second, random ties last',scope='selector only, unchanged joint restoration',rng_seed=seed+int(offset),candidate_count=len(ids),candidate_sha256=digest(self.eligible),region_capacity=count,region_sha256=digest(self.region),continuous_subject_probability_used=False,risk_used_in_selection=False,innovation_used_in_selection=False,importance_used_in_selection=False,per_step=[])
  def select(self,idx):
   c=self.core;age=(c.lam(c.sigmas[idx])-c.lam(c.sigmas[c.i0])).clamp_min(0);n=int(ref['refresh_events'][idx-10].sum())
   due,score,info=select(age,self.region,self.eligible,n,self.rng)
   if idx==49:assert torch.equal(due,self.eligible)
   self.counts+=due.to(torch.int16);self.masks.append(due);self.scores.append(score);self.steps.append(dict(step=idx+1,computed_tokens=n,subject_computed_tokens=int(due[self.subject_indices].sum()),low_computed_tokens=int(due[self.low_indices].sum()),wan_calls=int(bool(n)),boundary_refresh=idx==49))
   trace['per_step'].append(dict(step=idx+1,quota=n,actual=int(due.sum()),mask_sha256=digest(due),**info));return due
  def report(self):return {**super().report(),'refinement_policy':kind,'fixed_refresh_tiers':False,'selector_only_ablation':True,'selection_uses_risk_or_innovation_feedback':False}
  def tensors(self):return {**super().tensors(),'coverage_region':self.region.reshape(self.token_shape).repeat_interleave(2,-2).repeat_interleave(2,-1)}
 return State
@contextlib.contextmanager
def intervention(arm,ref,seed,trace):
 import restore_fast.pipeline as p
 oldroute,oldstate=p.route,p.RefinementState
 def route(*args,**kwargs):
  r=oldroute(*args,**kwargs);assert torch.equal(r[0].cpu()[0],ref['active']) and torch.equal(r[2].cpu()[0],ref['subject_binary'])
  for k,v in r[4].items():assert torch.equal(v.cpu()[0],ref[k])
  return r
 p.route=route;p.RefinementState=make_state(oldstate,arm,ref,seed,trace)
 try:yield
 finally:p.route=oldroute;p.RefinementState=oldstate
