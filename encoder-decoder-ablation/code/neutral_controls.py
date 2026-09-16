"""Region-matched oldest-first coverage controls; no change to Full or its weights.

Within each cue region, replace independent random selection by oldest-first
selection. Preserve each corresponding region-random arm's inside/outside quota.
Only age and a binary region enter this selector. Original reconstruction and
real-observation integration are retained. Random ties share the baseline RNG.
"""
import contextlib,hashlib
import torch
from torch.nn import functional as F

def digest(x):return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

def coverage_select(age,region,eligible,n,rng):
 ids=eligible.nonzero().flatten();assert 0<=n<=len(ids) and bool(torch.isfinite(age[ids]).all()) and bool((age[ids]>=0).all())
 # One identical permutation per step in both the matched random baseline and
 # this policy. It affects only ties, never the age or region priority.
 shuffled=ids[torch.randperm(len(ids),generator=rng,device=ids.device)]
 age_order=shuffled[torch.argsort(age[shuffled],descending=True,stable=True)]
 order=age_order[torch.argsort(region[age_order].to(torch.int8),descending=True,stable=True)]
 due=torch.zeros_like(eligible);due[order[:n]]=True
 score=region.float()+age/(1+age[ids].max())
 cut=score[order[n-1]] if n else None
 above=int((score[ids]>cut).sum()) if n else 0;at=int((score[ids]==cut).sum()) if n else 0
 assert int(due.sum())==n and above<=n<=above+at
 if n:assert float(score[due].min())>=float(score[eligible&~due].max()) if bool((eligible&~due).any()) else True
 info=dict(threshold=float(cut) if n else None,above_threshold=above,at_threshold=at,selected_at_threshold=n-above,inside_selected=int((due&region).sum()),outside_selected=int((due&~region).sum()),mean_selected_age=float(age[due].mean()) if n else 0.)
 assert info['inside_selected']==min(n,int((region&eligible).sum()))
 return due,score,info

def coverage_state(base,arm,ref,seed,trace):
 kind,offset=arm.rsplit('_',1);assert kind in ['uniform_coverage','subject_coverage','detail_coverage'] and int(offset) in [101,202,303]
 class State(base):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);heat=args[4];ids=self.eligible.nonzero().flatten()
   subject=ref['subject_binary'][...,::2,::2].flatten().to(self.eligible.device).bool()&self.eligible
   detail_values=F.avg_pool3d(heat.float(),(1,2,2),(1,2,2)).flatten().clamp(0,1)
   count=int(subject.sum());assert 0<count<len(ids)
   detail=torch.zeros_like(subject);ordered=ids[torch.argsort(detail_values[ids],descending=True,stable=True)];detail[ordered[:count]]=True
   self.region=subject if kind=='subject_coverage' else detail if kind=='detail_coverage' else torch.zeros_like(subject)
   assert torch.equal(self.eligible.cpu(),ref['active'][...,::2,::2].flatten().bool())
   self.rng=torch.Generator(device=self.eligible.device).manual_seed(seed+int(offset))
   trace.update(policy=kind,rng_seed=seed+int(offset),scope='refresh selector only; original joint reconstruction and routing retained',candidate_count=len(ids),candidate_sha256=digest(self.eligible),region_capacity=int(self.region.sum()),region_sha256=digest(self.region),original_subject_region_sha256=digest(subject),matched_detail_region_sha256=digest(detail),subject_detail_overlap=int((subject&detail).sum()),visual_region_cutoff=float(detail_values[ordered[count-1]]),selection_inputs=['binary_region','time_since_last_real_update','random_tie_breaker'],continuous_subject_probability_used=False,risk_used_in_selection=False,innovation_used_in_selection=False,importance_used_in_selection=False,per_step=[])
  def select(self,idx):
   c=self.core;age=(c.lam(c.sigmas[idx])-c.lam(c.sigmas[c.i0])).clamp_min(0)
   n=int(ref['refresh_events'][idx-10].sum());due,score,info=coverage_select(age,self.region,self.eligible,n,self.rng)
   if idx==49:assert torch.equal(due,self.eligible)
   self.counts+=due.to(torch.int16);self.masks.append(due);self.scores.append(score)
   self.steps.append(dict(step=idx+1,computed_tokens=n,subject_computed_tokens=int(due[self.subject_indices].sum()),low_computed_tokens=int(due[self.low_indices].sum()),wan_calls=int(bool(n)),boundary_refresh=idx==49))
   trace['per_step'].append(dict(step=idx+1,quota=n,actual=int(due.sum()),mask_sha256=digest(due),score_sha256=digest(score),**info));return due
  def report(self):return {**super().report(),'refinement_policy':kind,'fixed_refresh_tiers':kind!='uniform_coverage','selector_only_ablation':True,'selection_uses_risk_or_innovation_feedback':False}
  def tensors(self):return {**super().tensors(),'coverage_region':self.region.reshape(self.token_shape).repeat_interleave(2,-2).repeat_interleave(2,-1)}
 return State

@contextlib.contextmanager
def intervention(arm,ref,seed,trace):
 import restore_fast.pipeline as pipeline
 original_route,original_state=pipeline.route,pipeline.RefinementState
 def route(*args,**kwargs):
  result=original_route(*args,**kwargs)
  assert torch.equal(result[0].cpu()[0],ref['active']) and torch.equal(result[2].cpu()[0],ref['subject_binary'])
  for k,v in result[4].items():assert torch.equal(v.cpu()[0],ref[k]),k
  return result
 pipeline.route=route;pipeline.RefinementState=coverage_state(original_state,arm,ref,seed,trace)
 try:yield
 finally:pipeline.route=original_route;pipeline.RefinementState=original_state
