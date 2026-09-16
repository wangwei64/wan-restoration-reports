import contextlib,hashlib
import torch
from torch.nn import functional as F
import restore_fast.pipeline as pm
ORIGINAL_ROUTE=pm.route;ORIGINAL_STATE=pm.RefinementState
def digest(x):return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def quota(score,ids,n):
 mask=torch.zeros_like(score,dtype=torch.bool);order=ids[torch.argsort(score[ids],descending=True,stable=True)];mask[order[:n]]=True
 return mask,dict(above_threshold=int((score[ids]>score[order[n-1]]).sum()) if n else 0,selected_at_threshold=n-int((score[ids]>score[order[n-1]]).sum()) if n else 0)

@contextlib.contextmanager
def intervention(arm,ref,seed,trace):
 kind,offset=arm.rsplit('_',1);assert kind in ['subject_random','detail_random'] and int(offset) in [101,202,303]
 assert pm.route is ORIGINAL_ROUTE and pm.RefinementState is ORIGINAL_STATE
 cache={}
 def route(*args):
  result=ORIGINAL_ROUTE(*args);cache.update(heat=args[4],route=result)
  for j,k in [(0,'active'),(2,'subject_binary')]:assert torch.equal(result[j].cpu()[0],ref[k])
  for k,v in result[4].items():assert torch.equal(v.cpu()[0],ref[k])
  return result
 class State(ORIGINAL_STATE):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.ids=self.eligible.nonzero().flatten()
   subject=ref['subject_binary'][...,::2,::2].flatten().to(self.eligible.device).bool()&self.eligible
   heat=F.avg_pool3d(cache['heat'].float(),(1,2,2),(1,2,2)).flatten().clamp(0,1)
   count=int(subject.sum());assert 0<count<len(self.ids)
   detail=torch.zeros_like(subject);ordered=self.ids[torch.argsort(heat[self.ids],descending=True,stable=True)];detail[ordered[:count]]=True
   region=subject if kind=='subject_random' else detail
   self.region=region;self.priority=region.float();self.rng=torch.Generator(device=self.eligible.device).manual_seed(seed+int(offset))
   trace.update(candidate_count=len(self.ids),candidate_sha256=digest(self.eligible),region_kind=kind,region_capacity=count,region_fraction=count/len(self.ids),region_sha256=digest(region),original_subject_region_sha256=digest(subject),matched_detail_region_sha256=digest(detail),subject_detail_overlap=int((subject&detail).sum()),visual_region_cutoff=float(heat[ordered[count-1]]),rng_seed=seed+int(offset),per_step=[])
  def select(self,idx):
   original=super().select(idx);n=int(ref['refresh_events'][idx-10].sum())
   shuffled=self.ids[torch.randperm(len(self.ids),generator=self.rng,device=self.eligible.device)]
   order=shuffled[torch.argsort(self.priority[shuffled],descending=True,stable=True)];due=torch.zeros_like(self.eligible);due[order[:n]]=True
   inside=int((due&self.region).sum());outside=int((due&~self.region).sum())
   assert int(due.sum())==n and inside==min(n,int(self.region.sum())) and outside==max(0,n-int(self.region.sum()))
   assert not bool((due&~self.eligible).any())
   if idx==49:assert torch.equal(due,self.eligible)
   self.counts-=original.to(torch.int16);self.counts+=due.to(torch.int16);self.masks[-1]=due;self.scores[-1]=self.priority.clone()
   self.steps[-1].update(computed_tokens=n,subject_computed_tokens=int(due[self.subject_indices].sum()),low_computed_tokens=int(due[self.low_indices].sum()),wan_calls=int(bool(n)))
   trace['per_step'].append(dict(step=idx+1,quota=n,actual=int(due.sum()),inside_selected=inside,outside_selected=outside,mask_sha256=digest(due)))
   return due
 pm.route=route;pm.RefinementState=State
 try:yield
 finally:pm.route=ORIGINAL_ROUTE;pm.RefinementState=ORIGINAL_STATE
