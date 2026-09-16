import contextlib,hashlib
import torch
import restore_fast.pipeline as pm
ORIGINAL_STATE=pm.RefinementState
def digest(x):return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
@contextlib.contextmanager
def intervention(arm,ref,seed,trace):
 assert arm in ['random_101','random_202','random_303'] and pm.RefinementState is ORIGINAL_STATE
 class State(ORIGINAL_STATE):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   assert torch.equal(self.eligible.cpu(),ref['active'][...,::2,::2].flatten().bool())
   self.ids=self.eligible.nonzero().flatten();self.rng=torch.Generator(device=self.eligible.device).manual_seed(seed+int(arm.split('_')[1]))
   trace.update(selector=arm,candidate_count=len(self.ids),candidate_sha256=digest(self.eligible),per_step=[])
  def select(self,idx):
   original=super().select(idx);n=int(ref['refresh_events'][idx-10].sum())
   chosen=self.ids[torch.randperm(len(self.ids),generator=self.rng,device=self.eligible.device)[:n]]
   due=torch.zeros_like(self.eligible);due[chosen]=True
   assert int(due.sum())==n and not bool((due&~self.eligible).any())
   self.counts-=original.to(torch.int16);self.counts+=due.to(torch.int16);self.masks[-1]=due
   self.steps[-1].update(computed_tokens=n,subject_computed_tokens=int(due[self.subject_indices].sum()),low_computed_tokens=int(due[self.low_indices].sum()),wan_calls=int(bool(n)))
   trace['per_step'].append(dict(step=idx+1,quota=n,actual=n,mask_sha256=digest(due),cutoff=None,above_cutoff=None,at_cutoff=None,selected_at_cutoff=None))
   return due
 pm.RefinementState=State
 try:yield
 finally:pm.RefinementState=ORIGINAL_STATE
