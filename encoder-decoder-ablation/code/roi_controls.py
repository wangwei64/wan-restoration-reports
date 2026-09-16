import contextlib,hashlib
import torch
from torch.nn import functional as F
import restore_fast.pipeline as pm

ORIGINAL_ROUTE=pm.route
ORIGINAL_STATE=pm.RefinementState
def digest(x):return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

@contextlib.contextmanager
def intervention(arm,ref,seed,trace):
 assert arm in ['roi','subject']
 assert pm.route is ORIGINAL_ROUTE and pm.RefinementState is ORIGINAL_STATE
 cache={}
 def route(*args):
  result=ORIGINAL_ROUTE(*args)
  assert torch.equal(result[0].cpu()[0],ref['active'])
  cache['routing']=result[4]
  return result
 class State(ORIGINAL_STATE):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.ids=self.eligible.nonzero().flatten()
   assert torch.equal(self.eligible.cpu(),ref['active'][...,::2,::2].flatten().bool())
   semantic=F.avg_pool3d(cache['routing']['subject_probability'].float(),(1,2,2),(1,2,2)).flatten()
   self.rng=torch.Generator(device=self.eligible.device).manual_seed(seed+(404 if arm=='roi' else 505))
   if arm=='subject':self.selection_priority=semantic
   else:
    rect=torch.zeros(self.token_shape,device=self.eligible.device,dtype=torch.bool)
    h,w=rect.shape[-2:];bbox=[h//4,w//4,h-h//4,w-w//4]
    rect[...,bbox[0]:bbox[2],bbox[1]:bbox[3]]=True
    self.selection_priority=rect.flatten().float()
    trace.update(roi_bbox_yx_halfopen=bbox,roi_eligible_tokens=int(rect.flatten()[self.ids].sum()))
   trace.update(candidate_count=len(self.ids),candidate_sha256=digest(self.eligible),selector=arm,priority_sha256=digest(self.selection_priority),original_subject_sha256=digest(semantic),per_step=[])
  def select(self,idx):
   previous=super().select(idx)
   n=int(ref['refresh_events'][idx-10].sum())
   shuffled=self.ids[torch.randperm(len(self.ids),generator=self.rng,device=self.eligible.device)]
   order=shuffled[torch.argsort(self.selection_priority[shuffled],descending=True,stable=True)]
   chosen=order[:n];due=torch.zeros_like(self.eligible);due[chosen]=True
   assert int(due.sum())==n and not bool((due&~self.eligible).any())
   cutoff=float(self.selection_priority[order[n-1]]) if n else None
   above=int((self.selection_priority[self.ids]>cutoff).sum()) if n else 0
   tied=int((self.selection_priority[self.ids]==cutoff).sum()) if n else 0
   chosen_tied=int((self.selection_priority[chosen]==cutoff).sum()) if n else 0
   assert above+chosen_tied==n and chosen_tied<=tied
   self.counts-=previous.to(torch.int16);self.counts+=due.to(torch.int16)
   self.masks[-1]=due;self.scores[-1]=self.selection_priority.clone()
   self.steps[-1].update(computed_tokens=n,subject_computed_tokens=int(due[self.subject_indices].sum()),low_computed_tokens=int(due[self.low_indices].sum()),wan_calls=int(bool(n)))
   trace['per_step'].append(dict(step=idx+1,quota=n,actual=int(due.sum()),cutoff=cutoff,above_cutoff=above,at_cutoff=tied,selected_at_cutoff=chosen_tied,mask_sha256=digest(due),outside_roi_selected=max(0,n-trace['roi_eligible_tokens']) if arm=='roi' else None))
   return due
 pm.route=route;pm.RefinementState=State
 try:yield
 finally:pm.route=ORIGINAL_ROUTE;pm.RefinementState=ORIGINAL_STATE
