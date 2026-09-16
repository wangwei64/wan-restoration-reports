import torch,json
read=lambda p:json.loads(p.read_text())
def checks(t,ref,budget,arm):
 masks=t['refresh_events'].flatten(1).bool();eligible=t['active'][...,::2,::2].flatten().bool()
 d=dict(prefix_z10_exact=torch.equal(t['x0_step10'],ref['x0_step10']),prefix_z9_exact=torch.equal(t['x0_step9'],ref['x0_step9']),per_step_count_equal=masks.sum(1).tolist()==budget['token_counts'][10:],candidate_count_equal=int(eligible.sum())==budget['capacity'],no_updates_outside_own_candidates=not bool((masks&~eligible).any()),final_refresh_all_own_candidates=torch.equal(masks[-1],eligible))
 outside=(~t['active'].bool()).expand_as(t['latent']);d['own_restoration_outside']=torch.equal(t['latent'][outside],t['restored'][outside])
 if arm not in ['no_semantic_original','no_visual_frozen']:
  keys=['restored','confidence','expected_error','subject_probability','subject_binary','routing_risk']
  if not arm.startswith('route_no_'):keys+=['active','routing_threshold','routing_normalized_risk']
  for k in keys:d[k+'_original_exact']=torch.equal(t[k],ref[k])
 if arm in ['full','full_replay']:
  for k in ['latent','refresh_events']:d[k+'_original_exact']=torch.equal(t[k],ref[k])
 assert all(d.values()),d
 return d
def actual_counts(dest,budget):
 run=read(dest/'run.json');actual=[round(x['active_ratio']*budget['total_tokens']) for x in run['steps']]
 assert actual==budget['token_counts'] and len(run['context_observations'])==1
 return actual
