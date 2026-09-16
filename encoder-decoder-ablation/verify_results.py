import json,hashlib,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
R=BASE/'results'
CODE=BASE/'code'
r=json.loads((R/'results.json').read_text()); reg=json.loads((R/'registration.json').read_text())
assert hashlib.sha256((CODE/'config.json').read_bytes()).hexdigest()==reg['config_sha256']
for n,h in reg['source_sha256'].items():assert hashlib.sha256((CODE/n).read_bytes()).hexdigest()==h,n
rows=r['rows']; assert len(rows)==216 and len({(x['key'],x['id']) for x in rows})==216
assert sum(not x['reused'] for x in rows)==36
assert len([x for x in rows if x['panel']=='shared'])==8
assert len({(x['key'],x['panel'],x['policy']) for x in rows})==120
metrics=['lpips','ssim','psnr_db']; wins={}
for key,s in r['per_sample'].items():
 assert s['encoder']['ours']==s['decoder']['ours']
 for panel in ['encoder','decoder']:
  for policy,v in s[panel].items():
   rr=[x for x in rows if x['key']==key and x['policy']==policy and x['panel'] in [panel,'shared']]
   assert rr
   for m in metrics:assert abs(statistics.mean(x['metrics'][m] for x in rr)-v[m])<1e-12
for panel,table in r['tables'].items():
 assert len(table)==8
 wins[panel]={}
 for policy,v in table.items():
  for m in metrics:assert abs(statistics.mean(s[panel][policy][m] for s in r['per_sample'].values())-v[m])<1e-12
  if policy=='ours':continue
  wins[panel][policy]={}
  for m in metrics:
   ds=[(s[panel]['ours'][m]-s[panel][policy][m])*(-1 if m=='lpips' else 1) for s in r['per_sample'].values()]
   wins[panel][policy][m]=dict(win=sum(d>1e-10 for d in ds),tie=sum(abs(d)<=1e-10 for d in ds),loss=sum(d< -1e-10 for d in ds))
assert r['tables']['encoder']['ours']==r['tables']['decoder']['ours']
out=dict(passed=True,raw_rows=216,new_rows=36,logical_conditions=120,shared_ours=8,source_hashes_verified=True,means_recomputed=True,ours_pairwise=wins)
# Read-only verification: preserve the published audit.
print(json.dumps(out))
