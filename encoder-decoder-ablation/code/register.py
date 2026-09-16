from runtime import *
def main():
 assert not (ROOT/'registration.json').exists();p=read(ROOT/'config.json');frozen=verify_frozen();pool={};cache={};manifest_hashes={}
 for source in p['source_manifests']:
  manifest_hashes[source]=sha(source)
  for x in read(source)['rows']:
   arm=x.get('id',x.get('arm'));panel='decoder' if arm.startswith('S') and '_C' in arm else 'encoder'
   pool[(x['key'],panel,arm)]=dict(record=x,manifest=source)
 records={};missing=[]
 for j in jobs(p):
  c=next(c for c in p['cases'] if c['key']==j['key']);ident=j['key']+'|'+j['id']
  assert sha(c['native_video'])==c['native_video_sha256']
  if j['id']=='ours':
   match=pool[(j['key'],'decoder','S1_C1_A1')];x=match['record'];assert x['video_sha256']==c['full_video_sha256'];x={**x,'directory':c['full_directory']}
   other=pool.get((j['key'],'encoder','full'))
   if other:
    assert other['record']['video_sha256']==x['video_sha256']
    for m in p['metrics']:assert abs(other['record']['metrics'][m]-x['metrics'][m])<1e-12
  else:
   match=pool.get((j['key'],j['panel'],j['arm']))
   if not match:missing.append(j);continue
   x=match['record']
  assert sha(Path(x['directory'])/'final.mp4')==x['video_sha256']
  if 'native_video_sha256' in x:assert x['native_video_sha256']==c['native_video_sha256']
  records[ident]=dict(record=x,manifest=match['manifest'])
  cache[j['key']+'|'+x['video_sha256']]=x['metrics']
 reg=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),config_sha256=sha(ROOT/'config.json'),source_sha256={f.name:sha(f) for f in ROOT.glob('*.py')},method_sha256=frozen,manifest_sha256=manifest_hashes,reused=records,metric_cache=cache,missing=missing,total=216,reuse_count=len(records),new_count=len(missing))
 save(ROOT/'registration.json',reg);print(json.dumps(dict(total=216,reused=len(records),new=len(missing))))
if __name__=='__main__':main()
