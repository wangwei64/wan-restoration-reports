from runtime import *
import gc,traceback,shutil
import torch
from safetensors.torch import load_file,save_file
from restore_fast.pipeline import RestorationPipeline
from restore_fast.config import GenerationConfig
from restore_fast.video import write_video
from encoder_controls import intervention
from validation import checks,actual_counts
import decoder_controls as decoder

def main():
 p,reg=verified();rows=[];pipe=None
 while gpu_pids():status('waiting_for_free_gpu');time.sleep(20)
 try:
  for c in p['cases']:
   ref=load_file(str(Path(c['full_directory'])/'outputs.safetensors'));N=ref['refresh_events'][0].numel()
   budget=dict(total_tokens=N,capacity=int(ref['active'][...,::2,::2].sum()),token_counts=[N]*10+ref['refresh_events'].flatten(1).sum(1).tolist())
   save(ROOT/'budgets'/(c['key']+'.json'),budget);save(ROOT/'references'/(c['key']+'.json'),dict(native=c['native_video'],native_sha256=c['native_video_sha256'],ours=c['full_video'],ours_sha256=c['full_video_sha256']))
   for j in [j for j in jobs(p) if j['key']==c['key']]:
    dest=ROOT/'runs'/j['key']/j['id'];measurement=dest/'measurement.json';ident=j['key']+'|'+j['id']
    status('generation',key=j['key'],arm=j['id'],completed=len(rows),total=216)
    if measurement.exists():
     row=read(measurement);assert row['id']==j['id'] and row['key']==j['key'] and sha(dest/'final.mp4')==row['video_sha256']
     rows.append(row);save(ROOT/'generation_summary.json',dict(rows=rows));continue
    if dest.exists():
     archive=ROOT/'incomplete'/time.strftime('%Y%m%d_%H%M%S')/j['key']/j['id'];archive.parent.mkdir(parents=True,exist_ok=True)
     assert dest.resolve().is_relative_to(ROOT.resolve()) and archive.resolve().is_relative_to(ROOT.resolve());dest.rename(archive)
    reused=reg['reused'].get(ident);trace={};audit={}
    if reused:
     x=reused['record'];source=Path(x['directory']);assert sha(source/'final.mp4')==x['video_sha256'];dest.mkdir(parents=True)
     for name in ['final.mp4','outputs.safetensors','run.json']:
      if (source/name).exists():(dest/name).symlink_to(source/name)
     if j['panel'] in ['encoder','shared']:
      audit=checks(load_file(str(dest/'outputs.safetensors')),ref,budget,j['arm']);actual_counts(dest,budget)
     else:audit=dict(source_manifest=reused['manifest'],source_audit=x.get('audit'),source_checks=x.get('checks'))
    else:
     if pipe is None:pipe=RestorationPipeline(config=GenerationConfig(**p['profile']))
     pipe.controller.reset();gc.collect();torch.cuda.empty_cache()
     with Guard() as guard:
      if j['panel']=='encoder':
       with intervention(j['arm'],ref,c['seed'],trace):pipe.generate(c['prompt'],c['seed'],dest)
       t=load_file(str(dest/'outputs.safetensors'));audit=checks(t,ref,budget,j['arm']);actual_counts(dest,budget);del t
       assert len(trace['per_step'])==40
      elif j['panel']=='decoder':
       s,ctx,a=[int(v[-1]) for v in j['arm'].split('_')]
       pref=ROOT/'prefix'/c['key']/f'S{s}_C{ctx}_A{a}'
       if pref.exists():
        archive=ROOT/'incomplete'/time.strftime('%Y%m%d_%H%M%S')/'prefix'/c['key']/pref.name;archive.parent.mkdir(parents=True,exist_ok=True);pref.rename(archive)
       with decoder.kv_mode(bool(a)) as trace:aligned,raw,audit=decoder.prefix(pipe,c,bool(s),bool(ctx),bool(a))
       assert torch.equal(aligned,raw);assert trace['calls']>0
       if not a:assert trace['effective_max']==0
       dest.mkdir(parents=True);frames,_=pipe.wan.decode(raw);write_video(dest/'final.mp4',frames,pipe.config.fps);save_file({'latent':raw[0].cpu().contiguous()},str(dest/'outputs.safetensors'));del aligned,raw
       assert audit['actual_50_step_token_counts']==budget['token_counts']
      else:raise AssertionError('Ours must be the registered shared reference')
     guard.verify()
    row=dict(**j,directory=str(dest),video_sha256=sha(dest/'final.mp4'),native_video_sha256=c['native_video_sha256'],reused=bool(reused),audit=audit,trace=trace)
    save(measurement,row);rows.append(row);save(ROOT/'generation_summary.json',dict(rows=rows))
   del ref
  assert len(rows)==216;verify_frozen();save(ROOT/'generation_validation.json',dict(passed=True,total=216,new=sum(not x['reused'] for x in rows)))
 finally:
  if pipe is not None:pipe.close()
if __name__=='__main__':
 try:main()
 except Exception:status('failed',phase='generation',traceback=traceback.format_exc());raise
