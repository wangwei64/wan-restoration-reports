from pathlib import Path
import os,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),'/root/autodl-tmp/experiments/wan_paper_experiments_20260912','/root/autodl-tmp/wan_restore_fast','/root/autodl-tmp/Wan2.1']
from common import verify_frozen,gpu_pids,Guard

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def status(stage,**kw):save(ROOT/'status.json',dict(stage=stage,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),**kw))
def verified():
 p=read(ROOT/'config.json');reg=read(ROOT/'registration.json');assert sha(ROOT/'config.json')==reg['config_sha256']
 for n,h in reg['source_sha256'].items():assert sha(ROOT/n)==h,n
 verify_frozen();return p,reg

def jobs(p):
 out=[]
 for c in p['cases']:
  out.append(dict(key=c['key'],id='ours',panel='shared',policy='ours',arm='full'))
  for pol in p['encoder_policies'][:-1]:
   arms=[pol] if pol=='roi' else [f'{pol}_{s}' for s in p['schedule_offsets']]
   out.extend(dict(key=c['key'],id='enc_'+a,panel='encoder',policy=pol,arm=a) for a in arms)
  out.extend(dict(key=c['key'],id='dec_'+a,panel='decoder',policy=a,arm=a) for a in p['decoder_arms'][:-1])
 assert len(out)==216;return out
