"""Shared CLI. Run on the GPU server with its existing Wan environment."""
from pathlib import Path
import argparse,subprocess,sys,os
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('command',choices=['register','generate','evaluate','summarize','all','status']);a=p.parse_args()
if a.command=='status':
 print((ROOT/'status.json').read_text() if (ROOT/'status.json').exists() else '{"stage":"not_started"}');sys.exit()
commands=['register','generate','evaluate','summarize'] if a.command=='all' else [a.command]
env=dict(os.environ,TORCH_HOME='/root/.cache/torch',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
for name in commands:
 if name=='register' and (ROOT/'registration.json').exists():continue
 with (ROOT/(name+'.log')).open('a',encoding='utf-8') as log:
  child=subprocess.Popen([sys.executable,str(ROOT/(name+'.py'))],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (ROOT/'worker.json').write_text(__import__('json').dumps(dict(stage=name,pid=child.pid,parent_pid=os.getpid())))
  code=child.wait()
 if code:
  from runtime import status
  status('failed',phase=name,exit_code=code)
  sys.exit(code)
