from runtime import *
import traceback
from metric_functions import metrics,video_read,torch,cv2,lpips

def main():
 p,reg=verified();assert not gpu_pids();g=read(ROOT/'generation_summary.json')['rows'];assert len(g)==216
 cache=dict(reg['metric_cache']);rows=[];model=None;torch.set_num_threads(3);cv2.setNumThreads(1)
 prior=read(ROOT/'evaluation_summary.json')['rows'] if (ROOT/'evaluation_summary.json').exists() else []
 for x in prior:cache[x['key']+'|'+x['video_sha256']]=x['metrics']
 with Guard() as guard:
  for c in p['cases']:
   assert sha(c['native_video'])==c['native_video_sha256'];native=None
   for x in [x for x in g if x['key']==c['key']]:
    status('evaluation',key=c['key'],arm=x['id'],completed=len(rows),total=216)
    f=Path(x['directory'])/'final.mp4';assert sha(f)==x['video_sha256'];key=c['key']+'|'+x['video_sha256']
    if key in cache:m=cache[key];source='matching sample and video SHA'
    else:
     if model is None:model=lpips.LPIPS(net='alex').cuda().eval()
     if native is None:native=video_read(c['native_video'])
     m=metrics(model,video_read(f),native);cache[key]=m;source='new evaluation'
    rows.append(dict(**x,metrics=m,metric_source=source));save(ROOT/'evaluation_summary.json',dict(rows=rows))
 guard.verify();verify_frozen();assert len(rows)==216
if __name__=='__main__':
 try:main()
 except Exception:status('failed',phase='evaluation',traceback=traceback.format_exc());raise
