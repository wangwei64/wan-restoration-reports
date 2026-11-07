"""Compare a previous server's plan with this package before a full rerun."""
import argparse,json
from runtime import ROOT,read,configuration
from benchmark import jobs
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('previous');p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--samples',type=int,default=5);p.add_argument('--flicker-samples',type=int,default=25)
    p.add_argument('--config');a=p.parse_args();old=read(a.previous)
    if 'protocol' in old:old=old['protocol']
    expected={'generation':configuration(a.config),'seed':a.seed,'samples':a.samples,'flicker_samples':a.flicker_samples,
              'vbench_source':read(ROOT/'data/vbench_source.json')['commit']}
    missing=[k for k in expected if k not in old]
    differences={k:{'previous':old[k],'requested':v} for k,v in expected.items() if k in old and old[k]!=v}
    result={'matches':not missing and not differences,'missing':missing,'differences':differences,
            'planned_videos_per_method':len(list(jobs(['sencache_ltx096'],a.seed,a.samples,a.flicker_samples))),
            'scope':'Plan settings only; this cannot certify unrecorded prompt edits or previous completion.'}
    print(json.dumps(result,indent=2))
    if not result['matches']:raise SystemExit(1)
if __name__=='__main__':main()
