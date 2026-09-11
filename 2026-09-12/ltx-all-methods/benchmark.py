"""Deterministic VBench plan, resumable per-GPU execution, and standard export."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from runtime import ROOT,Runner,GPUContended,configuration,fingerprint,read,save,sha

def prompt_groups():
    rows=read(ROOT/'data/VBench_full_info.json');groups={}
    for index,row in enumerate(rows):
        prompt=row['prompt_en']
        if prompt not in groups:groups[prompt]={'id':f'vbench_{index:04d}','prompt':prompt,'dimensions':[],'record_indices':[]}
        group=groups[prompt];group['record_indices'].append(index)
        group['dimensions']=sorted(set(group['dimensions'])|set(row['dimension']))
    return list(groups.values())

def jobs(methods,base_seed=20260912,samples=5,flicker_samples=25):
    for group in prompt_groups():
        n=max(samples,flicker_samples) if 'temporal_flickering' in group['dimensions'] else samples
        for index in range(n):
            # Same content/seed across every method and across scheduling/sharding.
            material=f'{base_seed}\0{group["prompt"]}\0{index}'.encode()
            seed=int.from_bytes(hashlib.sha256(material).digest()[:8],'big')%(2**31-1)
            for method in methods:yield dict(group,index=index,seed=seed,method=method)

def job_directory(out,job):return out/job['method']/job['id']/f'{job["index"]:02d}'

def checked_job(out,job,generation,methods):
    marker=out/'completed.json'
    if not marker.exists():return None
    done=read(marker);run=read(out/'run.json')
    if sha(out/'video.mp4')!=done['video_sha256'] or sha(out/'run.json')!=done['run_sha256']:raise RuntimeError('Corrupt completed output: '+str(out))
    request=run['request']
    if done['request_sha256']!=fingerprint(request) or run['request_sha256']!=done['request_sha256']:raise RuntimeError('Request fingerprint mismatch: '+str(out))
    expected={'prompt':job['prompt'],'seed':job['seed'],'method':job['method'],'preset':methods[job['method']],'generation':generation}
    if any(request[k]!=value for k,value in expected.items()):raise RuntimeError('Output protocol/preset mismatch: '+str(out))
    if any(run[k]!=job[k] for k in ['prompt','seed','method']):raise RuntimeError('Run metadata mismatch: '+str(out))
    if not run['exclusive_gpu_timing_verified']:raise RuntimeError('Unverified GPU timing: '+str(out))
    return run,done

def plan(args):
    methods=list(read(ROOT/'configs/methods.json')) if args.methods==['all'] else args.methods
    unknown=set(methods)-set(read(ROOT/'configs/methods.json'))
    if unknown or len(set(methods))!=len(methods):raise ValueError('Invalid/repeated methods: '+str(unknown))
    if not 0<=args.shard<args.shards or args.shards<1 or args.samples<1 or args.flicker_samples<1:raise ValueError('Invalid shard/sample count')
    all_jobs=list(jobs(methods,args.seed,args.samples,args.flicker_samples))
    selected=[j for i,j in enumerate(all_jobs) if i%args.shards==args.shard]
    if args.limit:selected=selected[:args.limit]
    details={'official_records':946,'unique_prompts':len(prompt_groups()),'methods':methods,'samples':args.samples,'flicker_samples':args.flicker_samples,
             'seed':args.seed,'total_jobs':len(all_jobs),'shard':args.shard,'shards':args.shards,'selected_jobs':len(selected),'limit':args.limit,
             'generation':configuration(args.config),'methods_config_sha256':fingerprint({m:read(ROOT/'configs/methods.json')[m] for m in methods}),'vbench_source':read(ROOT/'data/vbench_source.json')['commit']}
    return details,selected

def execute(args):
    details,todo=plan(args);args.output.mkdir(parents=True,exist_ok=True)
    presets=read(ROOT/'configs/methods.json')
    pending=[m for m in details['methods'] if presets[m].get('pending_validation')]
    if pending:raise RuntimeError('Unfinished preset validation: '+str(pending))
    plan_path=args.output/f'plan_{args.shard:04d}_of_{args.shards:04d}.json'
    if plan_path.exists() and read(plan_path)!=details:raise RuntimeError('Output plan differs; use a new output directory')
    save(plan_path,details)
    # Cross-shard immutable protocol, including method list, rejects mixed experiments.
    shared={k:v for k,v in details.items() if k not in ['shard','selected_jobs','limit']}
    shared_path=args.output/'benchmark_protocol.json'
    import fcntl
    with (args.output/'.protocol.lock').open('a') as protocol_lock:
        fcntl.flock(protocol_lock,fcntl.LOCK_EX)
        if shared_path.exists():
            if read(shared_path)!=shared:raise RuntimeError('Another shard uses a different protocol')
        else:save(shared_path,shared)
    lock_path=args.output/f'.shard_{args.shard}.lock'
    # Linux flock is automatically released on a crash, unlike PID marker files.
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=Runner(args.paths,args.config)
        for position,job in enumerate(todo):
            deadline=time.time()+args.gpu_wait_hours*3600
            while True:
                status={'stage':'running','position':position,'total':len(todo),'job':job,'updated':time.time()}
                save(args.output/f'status_{args.shard}.json',status)
                try:
                    runner.generate(job['prompt'],job['seed'],job['method'],job_directory(args.output,job),keep_latents=args.keep_latents)
                    break
                except GPUContended as error:
                    save(args.output/f'status_{args.shard}.json',dict(status,stage='waiting_for_gpu',reason=str(error)))
                    if time.time()>deadline:raise
                    time.sleep(15)
        save(args.output/f'status_{args.shard}.json',{'stage':'complete','jobs':len(todo),'updated':time.time()})

def export(args):
    details,todo=plan(args)
    if args.shards!=1 or args.limit:raise ValueError('Export the combined full run with --shards 1 and no --limit')
    target=args.export_dir;target.mkdir(parents=True,exist_ok=True);missing=[];mapping={};seen={};methods=read(ROOT/'configs/methods.json')
    for job in todo:
        out=job_directory(args.output,job);checked=checked_job(out,job,details['generation'],methods)
        if checked is None:missing.append(str(out));continue
        run,done=checked
        prompt=job['prompt']
        if any(x in prompt for x in ['/', '\\', '\0']) or len((prompt+'-24.mp4').encode())>255:raise ValueError('Prompt cannot be represented as an official filename; use metadata mapping export: '+job['id'])
        dest=target/job['method']/(prompt+f'-{job["index"]}.mp4');dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            if sha(dest)!=done['video_sha256']:raise RuntimeError('Conflicting export: '+str(dest))
        else:
            try:os.link(out/'video.mp4',dest)
            except OSError:shutil.copyfile(out/'video.mp4',dest)
        mapping[str(dest.resolve())]=prompt;seen[(job['method'],job['id'],job['index'])]=True
    save(target/'export_receipt.json',{'complete':not missing,'exported':len(seen),'missing_count':len(missing),'missing':missing,'protocol':details})
    save(target/'filename_to_prompt.json',mapping)
    if missing:raise RuntimeError(f'{len(missing)} videos missing; official evaluation must wait for every shard')
    print(json.dumps({'exported':len(seen),'directory':str(target)}))

def summarize(args):
    details,todo=plan(args);rows=[];methods=read(ROOT/'configs/methods.json')
    for job in todo:
        out=job_directory(args.output,job)
        checked=checked_job(out,job,details['generation'],methods)
        if checked is None:continue
        run,_=checked;rows.append({'method':job['method'],'id':job['id'],'index':job['index'],'seed':job['seed'],'seconds':run['online_seconds'],'gpu_name':run.get('hardware',{}).get('gpu_name'),'gpu_uuid':run.get('hardware_gpu_uuid')})
    by=collections.defaultdict(list)
    for row in rows:by[row['method']].append(row['seconds'])
    result={'complete':len(rows)==len(todo),'completed':len(rows),'expected':len(todo),'methods':{k:{'videos':len(v),'mean_seconds':sum(v)/len(v)} for k,v in by.items()}}
    save(args.output/'timing_summary.json',result);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run','export','summarize'])
    p.add_argument('--methods',nargs='+',default=['all']);p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--samples',type=int,default=5);p.add_argument('--flicker-samples',type=int,default=25)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/vbench');p.add_argument('--paths',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--export-dir',type=Path,default=ROOT/'outputs/vbench_standard');p.add_argument('--gpu-wait-hours',type=float,default=12);p.add_argument('--keep-latents',action='store_true')
    args=p.parse_args()
    if args.action=='plan':print(json.dumps(plan(args)[0],indent=2));return
    if args.action=='run':execute(args)
    elif args.action=='export':export(args)
    else:summarize(args)

if __name__=='__main__':main()
