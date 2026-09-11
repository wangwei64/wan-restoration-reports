"""Run the pinned external official VBench CLI, including static filtering."""
import argparse,json,shlex,subprocess,time
from pathlib import Path
from runtime import ROOT,read,save,sha,fingerprint

DIMENSIONS=['subject_consistency','background_consistency','temporal_flickering','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality','object_class','multiple_objects','human_action','color','spatial_relationship','scene','temporal_style','appearance_style','overall_consistency']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vbench-repo',type=Path,required=True);p.add_argument('--python',default='python')
    p.add_argument('--videos',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--methods',nargs='+',default=['all'])
    p.add_argument('--dimensions',nargs='+',choices=DIMENSIONS,default=DIMENSIONS);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    repo=a.vbench_repo.resolve();videos=a.videos.resolve();output=a.output.resolve();methods=list(read(ROOT/'configs/methods.json')) if a.methods==['all'] else a.methods
    if not set(methods)<=set(read(ROOT/'configs/methods.json')):raise ValueError('Unknown method')
    if not a.dry_run:
        receipt=read(videos/'export_receipt.json')
        if not receipt['complete'] or receipt['protocol']['samples']<5 or receipt['protocol']['flicker_samples']<25:raise RuntimeError('Full standard sampling and complete export required')
        if not set(methods)<=set(receipt['protocol']['methods']):raise RuntimeError('Requested method not in export')
        if sha(repo/'vbench/VBench_full_info.json')!=sha(ROOT/'data/VBench_full_info.json'):raise RuntimeError('External VBench metadata differs')
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if head!=read(ROOT/'data/vbench_source.json')['commit']:raise RuntimeError('VBench source commit differs')
        subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD'],check=True)
    for method in methods:
        source=videos/method;dest=output/method
        for dimension in a.dimensions:
            current=source
            identity=None if a.dry_run else fingerprint({'export_receipt_sha256':sha(videos/'export_receipt.json'),'method':method,'dimension':dimension,'source':str(source),'vbench_commit':head})
            if dimension=='temporal_flickering':
                filter_dir=dest/'static_filter';current=filter_dir/'filtered_videos'
                command=[a.python,str(repo/'static_filter.py'),'--videos_path',str(source),'--result_path',str(filter_dir)]
                if a.dry_run:print(shlex.join(command))
                else:
                    filter_dir.mkdir(parents=True,exist_ok=True)
                    filter_marker=filter_dir/'completed.json'
                    if not filter_marker.exists():
                        with (filter_dir/'filter.log').open('w') as log:subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
                        save(filter_marker,{'input_sha256':identity,'files':{str(f.relative_to(filter_dir)):sha(f) for f in filter_dir.rglob('*') if f.is_file() and f.name!='completed.json'}})
                    checked=read(filter_marker)
                    if checked['input_sha256']!=identity or any(sha(filter_dir/name)!=digest for name,digest in checked['files'].items()):raise RuntimeError('Static filter evidence differs')
                    filtered=read(filter_dir/'filtered_static_video.json');missing={k:v['static_count'] for k,v in filtered.items() if v['static_count']<5}
                    expected={r['prompt_en'] for r in read(ROOT/'data/VBench_full_info.json') if 'temporal_flickering' in r['dimension']}
                    if set(filtered)!=expected:raise RuntimeError('Static filter prompt coverage differs')
                    save(filter_dir/'coverage.json',{'complete':not missing,'missing':missing})
                    if missing:raise RuntimeError('Insufficient static samples for '+method+'; see '+str(filter_dir/'coverage.json'))
            result_dir=dest/dimension
            command=[a.python,str(repo/'evaluate.py'),'--videos_path',str(current),'--dimension',dimension,'--output_path',str(result_dir),'--mode','vbench_standard','--full_json_dir',str(ROOT/'data/VBench_full_info.json')]
            if a.dry_run:print(shlex.join(command));continue
            result_dir.mkdir(parents=True,exist_ok=True)
            marker=result_dir/'completed.json'
            if marker.exists():
                checked=read(marker)
                if checked['input_sha256']!=identity or any(sha(result_dir/name)!=digest for name,digest in checked['result_files'].items()):raise RuntimeError('Cached VBench result differs')
                continue
            with (result_dir/'evaluate.log').open('w') as log:subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
            results=list(result_dir.glob('*_eval_results.json'))
            if not results:raise RuntimeError('VBench returned without an evaluation result')
            save(marker,{'method':method,'dimension':dimension,'input_sha256':identity,'completed':time.time(),'result_files':{r.name:sha(r) for r in results}})

if __name__=='__main__':main()
