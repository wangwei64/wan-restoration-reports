"""Run one method or inspect the installation without loading a GPU model."""
import argparse,json
from pathlib import Path
from runtime import ROOT,Runner,configuration,doctor,paths,read,verify_external_models

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=list(read(ROOT/'configs/methods.json')),default='native')
    parser.add_argument('--prompt');parser.add_argument('--seed',type=int,default=20260912)
    parser.add_argument('--out',type=Path);parser.add_argument('--paths',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--doctor',action='store_true');parser.add_argument('--verify-weights',action='store_true');parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    if args.describe:
        print(json.dumps({'generation':configuration(args.config),'methods':read(ROOT/'configs/methods.json')},indent=2));return
    if args.doctor:
        location=paths(args.paths);result=doctor(location,verify_weights=args.verify_weights)
        if args.verify_weights:result.update(verify_external_models(location))
        print(json.dumps(result,indent=2));return
    if not args.prompt or not args.out:parser.error('--prompt and --out are required for generation')
    run=Runner(args.paths,args.config).generate(args.prompt,args.seed,args.method,args.out)
    print(json.dumps({'video':str((args.out/'video.mp4').resolve()),'method':args.method,'online_seconds':run['online_seconds'],'exclusive_gpu_timing_verified':run['exclusive_gpu_timing_verified']}))

if __name__=='__main__':main()
