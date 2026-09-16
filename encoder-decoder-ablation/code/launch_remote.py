from pathlib import Path
import subprocess,json,sys
R=Path('/root/autodl-tmp/experiments/wan_encoder_decoder_unified_20260916')
sys.path.insert(0,str(R));from runtime import verify_frozen,gpu_pids
assert not (R/'launch.json').exists();verify_frozen()
with (R/'launcher.log').open('w') as log:
 p=subprocess.Popen(['/root/autodl-tmp/envs/wan-downscaler-full/bin/python',str(R/'run.py'),'all'],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
x=dict(pid=p.pid);(R/'launch.json').write_text(json.dumps(x));print(json.dumps(x))
