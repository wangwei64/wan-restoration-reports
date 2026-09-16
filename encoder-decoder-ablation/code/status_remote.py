from pathlib import Path
import json
R=Path('/root/autodl-tmp/experiments/wan_encoder_decoder_unified_20260916');out={}
for n in ['status','worker','generation_validation','final_audit']:
 if (R/(n+'.json')).exists():out[n]=json.loads((R/(n+'.json')).read_text())
if (R/'registration.json').exists():
 p=json.loads((R/'registration.json').read_text());out['counts']={k:p[k] for k in ['total','reuse_count','new_count']}
if any(v.get('stage')=='failed' for v in out.values()):out['logs']={f.name:f.read_text(errors='replace')[-3500:] for f in R.glob('*.log')}
print(json.dumps(out))
