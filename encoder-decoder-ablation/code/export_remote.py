from pathlib import Path
import json,hashlib
R=Path('/root/autodl-tmp/experiments/wan_encoder_decoder_unified_20260916')
assert json.loads((R/'final_audit.json').read_text())['passed']
p=R/'evidence.zip';print(json.dumps(dict(zip_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),zip_size=p.stat().st_size,results=json.loads((R/'results.json').read_text()))))
