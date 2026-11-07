"""Verify every shipped file without importing GPU libraries."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'package_manifest.json').read_text())
    for name,meta in manifest['files'].items():
        p=ROOT/name
        if p.stat().st_size!=meta['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']:
            raise RuntimeError('Package checksum mismatch: '+name)
    print(json.dumps({'verified_files':len(manifest['files']),'sensitivity_sha256':manifest['files']['data/sensitivity_ltx_096.npz']['sha256']}))
if __name__=='__main__':main()
