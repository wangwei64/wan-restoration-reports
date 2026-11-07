"""Fetch the exact public official base weights into a user-selected external dir."""
import argparse,json
from pathlib import Path
import shutil
from runtime import ROOT,read,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    from huggingface_hub import hf_hub_download
    sources=read(ROOT/'configs/external_download_sources.json');manifest=read(ROOT/'configs/external_models.json')
    root=a.directory.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    for kind in ['checkpoint','text_encoder_official']:
        spec=sources[kind]
        for name,item in spec['files'].items():
            target=root/name if kind=='checkpoint' else root/'text_encoder_reference'/name
            expected=manifest['checkpoint'] if kind=='checkpoint' else manifest['text_encoder_safetensors'][name]
            if target.exists():
                if target.stat().st_size!=expected['bytes'] or sha(target)!=expected['sha256']:raise RuntimeError('Existing external file differs: '+str(target))
                continue
            cached=Path(hf_hub_download(spec['repository'],item['remote_path'],revision=spec['revision']))
            if cached.stat().st_size!=expected['bytes'] or sha(cached)!=expected['sha256']:raise RuntimeError('Official download checksum differs: '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            try:target.symlink_to(cached.resolve())
            except OSError:shutil.copyfile(cached,target)
    print(json.dumps({'LTX_CHECKPOINT':str(root/manifest['checkpoint']['filename']),'LTX_TEXT_ENCODER':str(root/'text_encoder_reference'),'verified':True}))

if __name__=='__main__':main()
