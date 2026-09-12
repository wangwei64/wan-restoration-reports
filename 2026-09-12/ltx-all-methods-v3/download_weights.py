"""Download the published owned restorer and verify every byte."""
import argparse,json,urllib.request
from pathlib import Path
from runtime import ROOT,read,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'weights/restorer.pt');a=p.parse_args()
    manifest=read(ROOT/'weights/manifest.json');row=manifest['restorer']
    if a.output.exists() and a.output.stat().st_size==row['bytes'] and sha(a.output)==row['sha256']:
        print(json.dumps({'verified':True,'path':str(a.output)}));return
    if a.output.exists():raise RuntimeError('Existing weight file differs; preserve it and choose a new --output path')
    if not manifest['published'] or not row['download_url']:raise RuntimeError('This package has not been published yet')
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.download')
    req=urllib.request.Request(row['download_url'],headers={'User-Agent':'LTX-All-Methods'})
    with urllib.request.urlopen(req,timeout=120) as response,tmp.open('wb') as out:
        while chunk:=response.read(4*1024*1024):out.write(chunk)
    if tmp.stat().st_size!=row['bytes'] or sha(tmp)!=row['sha256']:raise RuntimeError('Downloaded weight hash mismatch')
    tmp.replace(a.output);print(json.dumps({'verified':True,'path':str(a.output),'sha256':row['sha256']}))

if __name__=='__main__':main()
