"""Score every frame against matching native videos with the frozen metric sizes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from runtime import read,save,sha
from restore.metrics import read_video,ssim_batch

class CommonMetrics:
    def __init__(self):
        import lpips
        self.lpips=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    @torch.inference_mode()
    def compare(self,reference,prediction):
        gt,fps=read_video(reference);pred,pfps=read_video(prediction)
        if gt.shape!=pred.shape or fps!=pfps:raise ValueError('Reference shape/fps differs')
        ssims=[];distances=[]
        for start in range(0,len(gt),4):
            images=[torch.from_numpy(v[start:start+4].copy()).cuda().permute(0,3,1,2).float()/255 for v in (gt,pred)]
            common=[F.interpolate(x,size=(480,832),mode='area').double() for x in images]
            ssims.extend(ssim_batch(*common).cpu().tolist())
            small=[F.interpolate(x,size=(240,416),mode='area')*2-1 for x in images]
            distances.extend(self.lpips(*small).flatten().cpu().tolist())
        return {'ssim':float(np.mean(ssims)),'lpips':float(np.mean(distances)),'per_frame_ssim':ssims,'per_frame_lpips':distances,'frames':len(gt),'fps':fps,'reference_sha256':sha(reference),'prediction_sha256':sha(prediction)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.method=='native':raise ValueError('Choose a non-native method')
    metric=CommonMetrics();cached=read(a.output) if a.output.exists() else []
    rows={(r['id'],r['sample_index']):r for r in cached}
    markers=sorted((a.results/a.method).glob('vbench_*/*/completed.json'))
    if not markers:raise RuntimeError('No completed prediction videos')
    for marker in markers:
        out=marker.parent;cid=out.parent.name;index=out.name;native=a.results/'native'/cid/index
        ref_done=read(native/'completed.json');pred_done=read(marker)
        gt=native/'video.mp4';pred=out/'video.mp4'
        if sha(gt)!=ref_done['video_sha256'] or sha(pred)!=pred_done['video_sha256']:raise RuntimeError('Video checksum mismatch')
        left,right=read(native/'run.json'),read(out/'run.json')
        if any(left['request'][k]!=right['request'][k] for k in ['prompt','seed','generation']):raise RuntimeError('Mismatched reference request')
        key=(cid,index)
        if key in rows:
            if rows[key]['reference_sha256']!=ref_done['video_sha256'] or rows[key]['prediction_sha256']!=pred_done['video_sha256']:raise RuntimeError('Cached score no longer matches video')
        else:
            rows[key]={'id':cid,'sample_index':index,'method':a.method,'seconds':right['online_seconds'],'native_seconds':left['online_seconds'],**metric.compare(gt,pred)}
            save(a.output,list(rows.values()))
    values=list(rows.values());protocol_path=a.results/'benchmark_protocol.json'
    expected=None
    if protocol_path.exists():
        protocol=read(protocol_path);expected=protocol['total_jobs']//len(protocol['methods'])
    save(a.output.with_name(a.output.stem+'_summary.json'),{'scored_videos':len(values),'available_prediction_videos':len(markers),'expected_videos_for_method':expected,'full_benchmark_complete':expected is not None and len(values)==expected,'mean_ssim':float(np.mean([r['ssim'] for r in values])),'mean_lpips':float(np.mean([r['lpips'] for r in values])),'speedup':sum(r['native_seconds'] for r in values)/sum(r['seconds'] for r in values)})

if __name__=='__main__':main()
