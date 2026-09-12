"""Full-frame video comparisons and explicitly resized LPIPS; no training use."""
import os
from pathlib import Path
import av
import numpy as np
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity


def read_video(path):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        fps=float(stream.average_rate)
        frames=np.stack([frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)])
    return frames,fps


def ssim_batch(x,y):
    """Match skimage default SSIM: 7x7 uniform window, sample covariance, valid crop.

    FP64 pooling preserves cancellation precision in smooth video regions.
    x,y are B,C,H,W floats in [0,1].
    """
    mean=lambda z:F.avg_pool2d(z,7,1)
    ux,uy=mean(x),mean(y)
    vx=(mean(x*x)-ux*ux)*(49/48)
    vy=(mean(y*y)-uy*uy)*(49/48)
    cov=(mean(x*y)-ux*uy)*(49/48)
    return (((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))).mean((1,2,3))


class VideoMetrics:
    def __init__(self):
        # TORCH_HOME follows the normal PyTorch cache convention on every host.
        assert Path(torch.hub.get_dir(),'checkpoints/alexnet-owt-7be5be79.pth').exists()
        import lpips
        self.lpips=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)

    @torch.inference_mode()
    def compare(self,reference,candidate):
        gt,fps=read_video(reference);pred,pfps=read_video(candidate)
        assert gt.shape==pred.shape and fps==pfps
        mse=[];ssim=[];temporal=[];lp=[]
        for start in range(0,len(gt),4):
            tensors=[]
            for video in (gt,pred):
                tensors.append(torch.from_numpy(video[start:start+4].copy()).cuda().permute(0,3,1,2))
            x,y=[v.double()/255 for v in tensors]
            mse.extend(((x-y).square().mean((1,2,3))*255**2).cpu().tolist())
            ssim.extend(ssim_batch(x,y).cpu().tolist())
            for i in range(start,min(start+4,len(gt))):
                if i:
                    da=gt[i].astype(np.float32)-gt[i-1].astype(np.float32)
                    db=pred[i].astype(np.float32)-pred[i-1].astype(np.float32)
                    temporal.append(float(np.abs(da-db).mean()/255))
            small=[F.interpolate(v.float()/127.5-1,(272,480),mode='bilinear',align_corners=False) for v in tensors]
            lp.extend(self.lpips(*small).flatten().cpu().tolist())
        mean_mse=float(np.mean(mse))
        return {'frames':len(gt),'width':gt.shape[2],'height':gt.shape[1],'fps':fps,
                'psnr_db':float(10*np.log10(255**2/max(mean_mse,1e-12))),
                'ssim_full_resolution':float(np.mean(ssim)),
                'lpips_alex_480x272':float(np.mean(lp)),
                'temporal_difference_mae_0_1':float(np.mean(temporal)),
                'per_frame_ssim':ssim,'per_frame_lpips':lp}


def comparison_video(reference,candidate,path):
    gt,fps=read_video(reference);pred,pfps=read_video(candidate)
    assert gt.shape==pred.shape and fps==pfps
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    from PIL import Image,ImageDraw
    contact=Image.new('RGB',(960,420),(20,20,20));draw=ImageDraw.Draw(contact)
    for row,(label,video) in enumerate((('Native 50 steps',gt),('Restoration + adaptive refinement',pred))):
        draw.text((8,row*210+5),label,fill='white')
        for column,index in enumerate((0,len(video)//2,len(video)-1)):
            contact.paste(Image.fromarray(video[index]).resize((320,181)),(column*320,row*210+27))
    contact.save(path.with_suffix('.jpg'),quality=92)
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=round(fps))
        stream.width=gt.shape[2]*2;stream.height=gt.shape[1];stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast'}
        for a,b in zip(gt,pred):
            frame=av.VideoFrame.from_ndarray(np.concatenate([a,b],axis=1),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
