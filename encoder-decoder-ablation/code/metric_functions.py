from pathlib import Path
import gc,hashlib,json,sys,math,traceback
R=Path('/root/autodl-tmp/experiments/wan_cue_coverage_pilot_20260915')
sys.path.insert(0,'/root/autodl-tmp/experiments/wan_paper_experiments_20260912')
from common import save,Guard,gpu_pids,verify_frozen
import torch,cv2,lpips,numpy as np
from skimage.metrics import structural_similarity
def video_read(path):
 cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
 cap.release();a=np.stack(frames);assert a.shape==(41,480,832,3) and abs(fps-16)<.01
 return a
@torch.inference_mode()
def metrics(model,pred,native):
 per=[];psnr=[];ssim=[]
 for start in range(0,41,8):
  pair=[torch.from_numpy(np.stack([cv2.resize(f,(416,240),interpolation=cv2.INTER_AREA) for f in x[start:start+8]])).permute(0,3,1,2).float().cuda()/127.5-1 for x in [pred,native]]
  per.extend(model(*pair).flatten().cpu().tolist())
 for a,b in zip(pred,native):
  mse=np.square(a.astype(np.float64)-b.astype(np.float64)).mean()
  psnr.append(10*math.log10(255**2/max(mse,1e-12)))
  ssim.append(float(structural_similarity(a,b,channel_axis=-1,data_range=255)))
 return {'lpips':float(np.mean(per)),'ssim':float(np.mean(ssim)),'psnr_db':float(np.mean(psnr)),'per_frame':{'lpips':per,'ssim':ssim,'psnr_db':psnr}}
