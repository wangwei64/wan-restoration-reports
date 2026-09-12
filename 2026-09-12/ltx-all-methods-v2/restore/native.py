"""Pinned official LTX pipeline; optional reversible online controller hooks."""
import json, os, time
from pathlib import Path
import av
import numpy as np
import torch
from ltx_video.inference import create_ltx_video_pipeline, seed_everething
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.schedulers.rf import simple_diffusion_resolution_dependent_timestep_shift
from restore.video import encode_rgb

NEGATIVE='worst quality, inconsistent motion, blurry, jittery, distorted'

def sync(): torch.cuda.synchronize()

def save_video(video,path,fps=30):
    # Official pipeline returns B,C,T,H,W in [0,1].
    frames=(video[0].permute(1,2,3,0).float().clamp(0,1)*255).round().byte().contiguous().cpu().numpy()
    encoding=encode_rgb(frames,path,fps)
    from PIL import Image
    contact=Image.new('RGB',(960,186))
    for k,i in enumerate((0,len(frames)//2,len(frames)-1)):
        contact.paste(Image.fromarray(frames[i]).resize((320,186)),(320*k,0))
    contact.save(Path(path).with_suffix('.jpg'),quality=90)
    return encoding

class NativeAdapter:
    def __init__(self, checkpoint, text_encoder):
        self.pipe=create_ltx_video_pipeline(
            str(checkpoint),'bfloat16',
            str(text_encoder),sampler='from_checkpoint',device='cuda',enhance_prompt=False)
        for model in (self.pipe.transformer,self.pipe.vae,self.pipe.text_encoder):
            model.eval().requires_grad_(False)
        self.pipe.set_progress_bar_config(disable=True)
        print(json.dumps({'stage':'loaded','transformer_parameters':sum(p.numel() for p in self.pipe.transformer.parameters()),'scheduler':dict(self.pipe.scheduler.config)}),flush=True)

    @torch.inference_mode()
    def generate(self,prompt,seed,out,height=704,width=1216,frames=121,steps=50,
                 capture=(9,10,15,20,25,30,40,50),decode=True,controller=None,timesteps=None,
                 flow_scale=.1,dynamic_flow=True,fps=30,negative_prompt=NEGATIVE,
                 guidance_scale=3.,stg_scale=1.,rescaling_scale=.7,skip_block_list=(19,),
                 decode_timestep=.05,decode_noise_scale=.025):
        if height%32 or width%32 or (frames-1)%8 or min(height,width)<32 or frames<9:
            raise ValueError('Use height/width divisible by 32 and frames=8n+1 (at least 9).')
        root=Path(out);root.mkdir(parents=True,exist_ok=False)
        p=self.pipe;seed_everething(seed);sync();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();snapshots={};counter=0;denoise_done=None
        original=p.denoising_step
        tokens=(frames//8+1)*(height//32)*(width//32)
        effective_flow_scale=flow_scale*(tokens/13376 if dynamic_flow else 1.)
        if flow_scale<=0:raise ValueError('flow_scale must be positive')
        if dynamic_flow or flow_scale!=1.:
            if timesteps is not None:raise ValueError('Supply a flow scale or explicit timesteps, not both.')
            initial=p.scheduler.get_initial_timesteps(steps).double()
            timesteps=simple_diffusion_resolution_dependent_timestep_shift(
                (1,128,frames//8+1,height//32,width//32),initial,n=tokens/effective_flow_scale).float().tolist()
        def observe(latents,noise_pred,current_timestep,conditioning_mask,t,extra_step_kwargs,t_eps=1e-6,stochastic_sampling=False):
            nonlocal counter,denoise_done
            counter+=1
            if controller is not None:
                result=controller.step(latents,noise_pred,current_timestep,conditioning_mask,t,extra_step_kwargs,t_eps,stochastic_sampling)
            else:
                result=original(latents,noise_pred,current_timestep,conditioning_mask,t,extra_step_kwargs,t_eps,stochastic_sampling)
            if counter in capture:
                snapshots[str(counter)]={'clean':(latents.float()-float(t)*noise_pred.float()).clone(),'sigma':float(t)}
            if counter==steps:
                snapshots['final']=result.clone()
                if controller is not None:controller.release_cache()
                sync();denoise_done=time.perf_counter()
            status={'stage':'sampling' if counter<steps else 'decoding','step':counter,'elapsed':time.perf_counter()-start}
            if controller is not None and controller.policy is not None and controller.policy.events:status.update(controller.policy.events[-1])
            (root/'status.json').write_text(json.dumps(status))
            if counter%5==0:print(json.dumps({'out':str(root),'step':counter,'seconds':time.perf_counter()-start}),flush=True)
            return result
        if controller is not None:controller.begin(p,original,(frames//8+1,height//32,width//32),steps)
        p.denoising_step=observe
        try:
            result=p(height=height,width=width,num_frames=frames,frame_rate=fps,prompt=prompt,negative_prompt=negative_prompt,
                num_inference_steps=steps,timesteps=timesteps,guidance_scale=guidance_scale,stg_scale=stg_scale,rescaling_scale=rescaling_scale,
                skip_layer_strategy=SkipLayerStrategy.AttentionValues,skip_block_list=list(skip_block_list),
                generator=torch.Generator('cuda').manual_seed(seed),output_type='pt' if decode else 'latent',
                decode_timestep=decode_timestep,decode_noise_scale=decode_noise_scale,vae_per_channel_normalize=True,
                enhance_prompt=False,stochastic_sampling=False,offload_to_cpu=False,is_video=True).images
            sync();decode_done=time.perf_counter()
            encoding=save_video(result,root/'video.mp4',fps=fps) if decode else {}
            sync();elapsed=time.perf_counter()-start
            report={'prompt':prompt,'negative_prompt':negative_prompt,'seed':seed,'height':height,'width':width,'frames':frames,'fps':fps,'steps':steps,
                    'online_seconds':elapsed,'text_and_sampling_seconds':denoise_done-start,
                    'decode_seconds':decode_done-denoise_done,'save_video_seconds':time.perf_counter()-decode_done,
                    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'sigmas':p.scheduler.timesteps.cpu().tolist()+[0.],
                    'flow_scale':flow_scale,'dynamic_flow':dynamic_flow,'effective_flow_scale':effective_flow_scale,
                    'flow_mode':'official SimpleDiffusion shape formula' if dynamic_flow else ('scaled LinearQuadratic' if flow_scale!=1 else 'official checkpoint'),
                    'scheduler':dict(p.scheduler.config),'guidance_scale':guidance_scale,'stg_scale':stg_scale,'rescaling_scale':rescaling_scale,
                    'skip_block_list':list(skip_block_list),'decode_timestep':decode_timestep,'decode_noise_scale':decode_noise_scale,
                    'prompt_enhancement':False,'mode':'native' if controller is None else 'restore_fast',
                    'decoded':decode,'video_pixel_transfer':'contiguous RGB uint8; x264 fast CRF18; fixed one encoder thread','transformer_parameters':sum(param.numel() for param in p.transformer.parameters()),
                    'timing_includes':['text encoding','sampling','restoration and context if fast','decode if enabled','video encoding if enabled'],
                    'timing_excludes':['model loading','offline training','latent serialization','quality metrics'],
                    'teacher_used_online':False,'full_reference_run_online':False}
            report.update(encoding)
            if controller is not None:report.update(controller.report())
            # Latent serialization is offline, after all online timing.
            cpu_snaps={k:({kk:(vv.cpu() if torch.is_tensor(vv) else vv) for kk,vv in v.items()} if isinstance(v,dict) else v.cpu()) for k,v in snapshots.items()}
            torch.save({'snapshots':cpu_snaps,'shape':[frames//8+1,height//32,width//32],'request':report},root/'latents.pt')
            (root/'run.json').write_text(json.dumps(report,indent=2))
            (root/'status.json').write_text(json.dumps({'stage':'complete','online_seconds':elapsed}))
            print(json.dumps({'out':str(root),'stage':'complete',**{k:report[k] for k in ('mode','steps','online_seconds','peak_gpu_allocated_gib')}}),flush=True)
            return report
        finally:
            p.denoising_step=original
            if controller is not None:controller.close()
