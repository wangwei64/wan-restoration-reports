"""One generation protocol and reversible method selection for external LTX."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(path)
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def completed_run(out,key):
    out=Path(out)
    if not (out/'completed.json').exists():return None
    done=read(out/'completed.json')
    if done['request_sha256']!=key or done['video_sha256']!=sha(out/'video.mp4') or done['run_sha256']!=sha(out/'run.json'):raise RuntimeError('Existing output does not match request or checksums: '+str(out))
    return read(out/'run.json')

def restoration_config(preset):
    config=dict(read(ROOT/'configs/base_profile.json')['controller'])
    config.update(preset['controller_overrides']);config['budget']=preset['beta']
    calibration=read(ROOT/'configs/calibration.json')
    return config,calibration

def configuration(path=None):
    value=read(path or ROOT/'configs/generation.json')
    required=set(read(ROOT/'configs/generation.json'))
    if set(value)!=required:raise ValueError('Generation config must contain all and only the documented fields')
    if value['guidance_scale']<=1 or value['stg_scale']<=0:raise ValueError('These adapters require native negative/positive/STG batch=3')
    if value['height']%32 or value['width']%32 or (value['frames']-1)%8:raise ValueError('LTX shape must be 32n x 32n, 8n+1 frames')
    if value['steps']!=50:raise ValueError('Published cache schedules and quality presets were calibrated for 50 steps')
    return value

def paths(path=None):
    values=read(path) if path else {}
    for key,env in [('ltx_repo','LTX_REPO'),('checkpoint','LTX_CHECKPOINT'),('text_encoder','LTX_TEXT_ENCODER')]:
        val=os.environ.get(env,values.get(key))
        if not val:raise ValueError('Set '+env+' or supply --paths configs/paths.local.json')
        values[key]=str(Path(val).expanduser().resolve())
        if not Path(values[key]).exists():raise FileNotFoundError(values[key])
    values['restorer']=str(Path(os.environ.get('LTX_RESTORER',values.get('restorer',ROOT/'weights/restorer.pt'))).expanduser().resolve())
    return values

def doctor(values,verify_weights=False):
    provenance=read(ROOT/'configs/training_provenance.json')['environment']
    for name,expected in provenance['native_source_sha256'].items():
        if sha(Path(values['ltx_repo'])/name)!=expected:raise RuntimeError('External LTX source mismatch: '+name)
    packages={name:importlib.metadata.version(name) for name in provenance['packages']}
    # Match the validated inference stack; version drift requires explicit new validation.
    for name,expected in provenance['packages'].items():
        if packages[name]!=expected:raise RuntimeError(f'Expected {name}=={expected}; found {packages[name]}')
    result={'external_source_verified':True,'packages':packages,'ltx_commit':provenance['declared_upstream_commit']}
    if verify_weights:
        pin=read(ROOT/'weights/manifest.json')
        if sha(values['restorer'])!=pin['restorer']['sha256']:raise RuntimeError('Restorer checksum mismatch')
        result['restorer_sha256']=pin['restorer']['sha256']
    return result

def verify_external_models(values):
    manifest=read(ROOT/'configs/external_models.json')
    actual=sha(values['checkpoint'])
    if actual!=manifest['checkpoint']['sha256']:raise RuntimeError('Expected original LTX 2B 0.9.6 checkpoint; checksum differs')
    safe_index=Path(values['text_encoder'])/'text_encoder/model.safetensors.index.json'
    safe_single=Path(values['text_encoder'])/'text_encoder/model.safetensors'
    if safe_single.exists():raise RuntimeError('Unexpected single-file text encoder checkpoint')
    if (Path(values['text_encoder'])/'text_encoder/pytorch_model.bin').exists():raise RuntimeError('Unexpected single-file binary text encoder checkpoint')
    variant='text_encoder_safetensors' if safe_index.exists() else 'text_encoder'
    for name,meta in manifest[variant].items():
        p=Path(values['text_encoder'])/name
        if p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:raise RuntimeError('Text encoder file differs: '+name)
    return {'checkpoint_sha256':actual,'text_encoder_manifest_sha256':fingerprint(manifest[variant]),'text_encoder_storage':variant}

class GPUContended(RuntimeError):pass

def selected_gpu_uuid(cuda_uuid):
    # PyTorch 2.5 exposes the UUID without NVML's GPU- prefix.
    try:candidate='GPU-'+str(uuid.UUID(str(cuda_uuid).removeprefix('GPU-')))
    except ValueError as error:raise RuntimeError('CUDA GPU UUID unavailable') from error
    available=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10)
    matches=[line.strip() for line in available.splitlines() if line.strip().lower()==candidate.lower()]
    if len(matches)!=1:raise RuntimeError('Selected CUDA GPU does not match an NVML GPU UUID')
    return matches[0]

class TimingGuard:
    def __init__(self,gpu_uuid):
        self.uuid=gpu_uuid;self.seen=set();self.errors=[];self.stop=threading.Event()
    def sample(self):
        try:
            output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
            for line in output.splitlines():
                fields=[x.strip() for x in line.split(',')]
                if len(fields)==2 and fields[0]==self.uuid and int(fields[1])!=os.getpid():self.seen.add(int(fields[1]))
        except Exception as exc:self.errors.append(type(exc).__name__)
    def verify(self):
        if self.errors:raise RuntimeError('Cannot validate exclusive timing: '+','.join(self.errors))
        if self.seen:raise GPUContended('Other processes overlapped this GPU: '+str(sorted(self.seen)))
    def poll(self):
        while not self.stop.wait(1):self.sample()
    def __enter__(self):
        self.sample();self.verify();self.worker=threading.Thread(target=self.poll,daemon=True);self.worker.start();return self
    def __exit__(self,*args):self.stop.set();self.worker.join();self.sample()

class Runner:
    def __init__(self,path_config=None,generation_config=None):
        os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
        self.paths=paths(path_config);self.generation=configuration(generation_config)
        self.doctor=doctor(self.paths)
        sys.path.insert(0,self.paths['ltx_repo'])
        import torch
        from restore.native import NativeAdapter
        if torch.cuda.device_count()!=1:raise RuntimeError('Select exactly one GPU per process using CUDA_VISIBLE_DEVICES')
        self.torch=torch;properties=torch.cuda.get_device_properties(0);self.gpu_uuid=selected_gpu_uuid(properties.uuid)
        self.hardware={'gpu_name':properties.name,'gpu_uuid':self.gpu_uuid,'compute_capability':[properties.major,properties.minor],'total_memory_bytes':properties.total_memory,'software':self.doctor['packages']}
        code_files=[*ROOT.joinpath('restore').rglob('*.py'),*ROOT.joinpath('accelerators').rglob('*.py'),ROOT/'runtime.py']
        self.identity={'code_sha256':fingerprint({str(p.relative_to(ROOT)):sha(p) for p in code_files}),
                       **verify_external_models(self.paths),'external_source_sha256':fingerprint(read(ROOT/'configs/training_provenance.json')['environment']['native_source_sha256'])}
        self.adapter=NativeAdapter(self.paths['checkpoint'],self.paths['text_encoder'])
        self.methods=read(ROOT/'configs/methods.json');self.restorer=None

    def generate(self,prompt,seed,method,out,keep_latents=True):
        from accelerators.teacache_adapter import TeaCacheLTX
        from accelerators.cache_adapters import NaviCacheLTX,SenCacheLTX
        from accelerators.profiling_adapter import ProfilingDiTLTX
        from restore.pipeline import RestorationController
        from restore.models.bundle import RestorationBundle
        if method not in self.methods:raise ValueError('Unknown method: '+method)
        preset=self.methods[method]
        if preset.get('pending_validation'):raise RuntimeError('ProfilingDiT validation has not frozen this preset yet')
        if preset['method']=='ours' and self.restorer is None:
            doctor(self.paths,verify_weights=True)
            self.restorer,_=RestorationBundle.load(self.paths['restorer'])
        identity=dict(self.identity)
        config=calibration=None
        if preset['method']=='ours':
            config,calibration=restoration_config(preset)
            identity.update(restorer_sha256=self.restorer.checkpoint_sha256,controller_config_sha256=fingerprint(config),calibration_sha256=fingerprint(calibration))
        if preset['method']=='sen':identity['sensitivity_sha256']=sha(ROOT/'data/sensitivity_ltx.npz')
        request={'prompt':prompt,'seed':seed,'method':method,'preset':preset,'generation':self.generation,'identity':identity}
        key=fingerprint(request);out=Path(out).resolve()
        previous_run=completed_run(out,key)
        if previous_run is not None:return previous_run
        if out.exists():
            previous=out.parent/'partial_runs'/(out.name+'_'+str(time.time_ns()))
            previous.parent.mkdir(exist_ok=True);out.rename(previous)
        cache=None;controller=None;model=self.adapter.pipe.transformer;steps=self.generation['steps']
        # Begin GPU ownership checks before installing hooks; a failure restores them.
        guard=TimingGuard(self.gpu_uuid)
        try:
            if preset['method']=='tea':cache=TeaCacheLTX(model,preset['threshold'],steps,10)
            elif preset['method']=='navi':cache=NaviCacheLTX(model,preset['threshold'],steps,10)
            elif preset['method']=='sen':cache=SenCacheLTX(model,preset['threshold'],ROOT/'data/sensitivity_ltx.npz',steps,10,preset['K'])
            elif preset['method']=='profiling':cache=ProfilingDiTLTX(model,preset['background_blocks'],steps,10,preset['max_interval'],preset['min_interval'])
            elif preset['method']=='ours':
                controller=RestorationController(self.restorer,calibration=calibration,**config)
            elif preset['method']!='native':raise ValueError(preset)
            with guard:run=self.adapter.generate(prompt,seed,out,controller=controller,capture=(),decode=True,**self.generation)
            guard.verify()
            final=self.torch.load(out/'latents.pt',map_location='cpu',weights_only=False)['snapshots']['final']
            if not bool(self.torch.isfinite(final).all()):raise RuntimeError('Non-finite output')
            if cache:run.update(cache.report())
            run.update(method=method,method_preset=preset,request=request,request_sha256=key,hardware_gpu_uuid=self.gpu_uuid,hardware=self.hardware,exclusive_gpu_timing_verified=True)
            save(out/'run.json',run)
            save(out/'completed.json',{'request_sha256':key,'video_sha256':sha(out/'video.mp4'),'run_sha256':sha(out/'run.json'),'completed':time.time()})
            if not keep_latents:(out/'latents.pt').unlink()
            return run
        except GPUContended:
            if out.exists():save(out/'timing_invalid.json',{'other_gpu_pids':sorted(guard.seen),'counted_in_benchmark':False})
            raise
        finally:
            if cache:cache.close()
            self.torch.cuda.empty_cache()
