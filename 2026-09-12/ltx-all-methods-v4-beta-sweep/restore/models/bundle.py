import json
import hashlib
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file
from .restorer import FeatureRestorer, LatentFeatures


class RestorationBundle(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.features=LatentFeatures()
        self.restorer=FeatureRestorer(config)

    def forward(self,z,sigma):
        features,heat=self.features(z,sigma)
        endpoint,confidence,aux=self.restorer(z,features,heat)
        return endpoint,confidence,{**aux,'heat':heat}

    def initialize_from_wan(self,path):
        details={}
        for name in ('features','restorer'):
            model=getattr(self,name)
            source=load_file(str(Path(path)/f'{name}.safetensors'))
            target=model.state_dict()
            matched={k:v for k,v in source.items() if k in target and v.shape==target[k].shape}
            model.load_state_dict(matched,strict=False)
            details[name]={'copied_parameters':sum(t.numel() for t in matched.values()),
                           'initialized_keys':[k for k in target if k not in matched]}
        # Begin at the native early clean prediction, rather than an arbitrary
        # cross-VAE projection. Hidden layers retain compatible initialization.
        for layer in (self.restorer.backbone.ending,self.restorer.backbone.detail_ending,
                      self.restorer.correction,self.restorer.quality[-1],self.features.condition[-1]):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None: nn.init.zeros_(layer.bias)
        return details

    @classmethod
    def load(cls,path,device='cuda'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model=cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model'],strict=True)
        model.checkpoint_path=str(Path(path).resolve())
        model.checkpoint_step=checkpoint['step']
        model.quality_training_steps=checkpoint.get('quality_training_steps',0)
        with open(path,'rb') as stream:model.checkpoint_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
        return model.to(device).eval().requires_grad_(False),checkpoint
