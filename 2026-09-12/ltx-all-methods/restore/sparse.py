"""Selected queries with two request-local BF16 K/V anchors and real residuals.

Input coordinates stay in the original full-grid coordinate system. The official
transformer runs its own normalization, RoPE, cross attention and MLP on selected
tokens. Only self-attention context is reconstructed from online anchors.
"""
import torch
from torch.nn import functional as F
from .kernels.context import materialize_restoration_context,query_inverse
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

class SparseLTX:
    def __init__(self,model,fused_context=False,shared_prefix=False):
        self.shared_prefix=shared_prefix
        self.fused_context=fused_context
        self.model=model;self.original=[];self.hooks=[];self.reset()
        for i,block in enumerate(model.transformer_blocks):
            original=block.attn1.processor;self.original.append((block.attn1,original))
            block.attn1.set_processor(self.processor(i,original))
            if i in (6,13,20):self.hooks.append(block.attn2.register_forward_pre_hook(self.observe_prompt,with_kwargs=True))

    def reset(self):
        self.mode='native';self.slot='left';self.anchors={};self.residual={}
        self.indices=None;self.fraction=1.;self.saliency_maps=[];self.maximum_bytes=0
        self.context_buffers={};self.context_query_map=None

    def close(self):
        for module,original in self.original:module.set_processor(original)
        for hook in self.hooks:hook.remove()
        self.reset()

    def observe_prompt(self,attention,args,kwargs):
        if self.mode!='capture' or self.slot!='left':return
        x=args[0];context=kwargs['encoder_hidden_states'];mask=kwargs.get('attention_mask')
        # Native positive branch. A spatial map of selective prompt response;
        # excludes padding and the first token (often an article).
        q=attention.q_norm(attention.to_q(x[1:2])).float()[0]
        k=attention.k_norm(attention.to_k(context[1:2])).float()[0]
        valid=torch.ones(len(k),device=k.device,dtype=torch.bool)
        if mask is not None:valid=mask[1].reshape(-1)>=0
        valid[0]=False;k=k[valid]
        if len(k)<2:return
        k=k-k.mean(0,keepdim=True)
        scores=(q@k.T)/(attention.heads*(q.shape[-1]//attention.heads)**.5)
        lo,hi=torch.quantile(scores,torch.tensor([.05,.95],device=x.device),dim=0)
        maps=((scores-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1)
        weights=scores.std(0).square();weights/=weights.sum().clamp_min(1e-6)
        self.saliency_maps.append((maps*weights).sum(-1))

    def saliency(self,n,device):
        return torch.stack(self.saliency_maps).mean(0) if self.saliency_maps else torch.full((n,),.5,device=device)

    def context(self,layer,key,value):
        if self.mode=='capture':
            cache_key,cache_value=(key[:2],value[:2]) if self.shared_prefix and layer<19 else (key,value)
            self.anchors.setdefault(layer,{})[self.slot]=(cache_key.clone(),cache_value.clone())
            return key,value
        if self.mode!='sparse':return key,value
        left=self.anchors[layer]['left'];right=self.anchors[layer]['right']
        if self.fused_context:
            if layer not in self.residual:self.residual[layer]=tuple(torch.zeros_like(x) for x in left)
            shape=tuple(left[0].shape)
            if shape not in self.context_buffers:self.context_buffers[shape]=tuple(torch.empty_like(x) for x in left)
            outputs=self.context_buffers[shape]
            view=lambda x:x.view(x.shape[0],x.shape[1],1,x.shape[2])
            materialize_restoration_context(*[view(x) for x in (*left,*right,*self.residual[layer],key,value)],self.context_query_map,*[view(x) for x in outputs],self.fraction)
            return outputs
        if key.shape[1]==left[0].shape[1]:
            # All K/V are current. Retain only the real residual needed by a
            # later partial refresh; materializing a second full context is idle work.
            self.residual[layer]=tuple(current-torch.lerp(hi,lo,self.fraction)
                for current,lo,hi in zip((key,value),left,right))
            return key,value
        if layer not in self.residual:self.residual[layer]=tuple(torch.zeros_like(x) for x in left)
        result=[]
        for current,lo,hi,delta in zip((key,value),left,right,self.residual[layer]):
            base=torch.lerp(hi,lo,self.fraction)
            delta.index_copy_(1,self.indices,current-base.index_select(1,self.indices))
            full=base+delta
            full.index_copy_(1,self.indices,current)
            result.append(full)
        return tuple(result)

    def bytes(self):
        values=[x for d in self.anchors.values() for pair in d.values() for x in pair]
        values += [x for pair in self.residual.values() for x in pair]
        return sum(x.numel()*x.element_size() for x in values)

    def processor(self,layer,original):
        def process(attn,hidden_states,freqs_cis,encoder_hidden_states=None,attention_mask=None,
                    temb=None,skip_layer_mask=None,skip_layer_strategy=None,**kwargs):
            if self.mode=='native':
                return original(attn,hidden_states,freqs_cis,encoder_hidden_states,attention_mask,temb,skip_layer_mask,skip_layer_strategy,**kwargs)
            assert hidden_states.ndim==3 and encoder_hidden_states is None
            assert attn.spatial_norm is None and attn.group_norm is None and attention_mask is None
            residual=hidden_states;b=hidden_states.shape[0];h=attn.heads
            query=attn.q_norm(attn.to_q(hidden_states))
            key=attn.k_norm(attn.to_k(hidden_states))
            if attn.use_rope:
                key=attn.apply_rotary_emb(key,freqs_cis);query=attn.apply_rotary_emb(query,freqs_cis)
            value=attn.to_v(hidden_states);stg_value=value
            key,value=self.context(layer,key,value)
            d=query.shape[-1]//h
            output=F.scaled_dot_product_attention(query.view(b,-1,h,d).transpose(1,2),
                key.view(b,-1,h,d).transpose(1,2),value.view(b,-1,h,d).transpose(1,2),
                dropout_p=0.,is_causal=False).transpose(1,2).reshape(b,-1,h*d).to(query.dtype)
            if skip_layer_mask is not None:
                mask=skip_layer_mask.reshape(b,1,1)
                if skip_layer_strategy==SkipLayerStrategy.AttentionValues:output=output*mask+stg_value*(1-mask)
                elif skip_layer_strategy==SkipLayerStrategy.AttentionSkip:output=output*mask+hidden_states*(1-mask)
            output=attn.to_out[1](attn.to_out[0](output))
            if attn.residual_connection:
                output=output+residual*(skip_layer_mask.reshape(b,1,1) if skip_layer_mask is not None and skip_layer_strategy==SkipLayerStrategy.Residual else 1)
            return output/attn.rescale_output_factor
        return process
