"""TeaCache4LTX algorithm adapted to pinned native LTX 0.9.6 via reversible hooks.

The signal, polynomial and residual boundary follow ali-vilab/TeaCache:
TeaCache4LTX-Video/teacache_ltx.py. Native LTX keeps its CFG/STG batches,
normalizations, scheduler and output projection. A user-matched 10-step full
warmup is added; the final step is always fully evaluated.
"""
import math
import numpy as np
import torch
from ltx_video.models.transformers.transformer3d import Transformer3DModelOutput

COEFFICIENTS = [2.14700694e1, -1.28016453e1, 2.31279151, .792487521, .00969274326]


class _CacheHit(Exception):
    pass


class TeaCacheLTX:
    def __init__(self, model, threshold, steps=50, warmup_steps=10):
        assert threshold >= 0 and 1 <= warmup_steps < steps
        assert not model.training and not model.use_tpu_flash_attention
        block = model.transformer_blocks[0]
        assert block.adaptive_norm == 'single_scale_shift'
        assert block.scale_shift_table.shape[0] == 6
        self.model = model
        self.threshold = float(threshold)
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.previous_modulated = None
        self.previous_residual = None
        self.projected_input = None
        self.accumulated = 0.
        self.index = 0
        self.replaying = False
        self.events = []
        self.original_forward = model.forward
        self.handles = [
            block.register_forward_pre_hook(self.before_first_block, with_kwargs=True),
            model.proj_out.register_forward_pre_hook(self.before_output_projection),
        ]
        model.forward = self.forward

    def before_first_block(self, block, args, kwargs):
        hidden = args[0] if args else kwargs['hidden_states']
        assert hidden.shape[0] == 3, 'Expected native negative/positive/STG batch.'
        self.projected_input = hidden
        temb = kwargs['timestep']
        values = block.scale_shift_table[None, None] + temb.reshape(hidden.shape[0], temb.shape[1], 6, -1)
        shift, scale = values.unbind(dim=2)[:2]
        modulated = block.norm1(hidden) * (1 + scale) + shift
        distance = increment = None
        forced = self.index < self.warmup_steps or self.index == self.steps - 1 or self.threshold == 0
        if forced:
            compute = True
            self.accumulated = 0.
        else:
            assert self.previous_modulated is not None and self.previous_residual is not None
            # Preserve official BF16 subtraction/abs/mean and polynomial evaluation.
            distance = float(((modulated - self.previous_modulated).abs().mean() /
                              self.previous_modulated.abs().mean()).cpu().item())
            assert math.isfinite(distance)
            increment = float(np.polyval(COEFFICIENTS, distance))
            self.accumulated += increment
            compute = self.accumulated >= self.threshold
        accumulated_before_reset = self.accumulated
        if compute:
            self.accumulated = 0.
        self.previous_modulated = modulated
        self.events.append({'step': self.index + 1, 'full_compute': compute, 'forced': forced,
                            'input_rel_l1': distance, 'rescaled_increment': increment,
                            'accumulated_before_reset': accumulated_before_reset})
        if not compute:
            raise _CacheHit()

    def before_output_projection(self, module, args):
        if not self.replaying:
            # Match official residual: after output norm + modulation, before proj_out.
            self.previous_residual = args[0] - self.projected_input

    def forward(self, *args, **kwargs):
        assert self.index < self.steps
        try:
            output = self.original_forward(*args, **kwargs)
        except _CacheHit:
            self.replaying = True
            try:
                projected = self.model.proj_out(self.projected_input + self.previous_residual)
            finally:
                self.replaying = False
            output = Transformer3DModelOutput(sample=projected) if kwargs.get('return_dict', True) else (projected,)
        self.index += 1
        self.projected_input = None
        if self.index == self.steps:
            self.previous_modulated = self.previous_residual = None
        return output

    def report(self):
        assert self.index == self.steps and len(self.events) == self.steps
        full = sum(x['full_compute'] for x in self.events)
        return {'algorithm': 'TeaCache4LTX native adaptation', 'rel_l1_thresh': self.threshold,
                'teacache_coefficients': COEFFICIENTS, 'full_warmup_steps': self.warmup_steps,
                'final_step_full': True, 'full_transformer_steps': full, 'cached_transformer_steps': self.steps - full,
                'transformer_compute_ratio': full / self.steps, 'teacache_trace': self.events,
                'restoration_network_used': False, 'custom_hardware_fusions': False,
                'cache_scope': 'one generation request, all three native guidance branches',
                'residual_boundary': 'post output normalization and modulation, pre output projection'}

    def close(self):
        self.model.forward = self.original_forward
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.projected_input = self.previous_modulated = self.previous_residual = None

