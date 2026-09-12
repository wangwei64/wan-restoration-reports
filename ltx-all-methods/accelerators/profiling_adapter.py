"""Thin LTX bridge to pinned, unmodified upstream ProfilingDiT Wan caching."""
from .vendor.profilingdit.cache import DitCache

UPSTREAM_COMMIT = 'fa9d1983418b481cc37eb69e7f6c5a3ec6e182a1'


class ProfilingDiTLTX:
    def __init__(self, model, background_blocks, steps=50, warmup_steps=6, interval=6):
        if model.training or steps != 50 or warmup_steps != 6 or interval != 6:
            raise ValueError('Published official-code mapping requires eval, 50 steps and fixed 6/6 cadence')
        self.model = model
        self.steps = steps
        self.i = 0
        self.background = list(background_blocks)
        count = len(model.transformer_blocks)
        if count != 28 or self.background != sorted(set(self.background)):
            raise ValueError('Expected 28 LTX blocks and a sorted unique cache list')
        if any(x < 0 or x >= count for x in self.background):
            raise ValueError('Cache block outside LTX model')
        # Upstream stores a contiguous group's input after its preceding block.
        # A group starting at block zero has no such boundary in the original code.
        if self.background[:2] == [0, 1]:
            raise ValueError('The unmodified upstream cache does not support a contiguous group starting at zero')
        self.cache = DitCache(step_start=6, step_interval=6, block_start=6, num_blocks=28,
            block_cache_list_background=self.background,
            block_cache_list_foreground=sorted(set(range(count)) - set(self.background)),
            step_cache_list=[6, 18, 26, 32, 38, 42, 46, 48])
        self.original = model.forward
        self.block_originals = []
        self.events = []
        self.block_computed = 0
        for idx, block in enumerate(model.transformer_blocks):
            original = block.forward
            self.block_originals.append(original)
            def counted(hidden_states, *args, _original=original, **kwargs):
                self.block_computed += 1
                return _original(hidden_states, *args, **kwargs)
            def wrapped(hidden_states, *args, _idx=idx, _counted=counted, **kwargs):
                return self.cache.forward_single_original(
                    _counted, self.i, _idx, hidden_states, *args, **kwargs)
            block.forward = wrapped
        model.forward = self.forward

    def forward(self, *args, **kwargs):
        if self.i >= self.steps:
            raise RuntimeError('Cache state cannot be reused across generations')
        self.block_computed = 0
        out = self.original(*args, **kwargs)
        count = len(self.block_originals)
        self.events.append({'step': self.i + 1, 'full_compute': self.block_computed == count,
                            'computed_blocks': self.block_computed,
                            'cached_blocks': count - self.block_computed})
        self.i += 1
        if self.i == self.steps:
            self.cache.delta_cache.clear()
            self.cache.prev_hidden = None
        return out

    def report(self):
        if self.i != self.steps:
            raise RuntimeError('Generation incomplete')
        count = len(self.block_originals)
        total = count * self.steps
        computed = sum(e['computed_blocks'] for e in self.events)
        return {'algorithm': 'ProfilingDiT official-code LTX port (mapped defaults)',
            'upstream_commit': UPSTREAM_COMMIT, 'upstream_cache_unmodified': True,
            'upstream_entry': 'DitCache.forward_single_original',
            'official_ltx_implementation': False, 'background_blocks': self.background,
            'refresh_steps_zero_based': [i for i in range(self.steps) if i < 6 or (i - 6) % 6 == 0],
            'full_transformer_steps': sum(e['full_compute'] for e in self.events),
            'cached_transformer_steps': 0, 'equivalent_full_steps': computed / count,
            'computed_block_calls': computed, 'cached_block_calls': total - computed,
            'total_block_calls': total, 'transformer_compute_ratio': computed / total,
            'cache_trace': self.events, 'full_warmup_steps': 6, 'final_step_full': False,
            'fixed_refresh_interval': 6, 'restoration_network_used': False,
            'custom_hardware_fusions': False,
            'residual_boundary': 'unmodified upstream single and contiguous block-group deltas'}

    def close(self):
        self.model.forward = self.original
        for block, original in zip(self.model.transformer_blocks, self.block_originals):
            block.forward = original
        self.cache.delta_cache.clear()
        self.cache.prev_hidden = None
