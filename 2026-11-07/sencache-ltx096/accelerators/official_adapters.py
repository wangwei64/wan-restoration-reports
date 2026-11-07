"""Restore upstream cache schedules while retaining the frozen native LTX bridge."""
from .teacache_adapter import TeaCacheLTX
from .cache_adapters import SenCacheLTX


class TeaCacheOfficialLTX(TeaCacheLTX):
    def __init__(self, model, threshold, steps=50):
        super().__init__(model, threshold, steps, warmup_steps=1)

    def report(self):
        value = super().report()
        value.update(algorithm='TeaCache LTX upstream schedule, native 0.9.6 bridge',
                     schedule_change='Only first and last steps are forced full; no added 10-step warmup',
                     upstream_commit='1857e572169b01f55225cf019541d34de3c3c702')
        return value


class SenCacheOfficialLTX(SenCacheLTX):
    def __init__(self, model, sensitivity_path, steps=50, threshold_start=.01,
                 threshold_main=.7, max_skips=4):
        retention = int(round(steps * .02))
        if retention != 1:
            raise ValueError('This validated bridge is scoped to the 50-step experiment')
        super().__init__(model, threshold_main, sensitivity_path, steps,
                         warmup_steps=retention, max_skips=max_skips)
        self.threshold_start = float(threshold_start)
        self.threshold_main = float(threshold_main)
        self.threshold_switch_step = int(round(steps * .2))

    def forward(self, *args, **kwargs):
        self.threshold = (self.threshold_start if self.i < self.threshold_switch_step
                          else self.threshold_main)
        output = super().forward(*args, **kwargs)
        self.events[-1]['threshold_unscaled'] = self.threshold
        self.events[-1]['threshold_phase'] = ('start' if self.i <= self.threshold_switch_step else 'main')
        return output

    def report(self):
        value = super().report()
        value.update(algorithm='SenCache LTX upstream dual-threshold schedule, native 0.9.6 bridge',
                     threshold_start=self.threshold_start, threshold_main=self.threshold_main,
                     threshold_switch_step_zero_based=self.threshold_switch_step,
                     upstream_commit='3fb96a41511ede1e38f3555c863b1cb7cddfe567')
        return value
