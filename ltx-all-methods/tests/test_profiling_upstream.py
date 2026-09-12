"""Check the LTX bridge against direct calls to the real upstream cache."""
import unittest
import torch
from accelerators.profiling_adapter import ProfilingDiTLTX
from accelerators.vendor.profilingdit.cache import DitCache

BG = [4, 7, *range(10, 28)]


class Block(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x, offset=0):
        return x + torch.sin(x * (0.01 * (self.index + 1)) + offset) * 0.02


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([Block(i) for i in range(28)])
    def forward(self, x, offset=0):
        for b in self.transformer_blocks:
            x = b(x, offset=offset)
        return x


class UpstreamBridgeTests(unittest.TestCase):
    def test_all_steps_match_direct_upstream_and_compute_counts(self):
        for dtype in [torch.float32, torch.bfloat16]:
            model, reference = Model().eval(), Model().eval()
            original, block_originals = model.forward, [b.forward for b in model.transformer_blocks]
            cache = ProfilingDiTLTX(model, BG)
            direct = DitCache(6, 6, 6, 28, block_cache_list_background=BG,
                              block_cache_list_foreground=sorted(set(range(28)) - set(BG)),
                              step_cache_list=[6, 18, 26, 32, 38, 42, 46, 48])
            with torch.no_grad():
                for step in range(50):
                    x = (torch.arange(84).reshape(3, 7, 4) / 100 + step / 90).to(dtype)
                    actual = model(x.clone(), offset=step / 40)
                    expected = x.clone()
                    for idx, block in enumerate(reference.transformer_blocks):
                        expected = direct.forward_single_original(block, step, idx, expected, offset=step / 40)
                    self.assertTrue(torch.equal(actual, expected), (dtype, step))
            report = cache.report()
            self.assertEqual(report['computed_block_calls'], 680)
            self.assertEqual(report['refresh_steps_zero_based'], [0, 1, 2, 3, 4, 5, 6, 12, 18, 24, 30, 36, 42, 48])
            self.assertFalse(report['cache_trace'][-1]['full_compute'])
            self.assertEqual(cache.cache.delta_cache, {})
            with self.assertRaises(RuntimeError):
                model(torch.zeros(3, 7, 4))
            cache.close()
            self.assertEqual(model.forward, original)
            self.assertEqual([b.forward for b in model.transformer_blocks], block_originals)
            self.assertTrue(torch.equal(model(x), reference(x)))

    def test_no_background_matches_native_all_steps(self):
        model, reference = Model().eval(), Model().eval()
        cache = ProfilingDiTLTX(model, [])
        with torch.no_grad():
            for step in range(50):
                x = torch.full((3, 4, 8), step / 50)
                self.assertTrue(torch.equal(model(x.clone()), reference(x.clone())))
        self.assertEqual(cache.report()['computed_block_calls'], 1400)
        cache.close()

    def test_partial_run_can_restore_hooks(self):
        model = Model().eval()
        originals = [b.forward for b in model.transformer_blocks]
        cache = ProfilingDiTLTX(model, BG)
        try:
            model(torch.ones(3, 2, 4))
            with self.assertRaises(RuntimeError):
                cache.report()
        finally:
            cache.close()
        self.assertEqual([b.forward for b in model.transformer_blocks], originals)
        self.assertEqual(cache.cache.delta_cache, {})

    def test_unsupported_upstream_zero_group_rejected(self):
        with self.assertRaises(ValueError):
            ProfilingDiTLTX(Model().eval(), [0, 1])


if __name__ == '__main__':
    unittest.main()
