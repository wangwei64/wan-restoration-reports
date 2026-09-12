import copy
import os
import unittest
from pathlib import Path
from speed_sweep import read, select_by_speed, validate_protocol, verify_package


class SpeedSelectionTests(unittest.TestCase):
    def test_quality_cannot_change_selection(self):
        rows = [{'mode': 'a', 'speedup': 1.9, 'ssim': .2, 'lpips': .9},
                {'mode': 'b', 'speedup': 2.1, 'ssim': .1, 'lpips': 1.1},
                {'mode': 'c', 'speedup': 1.01, 'ssim': .999, 'lpips': .001}]
        expected = select_by_speed(rows, [2.1], .05)
        for row in rows:
            row['ssim'], row['lpips'] = 1 - row['ssim'], 2 - row['lpips']
        self.assertEqual(expected, select_by_speed(rows, [2.1], .05))
        self.assertEqual([x['candidate'] for x in expected['tiers'].values()], ['b'])
        self.assertEqual(list(expected['tiers']), ['profiling_speed_fast'])
        self.assertTrue(expected['all_targets_met'])

    def test_missed_targets_are_explicit(self):
        result = select_by_speed([{'mode': 'a', 'speedup': 1.2}, {'mode': 'b', 'speedup': 1.4}], [2.1], .05)
        self.assertFalse(result['all_targets_met'])
        self.assertFalse(result['selection_uses_quality'])

    def test_selection_is_order_independent(self):
        rows = [{'mode': 'b', 'speedup': 2.11}, {'mode': 'a', 'speedup': 1.88}, {'mode': 'c', 'speedup': 2.11}]
        self.assertEqual(select_by_speed(rows, [2.1], .05), select_by_speed(rows[::-1], [2.1], .05))

    def test_rejects_invalid_timing_and_duplicate_candidates(self):
        for rows in [[{'mode': 'a', 'speedup': float('nan')}],
                     [{'mode': 'a', 'speedup': 1.9}, {'mode': 'a', 'speedup': 2.1}],
                     []]:
            with self.assertRaises(ValueError):
                select_by_speed(rows, [2.1], .05)

    def test_frozen_protocol_and_package(self):
        here = Path(__file__).resolve().parent
        protocol = read(here / 'protocol.json')
        validate_protocol(protocol)
        package = Path(os.environ.get('LTX_SPEED_BASE_PACKAGE', str(here.parents[1] / 'results/2026-09-11_ltx_profiling_vbench/published_package/ltx_all_methods')))
        verify_package(package, protocol)
        bad = copy.deepcopy(protocol)
        bad['holdout'][0]['prompt'] = bad['validation'][0]['prompt']
        with self.assertRaises(ValueError):
            validate_protocol(bad)
        bad = copy.deepcopy(protocol)
        bad['quality_gate'] = {'ssim': .98}
        with self.assertRaises(ValueError):
            validate_protocol(bad)


if __name__ == '__main__':
    unittest.main()
