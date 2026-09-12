import argparse
import collections
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import jobs,plan,prompt_groups
from runtime import configuration

class PlanningTests(unittest.TestCase):
    def args(self,shard=0,shards=1):return argparse.Namespace(methods=['native','teacache_fast'],seed=42,samples=1,flicker_samples=1,shard=shard,shards=shards,limit=None,config=None)
    def test_full_official_coverage_without_duplicate_work(self):
        groups=prompt_groups()
        self.assertEqual(len(groups),944)
        self.assertEqual(sorted(i for g in groups for i in g['record_indices']),list(range(946)))
    def test_shards_are_disjoint_and_cover_identical_jobs(self):
        _,full=plan(self.args())
        shards=[plan(self.args(i,3))[1] for i in range(3)]
        key=lambda x:(x['id'],x['index'],x['method'],x['seed'])
        expected={key(j) for j in full};parts=[{key(j) for j in s} for s in shards]
        self.assertEqual(set.union(*parts),expected)
        self.assertEqual(sum(map(len,parts)),len(expected))
    def test_seed_invariant_to_methods_and_order(self):
        a=list(jobs(['native','ours_beta_4p0'],42,1,1));b=list(jobs(['ours_beta_4p0'],42,1,1))
        self.assertEqual([(x['id'],x['seed']) for x in a if x['method']=='ours_beta_4p0'],[(x['id'],x['seed']) for x in b])
    def test_official_flicker_sampling(self):
        counts=collections.Counter(j['id'] for j in jobs(['native']))
        for g in prompt_groups():self.assertEqual(counts[g['id']],25 if 'temporal_flickering' in g['dimensions'] else 5)
        self.assertEqual(sum(counts.values()),6220)
    def test_generation_protocol_is_shared(self):
        g=configuration();self.assertEqual((g['width'],g['height'],g['frames'],g['steps'],g['flow_scale'],g['dynamic_flow']),(1216,704,121,50,.1,True))

if __name__=='__main__':unittest.main()
