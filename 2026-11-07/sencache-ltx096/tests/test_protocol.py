import argparse,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
import benchmark

class ProtocolTests(unittest.TestCase):
    def test_complete_default_plan_and_seed_separation(self):
        rows=list(benchmark.jobs(['sencache_ltx096']))
        self.assertEqual(len(rows),6220)
        self.assertEqual(len(benchmark.prompt_groups()),944)
        seeds={r['seed'] for r in rows}
        self.assertTrue(seeds.isdisjoint({20260828,20260829,20260830}))
        old=list(benchmark.jobs(['native']))
        self.assertEqual([(r['id'],r['index'],r['seed']) for r in old],[(r['id'],r['index'],r['seed']) for r in rows])

    def test_profiles_and_changed_guidance(self):
        cfg=runtime.configuration();methods=runtime.read(runtime.ROOT/'configs/methods.json')
        for m in ['sencache_ltx096','sencache_original091']:
            self.assertTrue(runtime.validate_profile(cfg,methods[m])['matches_validated_generation'])
        changed=dict(cfg,guidance_scale=4.0)
        self.assertFalse(runtime.validate_profile(changed,methods['sencache_ltx096'])['matches_validated_generation'])
        with self.assertRaises(ValueError):runtime.validate_profile(dict(cfg,frames=81),methods['sencache_ltx096'])
        with patch.object(runtime,'sha',return_value='wrong'):
            with self.assertRaises(RuntimeError):runtime.validate_profile(cfg,methods['sencache_ltx096'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'invalid.json';p.write_text(json.dumps(dict(cfg,guidance_scale=1.0)))
            with self.assertRaises(ValueError):runtime.configuration(p)

    def test_resume_rejects_wrong_request_or_corrupt_video(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'video.mp4').write_bytes(b'fixture')
            runtime.save(out/'run.json',{'ok':True})
            runtime.save(out/'completed.json',{'request_sha256':'k','video_sha256':runtime.sha(out/'video.mp4'),'run_sha256':runtime.sha(out/'run.json')})
            self.assertEqual(runtime.completed_run(out,'k'),{'ok':True})
            with self.assertRaises(RuntimeError):runtime.completed_run(out,'different')
            (out/'video.mp4').write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):runtime.completed_run(out,'k')

if __name__=='__main__':unittest.main()
