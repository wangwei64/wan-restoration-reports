import argparse,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import completed_run,save,sha,TimingGuard,GPUContended,configuration,fingerprint,read,ROOT,selected_gpu_uuid
import benchmark

class ResumeAndExportTests(unittest.TestCase):
    def complete(self,p,key='request',run=None):
        p.mkdir(parents=True,exist_ok=True);(p/'video.mp4').write_bytes(b'verified-video')
        save(p/'run.json',run or {'online_seconds':1.})
        if run and 'request_sha256' in run:key=run['request_sha256']
        save(p/'completed.json',{'request_sha256':key,'video_sha256':sha(p/'video.mp4'),'run_sha256':sha(p/'run.json')})
    def run_record(self,job):
        request={k:job[k] for k in ['method','prompt','seed']}
        request.update(generation=configuration(),preset=read(ROOT/'configs/methods.json')[job['method']],identity={})
        return {**{k:job[k] for k in ['method','prompt','seed']},'request':request,'request_sha256':fingerprint(request),'online_seconds':1.,'exclusive_gpu_timing_verified':True}
    def test_resume_refuses_changed_request_or_mutated_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.complete(p)
            self.assertEqual(completed_run(p,'request')['online_seconds'],1.)
            with self.assertRaises(RuntimeError):completed_run(p,'different-seed')
            (p/'video.mp4').write_bytes(b'changed-video!')
            with self.assertRaises(RuntimeError):completed_run(p,'request')
    def test_partial_output_is_never_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'video.mp4').write_bytes(b'partial')
            self.assertIsNone(completed_run(p,'request'))
    def test_other_gpu_does_not_invalidate_timing(self):
        with patch('runtime.subprocess.check_output',return_value=f'GPU-A, {os.getpid()}\nGPU-B, 999999\n'):
            g=TimingGuard('GPU-A');g.sample();g.verify()
    def test_overlap_on_same_gpu_invalidates_even_if_later_clear(self):
        with patch('runtime.subprocess.check_output',side_effect=['GPU-A, 999999\n','']):
            g=TimingGuard('GPU-A');g.sample();g.sample()
            with self.assertRaises(GPUContended):g.verify()
    def test_monitor_failure_prevents_unverified_timing(self):
        with patch('runtime.subprocess.check_output',side_effect=OSError('nvidia-smi unavailable')):
            g=TimingGuard('GPU-A');g.sample()
            with self.assertRaises(RuntimeError):g.verify()
    def test_torch_uuid_matches_selected_gpu_in_nvml(self):
        bare='12601ae6-4305-93c8-504b-b51023632b5b';expected='GPU-'+bare
        with patch('runtime.subprocess.check_output',return_value='GPU-00000000-0000-0000-0000-000000000000\n'+expected+'\n'):
            for value in [bare,expected]:self.assertEqual(selected_gpu_uuid(value),expected)
    def test_unknown_or_invalid_gpu_uuid_is_rejected(self):
        with patch('runtime.subprocess.check_output',return_value='GPU-00000000-0000-0000-0000-000000000000\n'):
            for value in ['unavailable','12601ae6-4305-93c8-504b-b51023632b5b']:
                with self.subTest(value=value),self.assertRaises(RuntimeError):selected_gpu_uuid(value)
    def test_export_requires_all_planned_videos(self):
        group={'id':'vbench_0000','prompt':'A test bird','dimensions':['object_class'],'record_indices':[0]}
        with tempfile.TemporaryDirectory() as d,patch('benchmark.prompt_groups',return_value=[group]):
            root=Path(d);a=argparse.Namespace(methods=['native'],seed=42,samples=2,flicker_samples=2,shard=0,shards=1,limit=None,config=None,output=root/'runs',export_dir=root/'export')
            _,todo=benchmark.plan(a)
            for j in todo[:1]:
                self.complete(benchmark.job_directory(a.output,j),run=self.run_record(j))
            with self.assertRaisesRegex(RuntimeError,'1 videos missing'):benchmark.export(a)
            self.assertFalse(json.loads((a.export_dir/'export_receipt.json').read_text())['complete'])
            j=todo[1];self.complete(benchmark.job_directory(a.output,j),run=self.run_record(j))
            benchmark.export(a)
            self.assertTrue(json.loads((a.export_dir/'export_receipt.json').read_text())['complete'])
    def test_export_and_summary_refuse_stale_preset(self):
        group={'id':'vbench_0000','prompt':'A test bird','dimensions':['object_class'],'record_indices':[0]}
        with tempfile.TemporaryDirectory() as d,patch('benchmark.prompt_groups',return_value=[group]):
            root=Path(d);a=argparse.Namespace(methods=['teacache_fast'],seed=42,samples=1,flicker_samples=1,shard=0,shards=1,limit=None,config=None,output=root/'runs',export_dir=root/'export')
            j=benchmark.plan(a)[1][0];run=self.run_record(j)
            run['request']['preset']['threshold']=0.123
            run['request_sha256']=fingerprint(run['request'])
            self.complete(benchmark.job_directory(a.output,j),run=run)
            for operation in [benchmark.export,benchmark.summarize]:
                with self.subTest(operation=operation.__name__),self.assertRaisesRegex(RuntimeError,'preset mismatch'):operation(a)
    def test_summary_rejects_missing_completed_video(self):
        group={'id':'vbench_0000','prompt':'A test bird','dimensions':['object_class'],'record_indices':[0]}
        with tempfile.TemporaryDirectory() as d,patch('benchmark.prompt_groups',return_value=[group]):
            a=argparse.Namespace(methods=['native'],seed=42,samples=1,flicker_samples=1,shard=0,shards=1,limit=None,config=None,output=Path(d)/'runs')
            j=benchmark.plan(a)[1][0];p=benchmark.job_directory(a.output,j)
            self.complete(p,run=self.run_record(j));(p/'video.mp4').unlink()
            with self.assertRaises(FileNotFoundError):benchmark.summarize(a)

if __name__=='__main__':unittest.main()
