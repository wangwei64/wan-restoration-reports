import importlib.util,sys,tempfile,unittest,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

@unittest.skipUnless(importlib.util.find_spec('av'),'PyAV is required in the Linux generation environment')
class VideoEncodingTests(unittest.TestCase):
    def test_identical_rgb_is_repeatable_under_cpu_load(self):
        import numpy as np
        from restore.video import encode_rgb
        frames=np.random.default_rng(946).integers(0,256,(33,128,192,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=encode_rgb(frames,root/'a.mp4')
            def load():
                x=np.arange(100000,dtype=np.float64)
                for _ in range(30):x=np.sqrt(x+1)
                return x.sum()
            with ThreadPoolExecutor(max_workers=4) as pool:
                tasks=[pool.submit(load) for _ in range(4)]
                b=encode_rgb(frames,root/'b.mp4')
                for task in tasks:task.result()
            c=encode_rgb(frames,root/'c.mp4')
            self.assertEqual(a,b);self.assertEqual(b,c)
            self.assertEqual(a['video_encoder']['threads'],1)
            self.assertTrue(a['video_encoder']['isolated_process'])
            self.assertTrue(a['video_encoder']['explicit_frame_timestamps'])
            self.assertEqual(len({hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.mp4')}),1)
    def test_rejects_wrong_dtype_before_encoding(self):
        import numpy as np
        from restore.video import encode_rgb
        with self.assertRaises(ValueError):encode_rgb(np.zeros((1,32,32,3),dtype=np.float32),'unused.mp4')
