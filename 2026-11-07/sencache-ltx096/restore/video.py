"""Deterministic x264 output shared by every generation method."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from fractions import Fraction
import av
import numpy as np

def encode_rgb(frames,path,fps=30):
    if frames.dtype!=np.uint8 or frames.ndim!=4 or frames.shape[-1]!=3 or not len(frames):
        raise ValueError('Expected nonempty T,H,W,3 uint8 RGB frames')
    frames=np.ascontiguousarray(frames)
    payload=frames.tobytes()
    command=[sys.executable,str(Path(__file__).resolve()),str(frames.shape[0]),str(frames.shape[1]),str(frames.shape[2]),str(fps),str(Path(path).resolve())]
    options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    result=subprocess.run(command,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300,**options)
    if result.returncode:raise RuntimeError('Video encoder failed: '+result.stderr.decode(errors='replace')[-2000:])
    info=json.loads(result.stdout)
    if info['decoded_rgb_sha256']!=hashlib.sha256(payload).hexdigest():raise RuntimeError('Encoder RGB transfer mismatch')
    return info

def _encode_worker(frames,path,fps):
    from av.video.reformatter import VideoReformatter
    reformatter=VideoReformatter()
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=fps)
        stream.width=frames.shape[2];stream.height=frames.shape[1];stream.pix_fmt='yuv420p'
        # Use a fresh CPU process for every video: fixing only the thread count
        # did not remove in-process x264 variability in the generation process.
        stream.codec_context.thread_count=1
        stream.options={'crf':'18','preset':'fast','x264-params':'threads=1:lookahead-threads=1:sliced-threads=0'}
        for index,frame in enumerate(frames):
            rgb=av.VideoFrame.from_ndarray(frame,format='rgb24')
            yuv=reformatter.reformat(rgb,format='yuv420p',src_colorspace='ITU601',dst_colorspace='ITU601',src_color_range='JPEG',dst_color_range='MPEG',interpolation='BILINEAR')
            yuv.pts=index;yuv.time_base=Fraction(1,fps)
            for packet in stream.encode(yuv):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    return {'decoded_rgb_sha256':hashlib.sha256(frames).hexdigest(),'video_encoder':{'codec':'libx264','crf':18,'preset':'fast','threads':1,'lookahead_threads':1,'sliced_threads':False,'isolated_process':True,'explicit_frame_timestamps':True,'color_conversion':'BT.601 RGB full to YUV limited, bilinear'}}

if __name__=='__main__':
    count,height,width,fps=map(int,sys.argv[1:5]);path=sys.argv[5]
    if min(count,height,width,fps)<=0 or height%2 or width%2:raise ValueError('Invalid video dimensions or frame rate')
    data=sys.stdin.buffer.read()
    if len(data)!=count*height*width*3:raise ValueError('Incomplete RGB input')
    frames=np.frombuffer(data,dtype=np.uint8).reshape(count,height,width,3)
    print(json.dumps(_encode_worker(frames,path,fps)))
