"""Identical RGB/H.264 export parameters to the chosen fastest report."""

import numpy as np
import imageio_ffmpeg


def write_video(path, frames, fps=16):
    array = np.asarray(frames, dtype=np.uint8)
    height, width = array.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "17", "-preset", "medium", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for f in array:
            writer.send(np.ascontiguousarray(f).tobytes())
    finally:
        writer.close()
