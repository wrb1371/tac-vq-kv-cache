"""Lossless common input archive with complete-source timing, using Decord."""

import argparse, pathlib, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    import decord
    from transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl import smart_resize

    reader = decord.VideoReader(a.video, num_threads=2)
    total = len(reader)
    fps = float(reader.get_avg_fps())
    h, w, _ = reader[0].shape
    h, w = smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=384**2)
    del reader
    reader = decord.VideoReader(a.video, num_threads=2, width=w, height=h)
    indices = np.linspace(0, total - 1, a.frames).round().astype(np.int64)
    frames = reader.get_batch(indices).asnumpy()
    duration = total / fps
    p = pathlib.Path(a.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        frames=frames,
        source_indices=indices,
        source_fps=np.float64(fps),
        source_total_frames=np.int64(total),
        duration_seconds=np.float64(duration),
        effective_fps=np.float64(a.frames / duration),
    )


if __name__ == "__main__":
    main()
