# M3 video and FaceCache baseline

Date: 2026-09-26
Environment: CPU, Python 3.11, DeepFace 0.0.101, RetinaFace 0.0.18,
Facenet512, batch size 8

## Outcome

M3 writes every frame in a requested video window, draws every cached RetinaFace detection
as `Unknown`, and persists detections and embeddings by absolute frame index. Failed frames
are isolated and retried on the next run. Downstream choices such as stride, threshold, pin
strategy, and output path do not invalidate the cache.

| Run | Selected | Hits | Newly computed | Faces | Failures | Wall | New frames/s | Wall/new frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold stride 3, frames 0–299 | 100 | 0 | 100 | 418 | 0 | 279.678 s | 0.358 | 2.797 s |
| Stride 1 after stride 3 | 300 | 100 | 200 | 1,262 | 0 | 536.171 s | 0.373 | 2.681 s |
| Warm stride 1 replay | 300 | 300 | 0 | 1,262 | 0 | 2.724 s | n/a | n/a |

The first two runs populated every one of the 300 frame entries exactly once. Their
815.849 s combined wall time is a conservative cold stride-1 equivalent because it also
includes two model/process startups and two complete video encodes. A separate cold
stride-1 run was not performed after the cache was complete: it would repeat all 300
RetinaFace calculations without adding evidence.

Across those two cold runs, the cache was populated at 0.368 unique frames/s, or 2.719 s
per unique frame. The warm replay wrote 110.13 frames/s (9.08 ms/frame). These are the
primary baseline numbers for later runtime work; compare the same 300-frame window on the
same CPU and retain output/count checks when testing an optimization.

A separate one-frame `--no-cache` probe measured 6.014 s of lazy model loading inside
10.888 s of perception (10.962 s wall time). Model loading is included in perception time,
not additive to it.

## Reproduction commands

```bash
# Initial preview population
.venv/bin/python label_video.py --input data/video-source/nimbus.mp4 \
  --output output/m3_stride3_300.mp4 --ref-dir data/reference-images \
  --stride 3 --batch-size 8 --max-frames 300 --cache-dir cache/m3

# Fill only the frames missing from that preview
.venv/bin/python label_video.py --input data/video-source/nimbus.mp4 \
  --output output/m3_stride1_after_stride3_300.mp4 --ref-dir data/reference-images \
  --stride 1 --batch-size 8 --max-frames 300 --cache-dir cache/m3

# Measure a fully cached replay
.venv/bin/python label_video.py --input data/video-source/nimbus.mp4 \
  --output output/m3_stride1_warm_300.mp4 --ref-dir data/reference-images \
  --stride 1 --batch-size 8 --max-frames 300 --cache-dir cache/m3
```

## Output checks

- Source: 1920×1080, 30000/1001 fps.
- Each M3 artifact: 1920×1080, 2997/100 fps, 300 frames, 10.010 s.
- Visual inspection at frame 150 showed two correctly positioned grey `Unknown` boxes on
  the two foreground faces.
- The fully warm replay reported zero model and perception time, proving the cache path
  does not load Facenet512 or run RetinaFace.

## D2 decision

Keep batch size 8. It completed the measured run without recovery or memory failure, and
RetinaFace detection remains the dominant one-frame-at-a-time cost. Eli selected this
workflow on 2026-09-26:

- Stride 3 for quick engineering smoke tests: approximately one-third of the detection
  work, with boxes up to two frames (about 67 ms) stale.
- Stride 2 for reviewable previews: approximately half of the detection work, with boxes
  up to one frame (about 33 ms) stale. At the measured rate, the first 300 frames are
  estimated at about 6m48s and the full 3,044-frame clip at about 1h09m; these are
  projections, not measured results.
- Stride 1 for the final deliverable, ensuring detection is attempted on every frame.

The CLI default remains stride 1. A preview followed by stride 1 does not reduce total
inference needed for the final output, but every preview result is cached and reused.
