# M4 perception runtime checkpoint

Date: 2026-09-26. CPU only; Python 3.11, DeepFace 0.0.101, RetinaFace 0.0.18,
Facenet512, base normalization, `all` gallery pins, batch size 8. This is the
Step 10–11 checkpoint before the 3,044-frame M4 run. No full run or optimization
has been started.

## Result and R1/R2 status

RetinaFace dominates the measured cold path. The three stride-1 windows contain
90 newly processed frames and 132 detected faces; all 132 were boxed and given a
character label or `Unknown` in the video and CSV. The samples include a
three-face frame, and there is no configured face cap. The sample proves those
frames were attempted at stride 1; it does not establish detection recall for
the full video or guarantee every visible face was found. Each warm replay
served all 30 selected frames from cache and made zero neural calls.

The measured steady perception rate, after removing each separate process's
model construction, is **2.663 s per newly processed frame** (90-frame mean).
At that rate, a single-process stride-1 run over 3,044 frames projects to about
**2 h 17 min** including measured per-frame non-perception overhead and one model
startup. Stride 2, suitable only for the already approved preview workflow,
projects to **1 h 10 min**. These are estimates, not completed runs.

## Exact profile commands

The `/usr/bin/time -p` prefix measured complete process wall time, including
Python import and cProfile exit. All paths below were run from the repository
root. `--csv` was left at its CLI default (`output/matches.csv`), and a snapshot
was copied after each warm repeat for the face-density check.

```bash
/usr/bin/time -p .venv/bin/python -m cProfile -o output/m4_profile_0600.pstats label_video.py \
  --input data/video-source/nimbus.mp4 --output output/m4_profile_0600.mp4 \
  --ref-dir data/reference-images --start-frame 600 --max-frames 30 \
  --stride 1 --batch-size 8 --cache-dir cache/m3
/usr/bin/time -p .venv/bin/python -m cProfile -o output/m4_profile_0600_warm.pstats label_video.py \
  --input data/video-source/nimbus.mp4 --output output/m4_profile_0600_warm.mp4 \
  --ref-dir data/reference-images --start-frame 600 --max-frames 30 \
  --stride 1 --batch-size 8 --cache-dir cache/m3
.venv/bin/python -c 'import pstats; pstats.Stats("output/m4_profile_0600.pstats").strip_dirs().sort_stats("cumulative").print_stats(50)'
cp output/matches.csv output/m4_profile_0600_matches.csv

/usr/bin/time -p .venv/bin/python -m cProfile -o output/m4_profile_1500.pstats label_video.py \
  --input data/video-source/nimbus.mp4 --output output/m4_profile_1500.mp4 \
  --ref-dir data/reference-images --start-frame 1500 --max-frames 30 \
  --stride 1 --batch-size 8 --cache-dir cache/m3
/usr/bin/time -p .venv/bin/python -m cProfile -o output/m4_profile_1500_warm.pstats label_video.py \
  --input data/video-source/nimbus.mp4 --output output/m4_profile_1500_warm.mp4 \
  --ref-dir data/reference-images --start-frame 1500 --max-frames 30 \
  --stride 1 --batch-size 8 --cache-dir cache/m3
.venv/bin/python -c 'import pstats; pstats.Stats("output/m4_profile_1500.pstats").strip_dirs().sort_stats("cumulative").print_stats(50)'
cp output/matches.csv output/m4_profile_1500_matches.csv

/usr/bin/time -p .venv/bin/python -m cProfile -o output/m4_profile_2700.pstats label_video.py \
  --input data/video-source/nimbus.mp4 --output output/m4_profile_2700.mp4 \
  --ref-dir data/reference-images --start-frame 2700 --max-frames 30 \
  --stride 1 --batch-size 8 --cache-dir cache/m3
/usr/bin/time -p .venv/bin/python -m cProfile -o output/m4_profile_2700_warm.pstats label_video.py \
  --input data/video-source/nimbus.mp4 --output output/m4_profile_2700_warm.mp4 \
  --ref-dir data/reference-images --start-frame 2700 --max-frames 30 \
  --stride 1 --batch-size 8 --cache-dir cache/m3
.venv/bin/python -c 'import pstats; pstats.Stats("output/m4_profile_2700.pstats").strip_dirs().sort_stats("cumulative").print_stats(50)'
cp output/matches.csv output/m4_profile_2700_matches.csv
```

## CLI and profiler evidence

`real` is `/usr/bin/time -p`; `elapsed`, gallery, model, and perception are the
CLI's `complete` line. CLI perception contains model loading when it first
occurs in video processing. The first cold run also populated ten gallery
embeddings in `cache/m3`; its gallery time is a one-time cost. All cold runs
reported 30 misses, zero failures, 30 written frames. All warm runs reported
30 hits, zero misses, zero failures, 30 written frames, `model=0.000s`, and
`perception=0.000s`.

| Frames | Run | Process real | CLI elapsed | Gallery | CLI model | CLI perception | Faces | Assigned labels |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 600–629 | Cold | 134.34 s | 132.282 s | 38.510 s | 9.174 s (gallery) | 92.630 s | 30 | Snape 30 |
| 600–629 | Warm | 1.54 s | 0.418 s | 0.018 s | 0 | 0 | 30 | Snape 30 |
| 1500–1529 | Cold | 89.44 s | 87.845 s | 0.014 s | 7.270 s (video) | 86.443 s | 30 | Unknown 30 |
| 1500–1529 | Warm | 1.42 s | 0.463 s | 0.018 s | 0 | 0 | 30 | Unknown 30 |
| 2700–2729 | Cold | 81.21 s | 79.813 s | 0.013 s | 6.550 s (video) | 78.608 s | 72 | Ron 22, Unknown 50 |
| 2700–2729 | Warm | 1.46 s | 0.545 s | 0.017 s | 0 | 0 | 72 | Ron 22, Unknown 50 |

Face density from the saved CSV snapshots: frames 600–629 and 1500–1529 each
have exactly one logged face per frame. Frames 2700–2729 have two faces on 18
frames and three faces on 12 frames (mean 2.4); no sampled frame has zero.
These are detector outputs, not a manually verified count of all visible faces.

The cold `pstats` cumulative stack gives the following measured calls. The
600 window's RetinaFace and Facenet totals include **10 gallery photos plus 30
video frames**; the other two runs have a warm gallery and isolate the video
path. Cumulative functions are nested and must not be added together.

| Profile | RetinaFace `detect_faces` | Facenet `forward` | `build_models` | `FaceCache.flush` | Video read / write / `draw` |
|---|---:|---:|---:|---:|---:|
| 0600 cold | 40 calls, 111.952 s | 14 calls, 6.737 s | 9.174 s | 4 calls, 0.620 s | 0.076 / 0.159 / 0.071 s |
| 1500 cold | 30 calls, 73.351 s | 4 calls, 3.434 s | 7.270 s | 4 calls, 0.832 s | 0.060 / 0.205 / 0.070 s |
| 2700 cold | 30 calls, 65.975 s | 4 calls, 3.720 s | 6.550 s | 4 calls, 0.627 s | 0.051 / 0.207 / 0.051 s |

RetinaFace's separate `build_model` call took 2.786, 2.116, and 2.064 s in
those respective cold processes. This lazy detector startup is in gallery time
for 0600 and video perception for 1500/2700; it is not included in the CLI's
Facenet `model` field. `pstats` shows 30 `cv2.VideoCapture.read`, 30
`cv2.VideoWriter.write`, and 30 `draw` calls in each cold run. For the two
video-only cold profiles, RetinaFace's 139.326 s of cumulative detection time
is 84.4% of 165.051 s CLI perception time; Facenet forward is 7.154 s across
eight batches. Low-level TensorFlow execution sits within these call stacks,
so its aggregate time cannot be assigned wholly to either network.

Each warm `.pstats` file has **zero** `represent`, RetinaFace `detect_faces`,
Facenet `forward`, or `predict_on_batch` calls. All six videos opened as 30
frames, 1920×1080 at 29.97 fps. The `cache/m3` FaceCache now contains 390
`ok` entries: the original first 300 plus all 90 profiled absolute indices.
Generated profiles, videos, CSV snapshots, and cache entries are gitignored.

## Runtime projection and limits

For a gallery-ready single process, remove each window's one-time model
startup before averaging its video perception: 92.630 s for 0600;
86.443−7.270−2.116 = 77.057 s for 1500; and
78.608−6.550−2.064 = 69.994 s for 2700. Their sum is 239.681 s / 90 =
2.66312 s per newly processed frame. The three cold runs' residual CLI cost
after gallery and perception is (1.142 + 1.388 + 1.192) / 90 =
0.04136 s per written frame. Average one-time Facenet plus RetinaFace startup
is (11.960 + 9.386 + 8.614) / 3 = 9.987 s. Thus:

| Scenario | Formula | Projection |
|---|---|---:|
| Gallery-ready, all-new stride 1 | 3,044 × 2.66312 + 3,044 × 0.04136 + 9.987 | 8,242 s ≈ 2 h 17 min |
| Gallery-ready, all-new stride 2 preview | 1,522 × 2.66312 + 3,044 × 0.04136 + 9.987 | 4,189 s ≈ 1 h 10 min |
| Current `cache/m3`, stride 1 | 2,654 missing × 2.66312 + 3,044 × 0.04136 + 9.987 | 7,204 s ≈ 2 h 00 min |
| Current `cache/m3`, stride 2 preview | 1,327 missing × 2.66312 + 3,044 × 0.04136 + 9.987 | 3,670 s ≈ 1 h 01 min |

The current-cache counts follow 300 initial + 90 sampled stride-1 frames;
stride 2 selects 150 + 45 of those even-numbered indices. A fresh gallery
cache would add the observed 38.510 s gallery build in place of the estimated
one-time video model startup, with no repeated startup between windows in a
single process. The observed per-window steady rates span 2.333–3.088 s/frame;
that alone gives roughly 2 h 01 min–2 h 39 min at stride 1 and 1 h 01 min–
1 h 20 min at stride 2, including the same simple overhead/startup terms.

These are extrapolations from only three 30-frame windows under cProfile.
Scenes, face sizes/density, machine load, TensorFlow warm-up, and profiling
overhead can shift the rate. The FaceCache rewrites its complete compressed
archive after each batch; the 0.6–0.8 s flush cost measured at 300–390 cached
frames may rise as it approaches 3,044. Therefore the point estimates may
understate a long run's cache I/O, even though current read/write/render work
is small relative to detection. The earlier unprofiled M3 baseline (2.719 s
per newly cached frame over the first 300) is consistent with this order of
magnitude, but is a different window/workflow.

## Ranked hotspots and owner decision

1. **RetinaFace per-frame detection**: 65.975–73.351 s per 30 video frames in
   the gallery-warm windows. It is the main lever, but skipping it on final
   frames directly changes R1.
2. **One-time model startup**: 8.614–11.960 s combined Facenet/RetinaFace
   construction per cold process. The full run should pay this once and keep
   both heavy models resident. A separate process for each window paid it
   repeatedly; an idle gate must not unload and rebuild models per trigger.
3. **Facenet512 embedding**: 3.434–3.720 s for four video batches of eight or
   fewer frames. It is already batched and much smaller than detection.
4. **FaceCache archive flush, video I/O, rendering**: each 30-frame cold
   window spent 0.627–0.832 s in cache flush and 0.27–0.34 s in the three
   measured video read/write/draw calls. Full-cache rewrite growth deserves
   attention when planning the long run.

**STOP — Eli's choice is required before Step 12.** The smallest path that
preserves R1 is **(a) continue with the correct per-frame stride-1 baseline**,
using the already cached 390 frames. Alternatively, **(b) amend the plan with
a measured R1-preserving optimization** (for example, better Facenet batching
or safe CPU thread tuning), including tests and acceptance criteria before
implementation. Given the measured split, embedding-only work has limited
possible impact. Or **(c) amend the spec and plan for periodic RetinaFace, a
cheap face/activity gate, or tracking between detections**. This changes the
final-frame detection guarantee: at stride 2 a newly appearing face can lack
a fresh detection for one frame (about 33 ms at 29.97 fps), and at period `k`
the worst nominal wait is `k−1` frames (`(k−1)/29.97` s). A cheap gate has an
additional false-negative risk: a missed trigger can postpone detection for
longer or indefinitely until another trigger; tracking cannot create a box
for an unseen entering face. Unknown entrants are affected just as known
characters are. Option (c) therefore needs explicit missed-face latency and
false-negative acceptance criteria and Eli's approval. No option (b) or (c)
implementation, tracker, gate, sentinel, or full-video run has begun.
