# M4 RetinaFace Black-Margin Optimization Gate

> **Historical snapshot as of `c7736ca`.** Eli accepted this gate and the optimized full
> run completed. See [`docs/STATUS.md`](../STATUS.md) for current state.

Status: accepted and implemented

Date: 2026-09-26

Post-gate outcome: the full run completed in 33m03s with all 3,044 frames,
5,353 detections, and zero failures. See [M4 Full Stride-1 Run](m4-full-run.md).

## Environment and versions

- Hardware/OS: Apple MacBook Air M4, macOS arm64, CPU-only
- Python: 3.11.15
- DeepFace: 0.0.101
- retina-face: 0.0.18
- TensorFlow / tf-keras: 2.21.0
- OpenCV: 5.0.0.93
- NumPy: 2.4.6
- Input SHA-256: `67c428ef51ccd86ccbe55a08a6e00efe5540eb6387bb03ff0fd4aade8fbba353`
- Candidate pipeline: `retinaface_exact_resize_crop_v1`, halo 32, FaceCache schema 2

## What improved

The old DeepFace call padded every 1920x1080 frame to 3840x2160 before RetinaFace resized
it to 1820x1024. Only one quarter of that detector tensor contained the source image. The
new path reproduces those resized image pixels exactly, crops the grid-aligned black-only
margin while retaining a 32 px halo, restores detections to full-frame coordinates, aligns
each face locally, and batch-embeds the crops with Facenet512.

| Measure | Baseline path | Optimized production path | Change |
| --- | ---: | ---: | ---: |
| RetinaFace input | 1820x1024 | 988x576 | 69.5% fewer pixels; 3.27x smaller |
| Black area in detector input | 75% | halo only | Dense convolution no longer covers the large border |
| 300-frame cold perception | 815.849 s | 265.390 s | 67.5% lower; 3.1x throughput |
| Per selected frame | 2.719 s | 0.884633 s | 1.834 s saved |
| Detected faces | 1,262 | 1,265 | 0 baseline losses; 3 real additions |
| Historical full-clip estimate | 2 h 17 m | about 46 m projected | about 1 h 31 m saved |
| Fully warm 300-frame replay | 2.724 s | 2.707 s | No inference in either path |

The optimization changes neither the public CLI nor the gallery cache. FaceCache identity
does change from schema 1 to schema 2 because boxes and embeddings come from a new
perception pipeline; old caches remain on disk and produce a clean miss instead of being
silently reused.

## Commands

The preserved baseline was read from
`/tmp/whiteswan-investigation.HOLoyc/baseline300/`. Candidate artifacts were isolated under
`/tmp/whiteswan-m4-gate.MyxQLx/`. The production cold command was:

```bash
/usr/bin/time -lp .venv/bin/python label_video.py \
  --input data/video-source/nimbus.mp4 \
  --output /tmp/whiteswan-m4-gate.MyxQLx/candidate.mp4 \
  --ref-dir data/reference-images \
  --cache-dir /tmp/whiteswan-m4-gate.MyxQLx/cache \
  --csv /tmp/whiteswan-m4-gate.MyxQLx/candidate.csv \
  --start-frame 0 --max-frames 300 --stride 1 --batch-size 8 --threshold 0.30
```

The changed-threshold warm replay used the same command with separate output/CSV paths and
`--threshold 0.25`. A second replay at 0.20 bracketed a SHA-256 check of the FaceCache.
The independent comparison script and generated review sheets remain outside the repo in
the temporary directory.

## Automated verification

| Check | Result |
| --- | --- |
| Fast suite | 165 passed, 3 deselected |
| Slow suite | 3 passed, 165 deselected |
| Dependency check | No broken requirements |
| CLI compatibility | No pipeline or halo flags added |

## Baseline integrity

The preserved schema-1 baseline contains exactly frame indices 0-299, all 300 statuses are
`ok`, and it contains 1,262 faces. Its video hash matches the source:
`67c428ef51ccd86ccbe55a08a6e00efe5540eb6387bb03ff0fd4aade8fbba353`.
The production candidate preflight reported `cached=0 to_infer=300`; its output cache is
schema 2 with the approved pipeline and halo and contains 300 `ok` frames with no failures.

## Detection comparison

| Metric | Result |
| --- | --- |
| Baseline faces | 1,262 |
| Candidate faces | 1,265 |
| Frames with equal counts | 297 / 300 |
| IoU >= 0.5 matches | 1,262 |
| Baseline losses | 0 |
| Candidate additions | 3 |
| IoU mean / median | 0.955647 / 0.960319 |
| IoU p05 / p95 / minimum | 0.905075 / 0.988319 / 0.853961 |

## Embedding comparison

All 1,262 matched pairs were compared using the independently loaded unit embeddings.

| Metric | Mean | Median | p05 | p95 | p99 | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cosine similarity | 0.977606 | 0.983535 | 0.936107 | 0.998188 | n/a | 0.727504 | 0.999994 |
| Cosine distance | 0.022394 | 0.016465 | 0.001812 | 0.063893 | 0.111699 | 0.000006 | 0.272496 |

The different alignment crops therefore move some embeddings, but the distribution remains
close overall. The minimum-similarity case is retained in the comparison JSON for future
inspection rather than hidden by an aggregate.

## R2 identity changes at threshold 0.30

There are 129 nearest-gallery-owner changes among matched faces, overwhelmingly above the
threshold and therefore still `Unknown`. Exactly five thresholded assignments change:

| Frame | Face | Baseline nearest / distance / assignment | Candidate nearest / distance / assignment |
| ---: | ---: | --- | --- |
| 183 | 0 | Harry Potter / 0.291481 / Harry Potter | Harry Potter / 0.302667 / Unknown |
| 190 | 7 | Prof. Severus Snape / 0.317055 / Unknown | Prof. Severus Snape / 0.286705 / Prof. Severus Snape |
| 213 | 4 | Prof. Severus Snape / 0.298577 / Prof. Severus Snape | Prof. Severus Snape / 0.302479 / Unknown |
| 219 | 4 | Prof. Severus Snape / 0.318801 / Unknown | Prof. Severus Snape / 0.296708 / Prof. Severus Snape |
| 249 | 3 | Prof. Severus Snape / 0.302215 / Unknown | Prof. Severus Snape / 0.286145 / Prof. Severus Snape |

All five are the correct visible character and lie within 0.0188 of the provisional 0.30
threshold. This is expected threshold-edge movement from slightly different aligned crops;
the threshold remains an M5 decision and is not changed here.

## Manual review

There are no losses to review. The three additions were inspected in
`/tmp/whiteswan-m4-gate.MyxQLx/additions-contact-sheet.png`:

- frame 221, box `(1822, 656, 34, 42)`, confidence 0.91: real small background face;
- frame 222, same box/confidence in the adjacent frame: the same real background face; and
- frame 245, box `(1799, 664, 59, 86)`, confidence 0.90: real background face.

The five assignment changes were also inspected in
`assignment-changes-contact-sheet.png`; they show Harry and Snape, matching the nearest
owners listed above.

## Timings

| Production cold measurement | Result |
| --- | ---: |
| Combined video model load | 6.661 s |
| Perception | 265.390 s |
| Per selected frame | 0.884633 s |
| Non-perception process-video work (cache + read/write + match/render/CSV) | 5.105 s |
| CLI video elapsed | 270.495 s |
| Gallery cache load | 0.058 s |
| Process wall / user / system | 271.64 / 1535.10 / 30.52 s |

An earlier instrumented execution of the same approved detector/align/embed algorithm
measured RetinaFace build 1.283 s, Facenet512 build 1.318 s, RetinaFace forward 252.923 s,
preprocessing plus local alignment 1.699 s, and Facenet512 forward 24.186 s over 300 frames.
Those nested timers are diagnostic and are not added to the production wall measurement.

The M3 baseline populated the same 300 frames in 815.849 s across two runs, or 2.719 s per
unique frame. Comparing per-frame perception, the production candidate is 67.5% faster,
comfortably exceeding the 50% target. Straight-line projection from the unprofiled
production wall time is about 46 minutes for 3,044 frames; allowing scene/cache-growth
variation, 45-55 minutes is the defensible range until the full run is authorized.

The cold candidate and both warm outputs contain 300 frames at 1920x1080 and 29.97 fps
(10.010 s). Every CSV has the fixed 12-column schema and 1,265 rows.

## Warm schema-2 replay

At threshold 0.25, preflight reported `cached=300 to_infer=0`. The replay published the
300-frame video and 1,265-row CSV in 2.707 s CLI time / 3.37 s process wall, with
`model=0.000s`, `perception=0.000s`, 300 hits, and zero misses/failures. A second threshold
0.20 replay repeated those zero-inference facts. The FaceCache SHA-256 was
`4fb32d5c9057a30d09e7db51b017057625361306a9ca67e72a1eac1e60ed3a5d` before and after
the second replay, proving cached boxes/embeddings were not rewritten.

## Limitations

- The 300-frame window is 9.9% of the clip and is regression evidence, not manually
  annotated ground truth for every visible face.
- The projected 45-55 minute full cold runtime remains a projection until owner approval
  permits the complete 3,044-frame run.

## R1 / R2 conclusion and STOP decision

The candidate passes the technical gate. For R1, it retains all 1,262 sampled baseline
faces and adds three manually confirmed real faces while preserving stride 1 and no face
cap. For R2, it preserves the recognition contract and exposes all five threshold-edge
assignment changes; every visible changed identity matches the correct nearest owner.
Runtime improves by 67.5% on the comparable perception rate, and warm replay remains
inference-free.

**STOP:** this evidence recommends accepting the switch, but the full 3,044-frame run must
not begin until Eli reviews and approves this gate.
