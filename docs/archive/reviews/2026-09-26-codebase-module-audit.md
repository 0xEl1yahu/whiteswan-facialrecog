# Video Runtime Investigation and Codebase Module Audit

> **Archived investigation snapshot.** Measurements and “current” language refer to commit
> `8519059` on 2026-09-26. The approved optimization was later implemented; current status
> lives in [`docs/STATUS.md`](../../STATUS.md).

Status: non-authoritative engineering assessment

Date: 2026-09-26

Branch: `investigation/video-runtime`

Commit: `8519059972f4f309d993bf958faeffa099daa68a`

## 1. Decision question and authority

This investigation asks why the 3,044-frame video can take up to roughly two hours to
process, which models and application stages consume that time, and whether the reported
DeepFace alignment-padding hypothesis explains a safe optimization path.

Authority, in descending order:

1. `docs/design/design-plan.md`, including R1 (box every detected face) and R2 (name or
   `Unknown`), CPU-only execution, fixed RetinaFace/Facenet512 models, stride 1 for the
   final output, and cache/determinism requirements.
2. `docs/implementation/implementation-plan.md` and the approved modularization amendment.
3. Stable data contracts, CLI behavior, cache formats, tests, and current implementation.
4. Historical profiles in `docs/results/`.

No application source, contract, CLI, dependency, model, or default was changed. The only
tracked artifact added by this investigation is this report. Any optimization requires a
plan/spec amendment and owner approval before implementation. This report does not certify
conformance with an external standard.

## 2. Snapshot, scope, and evidence limits

### Repository and environment

- Repository: `/Users/elibelilty/Documents/GitHub/whiteswan-facialrecog`
- Working tree started clean on `main`; investigation branch created from the exact `main`
  commit above.
- Hardware: Apple MacBook Air, M4, 10 cores (4 performance, 6 efficiency), 16 GB RAM.
- OS/architecture: macOS 26.5.1, arm64.
- Python: 3.11.15.
- TensorFlow exposes one CPU device and no GPU. TensorFlow intra/inter-op thread settings
  are both `0` (automatic); OpenCV reports 10 threads.
- Models: RetinaFace, 29,495,084 parameters, 113 MB weights; Facenet512,
  23,497,424 parameters, 91 MB weights.
- Input: H.264, 1920x1080, 30000/1001 fps, 3,044 frames, 101.568 seconds,
  SHA-256 `67c428ef51ccd86ccbe55a08a6e00efe5540eb6387bb03ff0fd4aade8fbba353`.
- Installed versions match the pins: DeepFace 0.0.101, retina-face 0.0.18,
  TensorFlow/tf-keras 2.21.0, OpenCV 5.0.0.93, NumPy 2.4.6.

### Scope

The stable module boundary is the executable plus the ten `face_labeller` modules. The
investigation covers video preflight/decoding, DeepFace orchestration, RetinaFace,
alignment, Facenet512, matching, drawing, CSV output, video encoding, gallery work, and
both cache families. Vendored/generated artifacts and private media are excluded except
where their runtime or cache state is direct evidence.

### Evidence limits

- No full 3,044-frame cold run was started because the repository is at the M4 STOP gate.
- The current repository cache contains 202 compatible stride-1 frame results, not a full
  cache. Current preflight is therefore 202 hits and 2,842 frames requiring inference.
- The historical profile contains three 30-frame cold windows. Fresh profiling repeated
  the 1,500-1,529 window. The complete 300-frame M3 baseline was reconstructed in an
  isolated cache and compared with the candidate detector and embeddings (9.9% of the
  video); no repository cache was modified.
- There is no manually annotated full-video face ground truth, hardware performance-counter
  trace, temperature/power trace, or production telemetry. Detector parity is therefore a
  strong regression signal, not proof that either path finds every visible face.
- Runtime projections are explicitly projections. Sustained TensorFlow throughput varied
  materially between runs.

## 3. Executive conclusion

The reported padding hypothesis is correct and explains the dominant avoidable compute.
With `align=True`, DeepFace adds half a frame of black border on every side before invoking
RetinaFace. A 1920x1080 frame becomes 3840x2160, then RetinaFace resizes it to 1820x1024.
Only about 910x512 of that tensor is image content: exactly 25% of the spatial area. Dense
convolutions still execute over the remaining 75% black area.

RetinaFace is the bottleneck, not video I/O, drawing, matching, or Facenet512. Historical
profiles put RetinaFace at 84.4% of steady perception time; a fresh exact-window profile
put it at 36.218 seconds of 44.574 seconds of perception for 30 frames. DeepFace loops over
frames for detection, so `batch_size=8` batches Facenet512 only and cannot remove 3,044
serial detector calls at final stride 1.

The original proposal is directionally right:

- Single-core forward-pass measurements reproduced the reported ratios almost exactly.
- Direct 960x540 detection plus crop-local alignment and batched `detector_backend="skip"`
  embeddings reduced prototype perception to 0.396 seconds/frame in a one-face window and
  0.445 seconds/frame in a 276-face/30-frame crowd window.
- However, plain 960x540 detection missed four of 71 cached boxes in a 12-frame challenge
  set; at least two misses were a real small background face. It is not yet admissible as a
  drop-in R1-preserving change.

The strongest candidate is a stricter version of the same insight: reproduce RetinaFace's
current 1820x1024 resized pixels, then crop black-only margins on 32-pixel feature-grid
boundaries, retaining a 39-pixel horizontal and 32-pixel vertical halo. RetinaFace then
receives 988x576, 30.5% of the present network area, at the exact same face scale and grid
phase. Across the complete 300-frame M3 baseline it matched all 1,262 baseline detections
at IoU >= 0.5 and found three additional, manually confirmed real faces. This is compelling
evidence for an R1-preserving optimization experiment, but not full-video proof.

The full 300-frame candidate prototype measured 0.932 seconds/frame, which extrapolates to
47.3 minutes of perception over 3,044 frames and roughly 50 minutes after normal pipeline
overhead. The historical network-area estimate was 56 minutes. The separate investigation's
roughly 50-minute projection is therefore directly corroborated by a sustained 300-frame
measurement, although it remains a projection rather than a measured full-video CLI run.

### Post-audit implementation outcome

The approved candidate has now been integrated into the production path and rerun over the
same 300-frame baseline. The production measurement improved on the prototype: perception
fell from the comparable M3 rate of 2.719 seconds/frame to 0.884633 seconds/frame, or 67.5%,
while matching all 1,262 baseline faces and adding three manually confirmed real faces.
Mean matched-box IoU was 0.956 and mean embedding cosine similarity was 0.978. A fully warm
replay remained inference-free and completed in 2.707 seconds.

The subsequent owner-approved full run measured 33m03s end to end for all 3,044 frames,
faster than the conservative 45-55 minute range. It produced 5,353 detections with zero
failed frames and a complete schema-2 cache. This measurement supersedes the projections.
See [M4 Full Stride-1 Run](../../results/m4-full-run.md); the preceding acceptance record is
[M4 RetinaFace Black-Margin Optimization Gate](../../results/m4-retinaface-black-margin-optimization.md).

## 4. Reproducibility and coverage

- First-party modules classified: 11/11 (executable plus ten package modules).
- Entrypoints resolved: CLI -> `core.run` -> preflight/cache/gallery -> `process_video`.
- External activation boundaries unresolved: none; this is a local CLI.
- Runtime workloads observed: historical 300-frame cache baseline, historical 3x30-frame
  profiles, fresh 30-frame exact-window profile, 16- and 30-frame split-pipeline prototypes,
  12-frame resolution challenge set, 202-frame detector-parity sweeps, and a complete
  300-frame M3 detection/embedding comparison.
- Fast tests: 148 passed, 3 deselected in 5.24 seconds.
- Slow tests: 2 passed, 1 failed, 148 deselected in 31.56 seconds. The failure is an
  existing stale test dependency described in section 13.
- `.venv/bin/python -m pip check`: no broken requirements.
- CLI help and `git diff --check`: passed.

## 5. System and compute map

```text
3,044 H.264 frames
  -> OpenCV decode
  -> groups of 8 selected frames
  -> DeepFace.represent(frames)
       -> loop once per frame
          -> add 960/540 black border on each side
          -> RetinaFace dense CNN on 1820x1024 tensor
          -> NMS + boxes + five landmarks
          -> face-local eye alignment/crop
       -> Facenet512 forward over all face crops in the group (160x160 each)
  -> FaceCache rewrite after each completed group
  -> 10-pin cosine matching + confidence mapping
  -> CSV + drawing + mp4v encode
```

The key asymmetry is that detection is serial per frame while recognition is batched per
face crop. A frame with no faces still pays the full RetinaFace cost. A crowded frame pays
approximately the same detector cost, then additional but much smaller Facenet512 work.

The cache changes the economics after the first pass: the documented warm replay wrote
300 frames in 2.724 seconds with zero neural calls. Threshold, pin-strategy, and rendering
changes are cheap; the first compatible stride-1 perception pass is expensive.

## 6. Source-level corroboration of the padding finding

### Fact: DeepFace pads before detection

Installed DeepFace `modules/detection.py:308-324` calculates borders as half of the input
height and width, applies `cv2.copyMakeBorder` when `align=True`, and only then calls the
detector. For this video:

| Stage | Width x height | Pixels | Image-content share |
|---|---:|---:|---:|
| Source frame | 1920x1080 | 2,073,600 | 100% |
| DeepFace padded frame | 3840x2160 | 8,294,400 | 25% |
| RetinaFace tensor | 1820x1024 | 1,863,680 | 25% (about 910x512 content) |

Installed RetinaFace uses target short side 1024 and maximum long side 1980
(`retinaface/commons/preprocess.py:99-116,134-136`). DeepFace's RetinaFace adapter calls
`detect_faces(... threshold=0.9)` without overriding `allow_upscaling`, so its default
`True` applies (`deepface/models/face_detection/RetinaFace.py:48`).

### Fact: alignment now happens on a local crop

DeepFace `modules/detection.py:396-419` explicitly says it no longer aligns the original
image; it extracts a face-local sub-image with margin and aligns that. The full-frame black
border remains earlier in the path.

To test whether border-free crop alignment is behaviorally equivalent when given the same
detection, the current padded path was replayed through `extract_face` on the original
full-resolution frame with `width_border=0` and `height_border=0`. Ten faces were checked:
one large face on frame 1500 and nine faces on frame 194, including a detection extending
beyond the right edge. Every aligned crop had the same shape and was byte-for-byte equal.

This supports the proposed split: detect separately, scale metadata to full resolution,
use crop-local `extract_face`, and batch Facenet512 with `detector_backend="skip"`.

### Important nuance

"75% of the input is black" is exact spatially. "75% of wall time is wasted" is an
approximation because fixed overhead, feature-map shapes, kernel selection, NMS, and
threading also contribute. The measured forward-pass ratios nevertheless track pixel area
closely, which is what a dense convolutional network predicts.

## 7. Model and profile measurements

### Controlled single-core RetinaFace forward pass

One RetinaFace model was warmed for each shape; each result is the median of three forward
passes on the installed stack.

| Network tensor | Pixels vs current | Median | Ratio vs current |
|---|---:|---:|---:|
| 1820x1024 padded | 100% | 5.2405 s | 1.000 |
| 1280x720 unpadded | 49.5% | 2.5637 s | 0.489 |
| 960x540 unpadded | 27.8% | 1.4364 s | 0.274 |

These reproduce the separate investigation's 3.97/1.95/1.12-second measurements to within
about one percentage point in relative ratios.

### Historical M4 profile

The two gallery-warm 30-frame windows reported:

- RetinaFace: 139.326 seconds across 60 frames (2.322 s/frame).
- Facenet512: 7.154 seconds across eight batches (0.119 s/frame).
- Steady perception after model construction: 2.663 s/frame across all three windows.
- Video read/write/draw: about 0.27-0.34 seconds per 30 frames.
- Projected all-new stride 1: 2 h 17 min.
- Projected with the then-390-frame cache: 2 h 00 min.

### Fresh exact-window profile on current `main`

Frames 1500-1529 were profiled cold with the gallery pre-cached and frame cache isolated:

| Metric | Fresh result | Historical same window |
|---|---:|---:|
| CLI elapsed | 44.883 s | 87.845 s |
| CLI perception | 44.574 s | 86.443 s |
| RetinaFace `detect_faces` | 36.218 s | 73.351 s |
| Facenet512 forward | 2.178 s | 3.434 s |
| Faces | 30 | 30 |

The same workload is now about twice as fast, despite identical pinned model versions.
TensorFlow thread counts are automatic and the historical report did not capture hardware,
thread scheduling, power, temperature, or background load. The exact source of the
variation is unresolved. It is evidence that one point estimate is unsafe; sustained
benchmarks must capture hardware and thread settings and report distributions.

Under the fresh rate, a naive all-new projection is approximately 66-70 minutes after
allowing for growing cache writes. Under the historical rate it remains roughly 2 h 17 min.
The repository's "up to two hours" statement is supported by measured history, but it is
not a stable invariant of the code.

## 8. Optimization experiments

### A. Plain unpadded resolutions

Twelve cached frames were selected to include 0, 1, 2, 3, 6, 8, 9, and 10-face scenes,
small faces, wide shots, and frame-edge detections.

| Detector input | Old boxes | New boxes | Old boxes matched at IoU >= 0.5 | Mean best IoU |
|---|---:|---:|---:|---:|
| 960x540 | 71 | 69 | 67 | 0.870 |
| 1280x720 | 71 | 66 | 64 | 0.811 |

Resolution is not monotonically related to parity. RetinaFace's anchor grids, scale, and
context alter decisions. The 960x540 misses included a genuine small background face on
frames 162 and 164. The sharper 1280x720 path missed more baseline boxes in this sample.
Neither is currently admissible under R1 without a changed acceptance rule.

On the 16-frame 1500 window, 960x540 boxes averaged IoU 0.960 against the baseline and the
new embeddings had mean cosine distance 0.0200 (maximum 0.0580) from baseline embeddings.
That is close but not identical; cache invalidation and matching regression checks are
mandatory.

### B. Split-pipeline prototype cost

The prototype performed direct RetinaFace detection, full-resolution crop-local alignment,
and batched Facenet512 `skip` embedding without modifying repository code.

| Window | Faces | RetinaFace input | Detect + align + embed | Seconds/frame |
|---|---:|---:|---:|---:|
| 1500-1529 | 30 | 960x540 | 11.885 s | 0.396 |
| 154-183 | 276 | 960x540 | 13.355 s | 0.445 |

The crowded window shows that Facenet512 becomes more visible after detector optimization,
but total cost remains dominated by detection. These are prototypes, not complete CLI runs.

### C. Exact-scale, feature-phase-preserving crop

An intermediate prototype resized the unpadded frame to approximately the same 910x512
content scale as the current detector and preserved the feature-grid phase. It matched
659/661 cached boxes over 202 frames and found one net additional box. The two misses were
both real faces, so this variant is informative but insufficient.

### D. Exact current resize, black-margin crop (recommended experiment)

The strongest candidate reproduces the current padded resize exactly, then trims black-only
margins by multiples of 32 before the network forward pass:

```text
current tensor: 1820x1024 = 910x512 content + about 455/256 black margins
candidate:       988x576 = same 910x512 pixels + 39/32 black halo
network area:    30.5% of current
```

The full 202-frame cached comparison produced:

- 661 existing detections; all 661 matched at IoU >= 0.5.
- 638/661 matched at IoU >= 0.75.
- Mean best IoU 0.944; minimum 0.571.
- 662 candidate detections: one additional box.
- Manual inspection confirmed the additional frame-222 box is a real small face.
- Pre-resize/pad/crop overhead: 0.780 seconds total (3.9 ms/frame).
- RetinaFace time during this later sustained run: 99.646 seconds (0.493 s/frame).

This result preserves the face scale, exact resized pixels, detector/threshold, and feature
grid phase while removing most black convolution. It is the preferred optimization
hypothesis. The 202-frame result is not permission to switch the production default: the
remaining 2,842 uncached frames include unseen scene types and must pass the same gate.

#### Complete M3 300-frame comparison

The original `cache/m3` directory was no longer present, so the baseline was reconstructed
without changing repository state: the 202 compatible cached frames were copied to an
isolated temporary cache and the unchanged production path computed the remaining 98.
The reconstructed baseline contained exactly 1,262 faces, matching the published M3 result.
The candidate was then run over all 300 frames, including crop-local alignment and batched
Facenet512 embedding, and compared face-by-face using maximum-IoU matching.

| Measure | Result |
|---|---:|
| Baseline faces | 1,262 |
| Candidate faces | 1,265 |
| Frames with identical face count | 297/300 |
| Baseline faces matched at IoU >= 0.5 | 1,262/1,262 |
| Lost baseline faces | 0 |
| Added candidate faces | 3 |
| IoU, mean / median | 0.948 / 0.959 |
| IoU, p05 / p95 / minimum | 0.895 / 0.988 / 0.571 |
| Embedding cosine similarity, mean / median | 0.978 / 0.983 |
| Embedding cosine similarity, p05 / p95 / minimum | 0.936 / 0.998 / 0.728 |
| Embedding cosine distance, p95 / p99 / maximum | 0.064 / 0.112 / 0.272 |
| Nearest gallery owner changed | 129/1,262 |
| Final assignment changed at threshold 0.30 | 5/1,262 |

The three count differences occurred at frames 221, 222, and 245. Each was one additional
candidate detection, and manual inspection confirmed all three boxes contain real small
background faces. No baseline detection was lost. The detector swap therefore improves
R1 on this sample, rather than merely preserving the old count.

Embeddings are close in aggregate but not cache-compatible or numerically identical. Most
nearest-owner changes are among weak `Unknown` matches; only five cross the configured
0.30 assignment decision. Those five must be reviewed explicitly, and M5 threshold evidence
must be rerun, before accepting the new perception pipeline for R2.

The complete candidate run took 279.530 seconds internally (0.932 seconds/frame) and
287.03 seconds process wall time. Its measured components were 252.923 seconds RetinaFace,
24.186 seconds embedding, 1.699 seconds preprocessing/alignment, and 2.601 seconds combined
model construction. The internal rate projects to 47.3 minutes of full-video perception.
It is 65.7% below the historical M3 815.849-second total, but that is not a controlled
apples-to-apples speedup ratio because machine state, output work, cache persistence, and
the historical run's two process startups differ.

## 9. Secondary efficiency findings

### FaceCache rewrite amplification — `OPTIMIZE`, secondary priority

`FaceCache.flush` serializes and compresses the complete accumulated archive after every
batch of eight. This gives interruption safety but creates quadratic cumulative work over a
long run. A synthetic workload shaped like the current cache measured one full flush at:

| Frames | Approx. faces | Flush | Archive |
|---|---:|---:|---:|
| 500 | 1,363 | 0.083 s | 2.51 MiB |
| 1,000 | 3,245 | 0.194 s | 5.98 MiB |
| 2,000 | 6,486 | 0.393 s | 11.94 MiB |
| 3,044 | 9,915 | 0.610 s | 18.25 MiB |

Piecewise interpolation over 380 eight-frame flushes gives about 104 seconds of compression
and serialization. Historical cProfile costs were somewhat higher. This is material but
cannot explain two hours. A chunked/append-safe cache could recover minutes after the
detector is fixed while preserving resume semantics.

### Instrumentation gap — `SIMPLIFY` when performance work begins

`model_load_seconds` records Facenet512 construction only. RetinaFace's lazy build is
included in perception, so the CLI's `model=` field under-reports total model startup and
overstates steady perception on the first batch. The historical profile corrected this
externally. Future benchmarks should expose detector build, detector forward, crop/align,
embedding forward, cache persistence, and video I/O separately.

### Eager recognition import — low priority

`recognition.py` imports DeepFace verification at module import. Even a fully cached replay
pays Python/TensorFlow import startup, which helps explain process wall time above the CLI's
internal elapsed time. This is seconds, not hours, and does not justify complexity before
the detector work.

### Confirmed non-bottlenecks

- Gallery indexing: ten photos, one-time, then cache-backed.
- Facenet512: batched and substantially smaller than RetinaFace in current profiles.
- Cosine matching: ten pins and one matrix-vector product per face.
- Drawing, CSV, H.264 decoding, and mp4v writing: collectively small in profiles.
- Model construction: roughly 5-10 seconds once per process, not per frame.

## 10. Module disposition matrix

| Module | Responsibility | Requirement link | Evidence | Disposition |
|---|---|---:|---|---|
| `label_video.py` | CLI/facade | 0.75 | Active entrypoint, 58 lines | KEEP |
| `contracts.py` | Stable records/constants | 1.00 | Used across all stages/tests | KEEP |
| `config.py` | Approved CLI/defaults | 1.00 | Active, contract-pinned | KEEP |
| `perception.py` | Models, detection, embedding | 1.00 | Measured dominant hotspot; DeepFace path adds redundant detector area | OPTIMIZE |
| `cache.py` | Identity and FaceCache | 1.00 | Required reuse; full-archive rewrite adds about minutes | OPTIMIZE |
| `gallery.py` | Curated references/pins | 1.00 | Required for R2, cached, low steady cost | KEEP |
| `recognition.py` | Pure cosine match | 1.00 | Required for R2, negligible hot-path cost | KEEP |
| `rendering.py` | Pure annotations | 1.00 | Required for R1/R2, negligible profile cost | KEEP |
| `evidence.py` | CSV/debug crops | 0.75 | Required deliverable, negligible default cost | KEEP |
| `video.py` | Planning/streaming/batching | 1.00 | Required coordinator, behavior covered; calls hotspot correctly per current contract | KEEP |
| `core.py` | Preflight/end-to-end orchestration | 1.00 | Required and low cost | KEEP |

No first-party module is unused, redundant, or a removal candidate. The current module
split is proportionate and makes the hotspot easier to isolate. The waste is mainly inside
the third-party orchestration selected by `perception.py`, not structural module bloat.

## 11. Candidate action frontier

Mandatory gates are R1, R2, deterministic CPU-only output, explicit fixed models,
compatible contracts, correct cache identity, failure recovery, and final stride 1.

| Action | Runtime | R1 risk | Change/contract cost | Status |
|---|---|---|---|---|
| Keep current DeepFace path | Historical 2 h 17 m projection; fresh about 66-70 m | Baseline | None | Admissible, slow |
| Final stride 2/3 | Roughly 1/2 or 1/3 detector work | Known stale/missed entry frames | Low | Inadmissible for final R1; preview only |
| Plain 960x540 split | Historical projection about 52 m; fresh prototype about 22-25 m | Known real-face misses in sample | Medium | Inadmissible without changed gate |
| Plain 1280x720 split | Historical projection about 79 m | More sample misses than 960 | Medium | Dominated in current evidence |
| Exact-resize 988x576 black-margin crop | Measured 0.932 s/frame; projects to about 47 m perception / about 50 m end-to-end | No baseline loss in all 300 M3 frames; full-video unknown | Medium | Preferred experiment |
| Cache sharding only | Saves roughly minutes | Low if atomic resume is preserved | Medium | Secondary optimization |
| GPU/Metal | Potentially faster | Outside approved CPU-only scope | High | Prohibited |

The exact-resize crop is nondominated in current evidence: it preserves the current face
scale and all sampled baseline detections while removing about 69.5% of network area. It
must still clear the hard gates; performance cannot average away a face-recall regression.

## 12. Recommended implementation experiment and exit conditions

### Proposed architecture

1. Keep a lazy, resident RetinaFace model and Facenet512 model.
2. Reproduce the current padded resize, crop black-only margins at feature-grid boundaries,
   and call `RetinaFace.detect_faces(... model=..., threshold=0.9,
   allow_upscaling=False)`.
3. Restore boxes and five landmarks to original full-resolution coordinates.
4. Call DeepFace's crop-local `detection.extract_face` on the original frame with zero
   outer borders and alignment enabled.
5. Batch all aligned crops from the configured frame group through
   `DeepFace.represent(... model_name="Facenet512", detector_backend="skip")`.
6. Reassemble the existing `Face` records, including deterministic ordering, clipped boxes,
   rounded confidence behavior, empty-frame behavior, and batch-fallback semantics.

This explicitly amends the current rule that every `represent` call uses RetinaFace: the
detector remains RetinaFace, but `represent` receives already detected/aligned crops. The
spec and plan must say so before implementation.

### Cache and compatibility requirements

- Add a perception-pipeline/preprocessing version and crop parameters to FaceCache identity,
  or bump the schema/key. Reusing old embeddings under the existing key would be incorrect.
- If gallery preprocessing also changes, update the gallery-cache identity independently.
- Do not silently overwrite compatible old caches; retain rollback/replay ability.
- Re-run matching/threshold evidence because the 300-frame comparison measured cosine
  distance up to 0.272 and five changed assignments at the configured 0.30 threshold.

### Required tests

- Exact coordinate/landmark scaling, frame-edge clipping, no-face frames, multi-face order,
  threshold 0.9, confidence semantics, and `max_faces=None`.
- Byte-equivalent crop-local alignment for identical detections, including edge faces.
- Flat/single and nested/batch DeepFace `skip` return shapes.
- A batch with one failure still retains other frames; failed frames retry next run.
- Deterministic repeat embeddings and output.
- Cache-key separation between old and new perception algorithms and zero-inference replay
  within each compatible variant.

### Performance and behavior gate

- Benchmark the same three existing 30-frame windows, in one controlled process after
  warm-up, with hardware, TensorFlow/OpenCV thread settings, wall/user time, and at least
  three repetitions captured.
- Practical target: at least 50% lower steady perception wall time and no material regression
  in another objective.
- Compare candidate against baseline across every frame for which baseline cache exists;
  before default switch, complete the planned stride-1 baseline or an owner-approved
  equivalent truth set covering all 3,044 frames.
- Require every baseline detection to have a candidate match at IoU >= 0.5, manually review
  every loss/addition and representative crowd/wide/motion/profile/edge scenes, and reject
  the change for any unexplained visible-face loss.
- Manually review the five changed 300-frame identity assignments and rerun the M5
  threshold/pin evaluation with the new cache variant before accepting R2 parity.
- Verify output frame count, FPS, duration, CSV schema/rows, failure handling, and warm
  zero-inference replay.
- Rollback: keep the existing perception path available until the owner accepts the new
  cache variant and evidence; switching back must select the old compatible cache.

For the cache optimization, the exit condition is a chunked or append-safe design that
loses at most the active batch on interruption, loads corrupt/missing chunks as bounded
misses, preserves key semantics, and reduces cumulative persistence time by at least 50%
on a 3,044-frame/approximately 10,000-face synthetic workload.

## 13. Baseline defect discovered during verification

The slow suite consistently fails
`tests/test_perception.py::test_real_frame_artifact_and_face_less_frame` before model work:
the helper calls `label_video.cv2`, but the approved modularization explicitly says the
facade does not proxy third-party globals such as `cv2`. Git blame shows the helper was
written in the M1 commit and was not migrated in the modularization commit. The two other
slow tests pass.

This is stale test coupling, not evidence of a perception runtime failure. The correct
future repair is to make the test own/import OpenCV (or patch the owning module), consistent
with the modularization plan. It was not changed here because this task authorized an
investigation, not implementation.

## 14. Reproduction commands and artifacts

Primary commands used:

```bash
.venv/bin/python -m pytest -m 'not slow' -q
.venv/bin/python -m pytest -m slow -q -s
.venv/bin/python -m pip check
.venv/bin/python -m cProfile -o /tmp/.../current_1500_30.pstats label_video.py \
  --input data/video-source/nimbus.mp4 --output /tmp/.../current_1500_30.mp4 \
  --ref-dir data/reference-images --start-frame 1500 --max-frames 30 \
  --stride 1 --batch-size 8 --cache-dir /tmp/.../fresh30 --csv /tmp/...csv
```

Controlled forward benchmarks used `TF_NUM_INTRAOP_THREADS=1` and
`TF_NUM_INTEROP_THREADS=1`. Operational prototypes used the repository's default automatic
TensorFlow threading. All profiles, temporary videos, CSVs, and visual comparison sheets
were written under `/tmp/whiteswan-investigation.HOLoyc`; repository caches were read but
not modified.

## 15. Final answer to the investigation question

The two-hour cost is not caused by Python modularization or video encoding. It comes from
running a 29.5-million-parameter RetinaFace network once for each of 3,044 frames on CPU,
through a DeepFace path that makes 75% of the detector tensor black before convolution.
Batch size cannot fix that because detection is not batched. Facenet512, matching, drawing,
and I/O are secondary, and the FaceCache makes later relabels fast.

The separate investigation found the right root cause. Plain 960x540 is fast but loses
known small-face detections. The higher-confidence route is to preserve RetinaFace's exact
current resized content and feature-grid phase while cropping most black margin, then use
the proposed full-resolution crop alignment and batched `skip` embedding. That variant
preserved every one of 1,262 baseline detections and found three real extra faces across the
complete 300-frame M3 sample. Mean matched-face IoU was 0.948 and mean embedding cosine
similarity was 0.978. The embeddings are not interchangeable with the old cache: five
thresholded identity assignments changed and require review. It is the optimization to
specify and test next, with a realistic target of roughly halving to quartering the current
cold-run time rather than promising a single number before the full gate.

## 16. Second audit: every process around RetinaFace and Facenet512

This follow-up interprets "both modules" as the two neural stages, RetinaFace detection and
Facenet512 recognition, and traces every first- and third-party operation that surrounds
them during frame analysis. It also covers the downstream per-frame pipeline so that a
non-model cost is not hidden by the model boundary.

### 16.1 No hidden third inference model

The normal video path loads only two neural models:

1. RetinaFace locates faces and returns a confidence, box, and five landmarks.
2. Facenet512 converts each aligned face crop into a 512-dimensional identity embedding.

DeepFace's model registry is singleton-like by task and name. Gallery and video calls to
Facenet512 therefore reuse the same object. The DeepFace RetinaFace adapter calls
`retinaface.RetinaFace.build_model()`, whose own module-level singleton is also returned by
the proposed direct detector path. A cold gallery followed by video inference may retain
two lightweight Python handles, but it does not load a second copy of the RetinaFace
weights.

There is one conditional detector that is not active in the observed path:
`detection.extract_face` can build OpenCV's eye detector when either eye landmark is
missing. RetinaFace returns both eye landmarks for its detections, so this fallback did not
run in the profile and the proposed mapping preserves those landmarks. Anti-spoofing,
demography, tracking, and smoothing are not called.

RetinaFace and Facenet512 do not expose a useful shared intermediate tensor. RetinaFace's
feature maps support box/landmark prediction over the full frame; Facenet512 requires an
eye-aligned 160x160 face image and produces an identity vector. Reusing RetinaFace feature
maps as Facenet embeddings would replace the fixed recognizer and violate the approved
stack rather than eliminate duplicated work.

### 16.2 Current cold-frame call graph

```text
OpenCV H.264 decode                                      once/output frame
  -> DeepFace frame loop                                once/selected frame
       -> validate NumPy image
       -> allocate half-frame black borders
       -> RetinaFace resize to short side 1024
       -> uint8 BGR -> float32 RGB tensor
       -> RetinaFace TensorFlow forward
       -> 3 FPN levels: anchors, boxes, landmarks
       -> threshold 0.9, score sort, NMS
       -> face-local doubled crop and eye rotation       once/detected face
       -> RGB conversion and [0,1] face normalization
  -> resize/pad each crop to 160x160
  -> configured Facenet input normalization
  -> concatenate crops into one batch
  -> Facenet512 TensorFlow forward                       once/frame batch
  -> L2-normalize 512-dimensional embeddings
  -> validate/copy Face records                          once/detected face
  -> rewrite compressed FaceCache                       once/frame batch
  -> 10-pin cosine match + confidence                   once/rendered face
  -> CSV row + box/label drawing + mp4v encode          once/output frame/face
```

At stride 1, every output frame is selected. At larger preview strides, decode, drawing,
CSV logging, and video writing still happen for every output frame, while neural perception
runs only on selected cache misses and skipped frames reuse the last annotations.

### 16.3 What each shared or repeated process is for

| Process | Cadence | Purpose | Duplicate or removable? |
|---|---|---|---|
| TensorFlow runtime/thread pools | Process lifetime | Executes both neural networks on CPU | Shared infrastructure; no duplicate model execution |
| DeepFace model registry | Each lookup, cached construction | Returns resident Facenet512 and detector wrappers | Repeated lookup is negligible; weights build once |
| OpenCV image operations | Frame/face | Decode, padding, resizing, rotation, drawing, encoding | Operations have different required outputs |
| RetinaFace resize and BGR-to-RGB conversion | Selected frame | Produces the detector's exact input scale and tensor format | Required; black-margin network area is the material waste |
| RetinaFace anchors/box/landmark decode/NMS | Selected frame | Converts dense outputs into final detections | Required; anchor grids could be cached by shape but measured cost is tiny |
| DeepFace local alignment | Detected face | Levels eyes and preserves edge context before recognition | Required for current embedding behavior |
| DeepFace `skip` image loading and channel views | Detected face in candidate | Adapts already aligned BGR arrays to the public represent API | Contains a cancelling RGB/BGR view pair, but measured/expected cost is negligible |
| 160x160 resize, padding, and normalization | Detected face | Produces Facenet512's fixed input | Required and distinct from detector preprocessing |
| Facenet512 batch concatenation/forward | Frame batch | Produces identity embeddings efficiently | Required; already batched |
| L2 normalization plus local norm validation | Detected face | Makes cosine distance valid and enforces the `Face` contract | Intentional validation after library normalization; negligible |
| FaceCache serialization | Frame batch | Makes interruptions resumable and later relabels inference-free | Required behavior, but full-archive rewrites are avoidable amplification |
| Matching/confidence | Rendered face | Chooses the nearest gallery pin and applies Unknown threshold | Required for R2; negligible at ten pins |
| Drawing/CSV/video encoding | Output frame/face | Produces the required deliverables and evidence | Required; measured non-bottlenecks |

Both networks resize and convert image data, but at different semantic boundaries.
RetinaFace consumes a full-scene tensor to find faces. Facenet512 consumes separate aligned
face tensors to recognize identities. Combining those resizes or color conversions would
change one model's input rather than remove redundant work.

### 16.4 Measured cost of the surrounding processes

The fresh 30-frame, 30-face cProfile gives the following nested measurements. Nested rows
must not be added to their parent totals.

| Operation | Calls | Cumulative wall time |
|---|---:|---:|
| RetinaFace detection, including preprocessing/postprocessing | 30 | 36.218 s |
| Facenet512 forward | 4 batches | 2.178 s |
| Full-frame `copyMakeBorder` | 30 | 0.033 s |
| RetinaFace spatial resize | 30 | 0.043 s |
| Anchor + box + landmark + clip + NMS functions | 390 combined | about 0.111 s |
| Face-local alignment/rotation | 30 | 0.040 s |
| Facenet 160x160 resize/pad | 30 | 0.027 s |
| Video decode | 30 | 0.042 s |
| Drawing | 30 | 0.031 s |
| Video encode/write | 30 | 0.116 s |
| Cosine matching | 30 | 0.001 s |
| CSV row writes | 31 including header | below profiler resolution |
| FaceCache flush | 4 | 0.010 s in this small fresh cache |

TensorFlow's low-level execution accounted for 35.722 seconds in the same profile, though
that aggregate includes calls from both networks and overlaps their cumulative totals. The
result confirms that there is no second Python/OpenCV process comparable to RetinaFace's
dense forward pass.

The operational environment uses CPU only, OpenCV reports 10 threads, and TensorFlow's
intra/inter-op settings are both `0` (automatic). The two networks run sequentially, so
their thread pools do not execute the models concurrently. Automatic scheduling remains a
source of run-to-run timing variance, not evidence of duplicated frame analysis.

### 16.5 What remains after the approved detector change

The 988x576 candidate deliberately retains two small costs:

- It first constructs the exact padded/resized image and then crops it, instead of trying
  to synthesize only the retained pixels. This preserves interpolation and feature phase.
  The complete 300-frame prototype measured preprocessing plus 1,265 local alignments at
  1.699 seconds total, so removing this allocation is not presently worth the parity risk.
- `DeepFace.represent(detector_backend="skip")` validates each NumPy crop and performs two
  opposite channel-slice views before its 160x160 resize. Calling the Facenet model's
  private `forward` API directly could remove that adapter work but would increase
  third-party coupling and bypass tested normalization/result handling without a measured
  material saving.

These are `DEFER`, not additions to the current implementation scope. The black-margin
network crop remains the only first-priority perception optimization.

After that change, the next measured optimization candidate is FaceCache persistence. A
full archive is compressed again every eight inferred frames. The earlier synthetic
projection is about 104 seconds over a 3,044-frame/approximately 10,000-face run. That is
only a few percent of the projected 50-minute candidate run, but it becomes the largest
known avoidable non-neural cost. It should remain a separate change because its hard gate
is interruption recovery, not detection recall.

Two still smaller simplification candidates are shape-keyed RetinaFace anchor caching and
replacing the progress logger's repeated scan of every selected cache entry with an
incremental face counter. Neither has evidence of a user-visible runtime effect, so both
remain `DEFER` until a post-switch profile shows otherwise.

The follow-up measurements are reproducible with:

```bash
.venv/bin/python -c 'import pstats; pstats.Stats("/tmp/whiteswan-investigation.HOLoyc/fresh30/current_1500_30.pstats").strip_dirs().sort_stats("cumulative").print_stats(80)'
.venv/bin/python -c 'import cv2, tensorflow as tf; print(cv2.getNumThreads()); print(tf.config.threading.get_intra_op_parallelism_threads()); print(tf.config.threading.get_inter_op_parallelism_threads()); print(tf.config.list_physical_devices())'
```

### 16.6 Second-audit conclusion

No newly discovered per-frame process blocks or changes the approved optimization. The
models are not redundantly loaded, no hidden analysis network runs, and their preprocessing
serves different inputs. The proposed split removes the dominant waste while retaining the
necessary alignment and Facenet preparation steps. Do not broaden the current change to
private Facenet calls, custom RetinaFace postprocessing, cache sharding, anchor caching, or
thread tuning; measure the production switch first, then reconsider FaceCache persistence
as an independent optimization.
