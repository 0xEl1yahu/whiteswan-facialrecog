# RetinaFace Black-Margin Optimization Design

**Status:** Approved by Eli on 2026-09-26

**Date:** 2026-09-26

**Authority:** This document is the owner-approved M4 perception amendment to
`docs/design/design-plan.md` and `docs/implementation/implementation-plan.md`. Those
documents retain authority for every requirement not explicitly amended here.

## 1. Intent

Replace the current video-frame perception call with the measured, R1-preserving
RetinaFace path validated during the runtime investigation. The new path removes most of
the black border from RetinaFace's network input while preserving the current resized
image pixels, face scale, 32-pixel feature-grid phase, detector, threshold, alignment,
recognizer, normalization, stride, and output contracts.

The change applies only to video-frame perception. Gallery photos continue through the
existing DeepFace RetinaFace/Facenet512 path so their cache and owner-curated evidence
remain valid.

This is an amendment inside the unfinished M4 milestone. It must pass the 300-frame M3
comparison gate and stop for owner review before the full 3,044-frame run.

## 2. Evidence and Decision

DeepFace 0.0.101 adds a black border equal to half the source width and height on every
side when alignment is enabled. For the 1920x1080 source, RetinaFace therefore receives a
resized 1820x1024 tensor whose source-image content occupies only about 910x512 pixels.
Exactly 75% of the tensor area is black.

The approved candidate reproduces that padded resize, then removes black-only margins in
multiples of RetinaFace's coarsest 32-pixel feature stride, retaining one 32-pixel stride
of black context. For the source video, the resulting detector input is 988x576, or 30.5%
of the current network area.

Across the complete 300-frame M3 sample, the prototype:

- matched all 1,262 baseline faces at IoU >= 0.5;
- lost no baseline detections;
- found three additional manually confirmed real faces;
- produced mean/median IoU of 0.948/0.959;
- produced mean/median embedding cosine similarity of 0.978/0.983;
- changed five final identity assignments at threshold 0.30; and
- ran at 0.932 seconds/frame, projecting to 47.3 minutes of perception and roughly
  50 minutes end-to-end for all 3,044 frames.

Plain unpadded 960x540 detection is rejected because it missed known small faces. Running
the old detector as a fallback is rejected because it restores most of the avoided compute
and adds an unproved trigger decision.

## 3. Goals

1. Preserve or improve R1 relative to the current per-frame baseline.
2. Preserve the existing R2 pipeline while explicitly revalidating changed embeddings.
3. Remove black-only RetinaFace convolution without reducing face scale.
4. Keep RetinaFace and Facenet512 resident and lazy.
5. Batch Facenet512 embeddings across every aligned face in the configured frame batch.
6. Preserve the CLI, `Face` contract, output formats, batch fallback, and CPU-only behavior.
7. Keep old FaceCache variants available for rollback while preventing incompatible reuse.
8. Provide enough timing instrumentation to distinguish detector, recognizer, and model
   construction costs at the M4 gate.

## 4. Non-goals

- No GPU, Metal, event gate, periodic detection, tracking, smoothing, or stride change.
- No plain 960x540 or 1280x720 detector path.
- No change to gallery preprocessing or gallery cache identity.
- No change to the five character names, cosine matching, threshold default, pin strategy,
  CSV schema, annotations, video codec, frame cadence, or CLI flags.
- No automatic reuse or conversion of old FaceCache embeddings.
- No promise that a projected full-video runtime is a measured full-video result.
- No full 3,044-frame run before the amended M4 STOP gate is approved.

## 5. Configuration and Cache Identity

`Config` gains two internal, non-CLI fields:

```python
perception_pipeline: str = "retinaface_exact_resize_crop_v1"
detector_black_halo: int = 32
```

The pipeline name describes the complete upstream frame-perception algorithm. The halo is
one coarsest RetinaFace feature stride and is a cache-relevant preprocessing input. Neither
field is exposed as a command-line tuning control in M4.

The FaceCache schema increments from 1 to 2. Its metadata and key add both fields. Existing
schema-1 files retain their filenames and contents and are not deleted or overwritten; the
new metadata selects a separate schema-2 filename. Threshold, pin strategy, stride, batch
size, frame window, rendering, and output paths remain excluded from the key.

The gallery cache metadata and key remain byte-for-byte unchanged because gallery
preprocessing does not change.

## 6. Model Lifecycle

`face_labeller.perception` owns two lazy model handles:

- Facenet512, constructed through `DeepFace.build_model("Facenet512")`; and
- RetinaFace, constructed through `RetinaFace.build_model()`.

Each is built at most once per process and only when uncached work requires it. A fully
cached gallery and FaceCache replay builds neither model. A frame with no detected faces
does not require Facenet512 embedding for that frame.

Instrumentation reports Facenet512 and RetinaFace construction separately and retains a
combined model-load value for the existing run summary.

## 7. Detector Preprocessing

For each BGR frame of width `W` and height `H`:

1. Set `width_border = int(0.5 * W)` and `height_border = int(0.5 * H)`, matching
   DeepFace 0.0.101.
2. Add those black borders with `cv2.copyMakeBorder`.
3. Call retina-face 0.0.18's `preprocess.resize_image` with scales `[1024, 1980]` and
   `allow_upscaling=True`, preserving its returned scale and exact resized pixels.
4. For each axis, calculate the removable black margin as the largest non-negative
   multiple of 32 that leaves at least `detector_black_halo` resized pixels before the
   original-image content boundary. For border `B` and resize scale `S`, the cut is
   `max(0, floor((B * S - detector_black_halo) / 32) * 32)`.
5. Crop the same amount from opposite sides. Cuts remain multiples of 32, preserving the
   origin of all RetinaFace FPN grids with strides 32, 16, and 8.
6. Call `RetinaFace.detect_faces` with the resident detector model, threshold `0.9`, and
   `allow_upscaling=False`. This prevents a second spatial resize; the call still performs
   RetinaFace's required tensor/color preprocessing and postprocessing.

The threshold, target size, maximum size, and FPN strides are pinned upstream RetinaFace
0.0.18 facts, not new tuning controls. Installed RetinaFace and DeepFace versions remain
part of the cache identity.

## 8. Coordinate Restoration and Alignment

Every detected box corner and landmark is restored to full-frame coordinates by adding
the detector crop offset, dividing by RetinaFace's first resize scale, and subtracting the
original DeepFace border. Integer conversion follows the installed adapter's truncation
behavior.

The restored values populate DeepFace's `FacialAreaRegion`. Alignment calls
`deepface.modules.detection.extract_face` on the original full-resolution BGR frame with:

```python
align=True
expand_percentage=cfg.expand_percentage
width_border=0
height_border=0
detector_backend="retinaface"
```

DeepFace's local alignment helper already pads its face-local sub-image when an edge face
needs context. The obsolete full-frame border is therefore not required for this stage.

Published boxes and landmarks retain current DeepFace sanitation semantics: boxes clip to
the source frame, invalid/out-of-frame landmarks are removed, detection confidence is
rounded to two decimals, and zero-area boxes remain safely representable. `max_faces`, when
configured, retains the largest detected regions using the current area rule. With the
default `None`, no face is capped.

## 9. Batched Embedding and Result Assembly

Aligned crops from every frame in one configured batch are flattened into one ordered
list. A single `DeepFace.represent` call embeds that list with:

```python
model_name="Facenet512"
detector_backend="skip"
enforce_detection=False
align=True
normalization=cfg.normalization
max_faces=None
l2_normalize=True
expand_percentage=0
```

`detector_backend="skip"` is permitted only here, after explicit RetinaFace detection and
DeepFace alignment. Every actual detector call remains RetinaFace. The implementation
normalizes DeepFace's flat single-image and nested multi-image return shapes, validates
each 512-dimensional float32 unit embedding, and reassembles `list[list[Face]]` in frame
and detector order using the saved boxes, landmarks, and confidences.

If no faces are detected in the batch, no Facenet512 call is made and one empty face list
is returned for each frame.

## 10. Error Handling and Determinism

The public `embed_faces(frames, cfg)` and `process_batch_with_fallback` interfaces remain
unchanged. A detector, alignment, shape, or embedding failure propagates from the batch;
the existing caller retries each frame separately, caches recovered frames as `ok`, and
marks only persistent failures as `failed` for the next run.

Detections retain RetinaFace's score order. `max_faces` uses the existing deterministic
largest-area selection. Flattening and reassembly preserve frame order. No random operation
or new parallel worker is introduced.

## 11. Files and Responsibilities

- `face_labeller/config.py`: internal pipeline identity and halo defaults; no CLI change.
- `face_labeller/cache.py`: FaceCache schema-2 metadata/key separation.
- `face_labeller/perception.py`: lazy RetinaFace lifecycle, exact resize/crop, coordinate
  restoration, local alignment, batched skip embedding, and timing accessors.
- `tests/test_config.py`: fixed internal defaults and unchanged CLI assertions.
- `tests/test_face_cache.py`: old/new algorithm key separation and downstream exclusions.
- `tests/test_perception.py`: preprocessing, mapping, detection, alignment, embedding,
  ordering, clipping, no-face, max-face, lazy-load, and failure behavior.
- `tests/test_video_integration.py`: unchanged external video/CSV behavior and warm replay.
- `docs/design/design-plan.md`: architecture/model-fact amendment and M4 gate link.
- `docs/implementation/implementation-plan.md`: test-first optimization task and STOP gate.
- `README.md` and `AGENTS.md`: runtime rationale, cache invalidation, and current checkpoint.

The existing stale slow-test helper is corrected to import and use OpenCV directly rather
than relying on the intentionally unsupported `label_video.cv2` facade attribute.

## 12. Test Strategy

Tests are written and observed failing before each production behavior is implemented.

Unit coverage must prove:

- 1920x1080 produces the exact 988x576 candidate detector tensor and expected offsets;
- another landscape size and an odd-sized frame produce valid symmetric, grid-aligned cuts;
- the cropped tensor contains the exact pixels from the current padded RetinaFace resize;
- detection uses the resident RetinaFace model, threshold 0.9, and no second upscaling;
- boxes and all five landmarks map back correctly and invalid landmarks are dropped;
- edge-face alignment uses the original frame with zero full-frame border;
- no-face, one-face, multi-face, multi-frame, and `max_faces` behavior is deterministic;
- one embedding call receives every aligned crop in batch order with explicit Facenet512
  and `skip` arguments;
- empty batches do not call Facenet512;
- single-crop and multi-crop DeepFace result shapes normalize correctly;
- invalid embeddings and result-count mismatches fail clearly;
- both models are lazy and idempotent; and
- FaceCache schema/key changes for the pipeline or halo but not downstream choices.

The normal fast suite, dependency check, CLI help, and applicable slow suite run before
the acceptance comparison. The stale slow test must be repaired rather than waived.

## 13. M4 Acceptance Gate

Reconstruct or reuse the complete 300-frame M3 baseline in an isolated cache. Run the
production implementation over the same frames and report:

- per-frame old/new face counts;
- greedy matched boxes at IoU >= 0.5;
- IoU mean, median, p05, p95, and minimum;
- cosine similarity/distance distribution for matched embeddings;
- added and lost detections with manual inspection;
- nearest-gallery-owner changes and final assignment changes at threshold 0.30;
- RetinaFace, Facenet512, alignment/preprocessing, cache, and total wall timings; and
- warm schema-2 replay proving zero model and inference calls.

Acceptance requires:

1. every one of the 1,262 M3 baseline detections has a candidate match at IoU >= 0.5;
2. no manually inspected visible baseline face is lost;
3. every added detection is manually classified as a real face or explained artifact;
4. all changed thresholded identity assignments are enumerated for owner review;
5. output contracts and deterministic repeat behavior remain intact; and
6. steady perception wall time is at least 50% below the historical M3 rate without a
   material regression in another objective.

Any unexplained visible-face loss rejects the switch. A failed gate rolls back by selecting
the unchanged production code from before this amendment and its preserved schema-1 cache.

After this gate, stop and report. Do not run all 3,044 frames until Eli approves the
implementation evidence.

## 14. Gallery Perception Consistency Correction

Owner approval on 2026-09-27 extends the optimized perception contract to reference
photos. The accepted video path detects, aligns, and embeds through
`face_labeller.perception.embed_faces`, while the original gallery path still called
`DeepFace.represent(... detector_backend="retinaface")` directly. That left the two sides
of each cosine comparison on slightly different crop pipelines.

Gallery indexing must now decode each owner-curated image and call the same
`embed_faces([image], replace(cfg, max_faces=1))` boundary used for video perception. An
empty result retains the existing invalid-reference warning/skip behavior. The gallery
continues to store one L2-normalised Facenet512 embedding per valid photo and the public
CLI, data contracts, threshold, normalization, and `all` pin strategy remain unchanged.

Gallery cache schema 2 adds `perception_pipeline` and `detector_black_halo` to its metadata
and key. This deliberately invalidates old-path gallery embeddings. The video FaceCache is
independent and unchanged: a compatible 3,044-frame cache replay must perform no RetinaFace
or Facenet512 video inference. The approved implementation and evidence steps are in
`docs/superpowers/plans/2026-09-27-gallery-perception-consistency.md`.
