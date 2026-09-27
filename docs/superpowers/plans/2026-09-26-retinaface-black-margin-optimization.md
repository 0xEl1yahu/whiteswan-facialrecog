# RetinaFace Black-Margin Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace redundant black-border RetinaFace computation with the approved exact-resize 988x576 crop path while preserving every sampled baseline face, the public CLI/data contracts, deterministic CPU execution, and recoverable cache behavior.

**Architecture:** `face_labeller.perception` will own a lazy direct RetinaFace detector, exact padded-resize/crop preprocessing, full-frame coordinate restoration, DeepFace crop-local alignment, and one batched Facenet512 `skip` call. `Config` and FaceCache metadata will version the new upstream algorithm, while gallery preprocessing and every downstream matching/rendering interface remain unchanged.

**Tech Stack:** Python 3.11, DeepFace 0.0.101, retina-face 0.0.18, TensorFlow/tf-keras 2.21.0, OpenCV 5.0.0.93, NumPy 2.4.6, pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-09-26-retinaface-black-margin-optimization-design.md`

## Global Constraints

- Preserve R1: no sampled baseline face may be lost, Unknown faces remain boxed, `max_faces` defaults to `None`, and final delivery remains stride 1.
- Preserve R2 interfaces: `Face`, `Gallery`, `Match`, cosine matching, threshold 0.30, `all` pins, CSV columns, and label rendering do not change.
- The detector remains RetinaFace and the recognizer remains Facenet512. Every actual detection call uses RetinaFace threshold 0.9; `detector_backend="skip"` is allowed only for already detected and aligned crops.
- CPU only. Add no GPU, Metal, alternate detector/model, worker pool, dependency, or CLI flag.
- Keep gallery preprocessing and gallery cache metadata/key byte-for-byte compatible.
- Set `perception_pipeline="retinaface_exact_resize_crop_v1"`, `detector_black_halo=32`, and FaceCache schema version 2. Both new fields are upstream FaceCache inputs and are not CLI controls.
- Preserve schema-1 cache files on disk but never load them as schema-2 results. Threshold, pin strategy, stride, batch size, window, rendering, and output choices remain excluded from FaceCache identity.
- Keep both neural models lazy and resident. A fully cached replay builds neither model; a no-face batch does not call Facenet512.
- Preserve `embed_faces(frames, cfg) -> list[list[Face]]` and `process_batch_with_fallback` error/retry semantics.
- Use test-driven development: add one behavior test, observe the expected failure, implement the minimum behavior, and rerun the owning and full relevant suites.
- Do not commit unless Eli explicitly asks. Conditional commit steps below do not grant permission.
- Do not run the full 3,044-frame video. Stop after the complete 300-frame M3 comparison and warm-replay gate.

## Review Focus

- A face touching or crossing a frame edge must use local black context for alignment, publish a clipped non-negative box, and drop only invalid landmarks; Task 3 tests it.
- An odd-sized, portrait, or extreme-aspect frame must never produce a negative, asymmetric, empty, or non-grid-aligned detector crop; Task 2 tests it.
- A frame batch containing zero faces, one face, and several faces must preserve one result list per input frame and make no unnecessary Facenet call for the all-empty case; Task 4 tests it.
- Detection order, `max_faces`, and flat-versus-nested DeepFace return shapes must not reorder embeddings or attach an embedding to the wrong frame/box; Tasks 3 and 4 test it.
- A fully compatible schema-2 replay must perform zero RetinaFace and Facenet512 work, while schema-1 or changed-pipeline caches remain clean misses; Tasks 1, 5, and 7 test it.

---

### Task 1: Version the Perception Configuration and FaceCache

**Files:**
- Modify: `face_labeller/config.py:13-50`
- Modify: `face_labeller/cache.py:18-70`
- Modify: `tests/test_config.py:27-52`
- Modify: `tests/test_face_cache.py:63-177`

**Interfaces:**
- Consumes: existing immutable `Config`, `face_cache_metadata`, `face_cache_key`, and unchanged gallery cache key.
- Produces: `DEFAULT_PERCEPTION_PIPELINE`, `DEFAULT_DETECTOR_BLACK_HALO`, `Config.perception_pipeline`, `Config.detector_black_halo`, and schema-2 FaceCache metadata.

- [x] **Step 1: Add failing configuration-default tests**

Extend `test_cli_defaults_match_design_spec` to assert the two internal defaults while asserting `build_parser().format_help()` exposes no pipeline or halo flag:

```python
assert cfg.perception_pipeline == "retinaface_exact_resize_crop_v1"
assert cfg.detector_black_halo == 32
assert "perception-pipeline" not in help_text
assert "detector-black-halo" not in help_text
```

- [x] **Step 2: Run the configuration test and observe RED**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_cli_defaults_match_design_spec -v`

Expected: FAIL because `Config` has no perception-pipeline or halo fields.

- [x] **Step 3: Add the internal configuration defaults**

Add the two named defaults and frozen dataclass fields in `face_labeller/config.py`. Do not add parser arguments; `load_config` must obtain the dataclass defaults when the namespace omits them.

- [x] **Step 4: Pass the configuration suite**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`

Expected: PASS with the public CLI unchanged.

- [x] **Step 5: Add failing FaceCache identity tests**

Update the literal expected metadata to schema 2 with both new fields. Add parameterized cases proving either field changes the key, retain the existing downstream-exclusion test, and retain the schema-1 gallery golden key `799c07df...f0fc` unchanged. Replace the old FaceCache golden with the hand-calculated schema-2 value after the literal-payload assertion establishes its contents.

- [x] **Step 6: Run the cache-key tests and observe RED**

Run: `.venv/bin/python -m pytest tests/test_face_cache.py -k 'key or metadata' -v`

Expected: FAIL because FaceCache still reports schema 1 and omits the new upstream inputs.

- [x] **Step 7: Implement schema-2 metadata**

Set `FACE_CACHE_SCHEMA_VERSION = 2` and add `perception_pipeline` and `detector_black_halo` to `face_cache_metadata`. Do not change NPZ array fields or gallery metadata.

- [x] **Step 8: Pass cache identity and persistence tests**

Run: `.venv/bin/python -m pytest tests/test_face_cache.py tests/test_gallery.py -v`

Expected: PASS; schema-2 FaceCache persistence works and the gallery golden key is unchanged.

- [x] **Step 9: Record the Task 1 verification without committing**

This is one amendment inside M4. Preserve the green state for the single conditional milestone commit after Task 7.

---

### Task 2: Build the Exact RetinaFace Detector Input

**Files:**
- Modify: `face_labeller/perception.py:1-96`
- Modify: `tests/test_perception.py:1-216`

**Interfaces:**
- Consumes: BGR `np.ndarray` frame and `Config.detector_black_halo`.
- Produces: internal frozen `_DetectorInput(image, scale, crop_x, crop_y, width_border, height_border)` and `_prepare_detector_input(frame, black_halo) -> _DetectorInput`.

- [x] **Step 1: Add a failing 1920x1080 transform test**

Create a deterministic non-black frame and assert `_prepare_detector_input(frame, 32)` returns scale `1024 / 2160`, borders `960/540`, offsets `416/224`, and an image of shape `(576, 988, 3)`. Independently construct the padded frame with OpenCV and retina-face's installed `resize_image`, then assert the returned image equals the literal `[224:800, 416:1404]` slice.

- [x] **Step 2: Run the exact transform test and observe RED**

Run: `.venv/bin/python -m pytest tests/test_perception.py::test_prepare_detector_input_preserves_exact_resized_pixels -v`

Expected: FAIL because `_prepare_detector_input` does not exist.

- [x] **Step 3: Implement `_DetectorInput` and `_prepare_detector_input`**

Define named upstream constants for RetinaFace threshold 0.9, resize scales `(1024, 1980)`, and coarsest FPN stride 32 with source comments. Implement, for border `B` and scale `S`, `max(0, floor((B * S - black_halo) / 32) * 32)`, then crop symmetrically. Reject non-3D/empty frames, negative halo, cuts that would make an empty tensor, or a halo not divisible by 32.

- [x] **Step 4: Pass the exact transform test**

Run: `.venv/bin/python -m pytest tests/test_perception.py::test_prepare_detector_input_preserves_exact_resized_pixels -v`

Expected: PASS.

- [x] **Step 5: Add failing geometry-boundary tests**

Add table-driven tests for an odd-sized landscape frame, portrait frame, extreme-aspect frame whose short-axis cut becomes zero, and invalid empty/negative-halo/non-grid-halo inputs. Assert cuts are symmetric multiples of 32, retained margins meet the configured halo when removable space permits, and output dimensions remain positive.

- [x] **Step 6: Run the geometry tests and observe RED where behavior is missing**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k 'prepare_detector_input' -v`

Expected: at least one new boundary assertion FAILS before validation/rounding is complete.

- [x] **Step 7: Complete the minimum validation and rounding behavior**

Keep all geometry in `_prepare_detector_input`; do not introduce a general image-transform abstraction or alternate resolution mode.

- [x] **Step 8: Pass all detector-input tests**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k 'prepare_detector_input' -v`

Expected: PASS.

- [x] **Step 9: Record the Task 2 verification without committing**

Preserve the green state for the single conditional milestone commit after Task 7.

---

### Task 3: Add Lazy RetinaFace Detection, Mapping, and Local Alignment

**Files:**
- Modify: `face_labeller/perception.py:15-96`
- Modify: `face_labeller/gallery.py:70-89`
- Modify: `tests/test_perception.py:57-216`
- Modify: `tests/test_gallery.py`

**Interfaces:**
- Consumes: `_DetectorInput`, direct retina-face detection dictionaries, DeepFace `FacialAreaRegion`, and crop-local `detection.extract_face`.
- Produces: `build_detector_model() -> Any`, `detector_model_load_seconds() -> float`, `facenet_model_load_seconds() -> float`, internal frozen `_AlignedFace(box, landmarks, det_conf, image)`, and `_detect_and_align_faces(frame, cfg) -> list[_AlignedFace]`.

- [x] **Step 1: Add failing detector lifecycle tests**

Reset both model states in the autouse fixture. Assert two `build_detector_model()` calls invoke `RetinaFace.build_model()` once, two `build_models(cfg)` calls invoke `DeepFace.build_model("Facenet512")` once, and `model_load_seconds()` equals the sum exposed by the two stage-specific accessors. Extend `test_gallery_cache_cold_then_warm_avoids_all_model_work` to require both builders on cold photo misses and neither builder on a fully cached gallery replay.

- [x] **Step 2: Run the lifecycle tests and observe RED**

Run: `.venv/bin/python -m pytest tests/test_perception.py tests/test_gallery.py -k 'model_load or detector_model or cold_then_warm' -v`

Expected: FAIL on missing detector lifecycle and timing accessors.

- [x] **Step 3: Implement separate lazy model lifecycle**

Keep imports lazy through `_get_deepface()`, `_get_retinaface()`, and `_get_deepface_detection()` so a compatible cache replay does not import/build model implementations. Hold the RetinaFace model object, track each build time separately, and retain `model_load_seconds()` as their sum for `video.py` compatibility. On a gallery-photo cache miss, call both lazy builders before the unchanged DeepFace gallery `represent` call; retina-face's module singleton makes the adapter reuse the same detector object and lets model timing include cold gallery construction without changing gallery embeddings or identity.

- [x] **Step 4: Pass lifecycle tests**

Run: `.venv/bin/python -m pytest tests/test_perception.py tests/test_gallery.py -k 'model_load or detector_model or cold_then_warm' -v`

Expected: PASS.

- [x] **Step 5: Add failing detector-call and mapping tests**

Use specific fakes for a detector response containing a box and all five landmarks. Assert `_detect_and_align_faces` calls:

```python
RetinaFace.detect_faces(
    prepared.image,
    model=resident_model,
    threshold=0.9,
    allow_upscaling=False,
)
```

Assert every coordinate is restored as `int((value + crop_offset) / scale - border)`, the raw restored region is passed to `extract_face` on the original frame, and alignment arguments are exactly `align=True`, `expand_percentage=cfg.expand_percentage`, zero borders, and `detector_backend="retinaface"`.

- [x] **Step 6: Run the detector/mapping tests and observe RED**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k 'detect_and_align' -v`

Expected: FAIL because `_detect_and_align_faces` does not exist.

- [x] **Step 7: Implement direct detection, mapping, and `_AlignedFace` assembly**

Preserve RetinaFace dictionary score order. Build raw `FacialAreaRegion` values for alignment, then publish DeepFace-compatible metadata: `x/y` clipped to at least zero, `w/h` capped at `frame_dimension - origin - 1` and never negative, invalid landmarks dropped, and confidence rounded to two decimals. Apply `cfg.max_faces` with the current largest-area rule before alignment.

- [x] **Step 8: Add failing edge, order, max-face, and empty-result tests**

Cover a detection extending beyond each frame edge, an invalid landmark outside the frame, three equal/different-area detections, `max_faces=2`, and a `{}` detector result. Assert local alignment still receives the original frame, ordering is deterministic, no OpenCV eye fallback is needed when RetinaFace eyes exist, and the empty result returns `[]`.

- [x] **Step 9: Run the new behavior tests and observe RED where incomplete**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k 'detect_and_align or max_faces or edge' -v`

Expected: at least one boundary/order assertion FAILS before the implementation is complete.

- [x] **Step 10: Complete the minimum boundary/order behavior and pass Task 3 tests**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k 'model_load or detector_model or detect_and_align or max_faces or edge' -v`

Expected: PASS.

- [x] **Step 11: Record the Task 3 verification without committing**

Preserve the green state for the single conditional milestone commit after Task 7.

---

### Task 4: Batch Facenet512 Embeddings and Reassemble Frames

**Files:**
- Modify: `face_labeller/perception.py:40-156`
- Modify: `tests/test_perception.py:87-216`
- Verify: `tests/test_face_cache.py:250-419`

**Interfaces:**
- Consumes: ordered `_AlignedFace` records from Task 3 and DeepFace's single/multi-image `represent` shapes.
- Produces: the unchanged public `embed_faces(frames, cfg) -> list[list[Face]]` using one Facenet512 call over all aligned crops.

- [x] **Step 1: Replace the old represent-call test with a failing split-pipeline test**

Fake `_detect_and_align_faces` for two frames with three total crops. Assert one `DeepFace.represent` call receives the three aligned BGR arrays in frame/detection order and the exact arguments:

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

Assert the returned embeddings attach to the saved boxes, landmarks, and confidences in the same order.

- [x] **Step 2: Run the split-pipeline test and observe RED**

Run: `.venv/bin/python -m pytest tests/test_perception.py::test_embed_faces_batches_aligned_crops_and_reassembles_frames -v`

Expected: FAIL because `embed_faces` still asks DeepFace to detect whole frames.

- [x] **Step 3: Implement crop flattening, one batch embedding, and frame reassembly**

Call `_detect_and_align_faces` once per input frame. If the flattened crop list is empty, return one empty list per frame without calling `build_models` or `DeepFace.represent`. Otherwise build Facenet512 lazily, normalize the flat-single versus nested-multi response, require exactly one embedding result per crop, validate every embedding, and construct `Face` from its paired `_AlignedFace` metadata.

- [x] **Step 4: Pass the split-pipeline test**

Run: `.venv/bin/python -m pytest tests/test_perception.py::test_embed_faces_batches_aligned_crops_and_reassembles_frames -v`

Expected: PASS.

- [x] **Step 5: Add failing shape, empty, validation, and ordering tests**

Cover one crop returning a flat `list[dict]`, multiple crops returning `list[list[dict]]`, mixed zero/one/many-face frames, all-empty frames, wrong result counts, an empty per-crop result, wrong-sized/non-unit embeddings, and an empty input frame list. Name each test for the production mutation it catches.

- [x] **Step 6: Run the embedding tests and observe RED where handling is incomplete**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k 'embed_faces' -v`

Expected: at least one new shape/count assertion FAILS before normalization is complete.

- [x] **Step 7: Complete result normalization without adding a second embedding path**

Remove superseded whole-frame result conversion only after all replacement tests are green. Keep `process_batch_with_fallback` unchanged except for imports/private helper names required by the refactor.

- [x] **Step 8: Pass perception and fallback suites**

Run: `.venv/bin/python -m pytest tests/test_perception.py tests/test_face_cache.py -v`

Expected: PASS, including individual retry isolation after a batch failure.

- [x] **Step 9: Record the Task 4 verification without committing**

Preserve the green state for the single conditional milestone commit after Task 7.

---

### Task 5: Prove Video Integration, Lazy Replay, and the Real Slow Path

**Files:**
- Modify: `tests/test_video_integration.py`
- Modify: `tests/test_perception.py:321-360`
- Verify: `face_labeller/video.py`
- Verify: `face_labeller/core.py`
- Verify: `label_video.py`

**Interfaces:**
- Consumes: schema-2 FaceCache, unchanged video coordinator, and the new `embed_faces` implementation.
- Produces: regression evidence that CLI/output behavior, cache replay, model accounting, and real-media perception remain valid.

- [x] **Step 1: Add an integration regression assertion for schema-2 warm replay**

Extend the existing fully cached replay test so both `_get_retinaface().build_model` and `DeepFace.build_model/represent` fail if called. Assert the second run still writes the expected frames/CSV from the schema-2 cache.

- [x] **Step 2: Run the focused integration test**

Run: `.venv/bin/python -m pytest tests/test_video_integration.py -k 'cached or replay' -v`

Expected: PASS only if the new detector remains behind the existing missing-frame boundary; a forbidden model call is a regression to fix before continuing.

- [x] **Step 3: Repair the pre-existing slow-test ownership defect**

Import `cv2` directly in `tests/test_perception.py` and replace the unsupported `label_video.cv2` references. Do not re-export OpenCV from the facade.

- [x] **Step 4: Run the real-frame slow test**

Run: `.venv/bin/python -m pytest tests/test_perception.py::test_real_frame_artifact_and_face_less_frame -v -s`

Expected: PASS; frame 150 still has two faces, frame 0 has none, and the artifact is written.

- [x] **Step 5: Run all directly affected tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_perception.py tests/test_face_cache.py tests/test_gallery.py tests/test_video_integration.py tests/test_core.py tests/test_video_plan.py -v
```

Expected: PASS.

- [x] **Step 6: Run complete fast and slow verification**

Run:

```bash
.venv/bin/python -m pytest -m "not slow" -v
.venv/bin/python -m pytest -m slow -v
.venv/bin/python -m pip check
.venv/bin/python label_video.py --help
```

Expected: all tests pass, no broken requirements, and the CLI has no new option.

- [x] **Step 7: Record the Task 5 verification without committing**

Preserve the green state for the single conditional milestone commit after Task 7.

---

### Task 6: Amend Authoritative Documentation and Prepare the Gate Record

**Files:**
- Modify: `docs/design/design-plan.md`
- Modify: `docs/implementation/implementation-plan.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Create: `docs/results/m4-retinaface-black-margin-optimization.md`
- Verify: `docs/reviews/CODEBASE-MODULE-AUDIT.MD`

**Interfaces:**
- Consumes: approved design, passing implementation behavior, and established M3/M4 evidence.
- Produces: one authoritative description of the selected M4 perception path and an initially empty results table for Task 7.

- [x] **Step 1: Amend the design plan without weakening R1/R2**

Link the approved optimization design from sections 4, 6, 9, and 10. Replace the obsolete statement that downscaling can never save work with the narrower evidence: naive unpadded downscaling is rejected, while exact padded resize followed by grid-aligned black-margin cropping is approved. Specify direct RetinaFace detection plus `skip` embedding and schema-2 cache identity.

- [x] **Step 2: Amend the milestone implementation sequence**

Add an M4 optimization subtask before the full-video step, link this plan, record Eli's selected exact-resize crop path, and require Task 7's 300-frame gate before the existing 3,044-frame step.

- [x] **Step 3: Update operator documentation**

Document that the first optimized run creates a new FaceCache, old caches remain on disk, gallery caches remain valid, the expected full cold-run range is about 50-60 minutes pending full measurement, and warm relabels remain inference-free. Update `AGENTS.md` current checkpoint to this gate.

- [x] **Step 4: Create the results record structure**

Add environment/versions, exact commands, tests, model-stage timings, face-count/IoU tables, embedding similarity/distance tables, changed assignments, manual review, warm replay, limitations, R1/R2 conclusions, and the explicit STOP decision. Do not fill unmeasured values.

- [x] **Step 5: Check documentation consistency**

Run:

```bash
rg -n "do NOT downscale|stride is the speed lever|schema_version.*1|full 3,044" README.md AGENTS.md docs/design docs/implementation docs/results
git diff --check
```

Expected: no unqualified obsolete claim conflicts with the approved design; diff check exits zero.

- [x] **Step 6: Record the Task 6 documentation state without committing**

Keep the documentation with the implementation for the single conditional milestone commit after Task 7.

---

### Task 7: Run the Complete 300-Frame M4 Acceptance Gate

**Files:**
- Modify: `docs/results/m4-retinaface-black-margin-optimization.md`
- Read only: `/tmp/whiteswan-investigation.HOLoyc/baseline300/faces_*.npz` when present
- Create outside repository: isolated candidate cache, output video, CSV, comparison JSON, and review contact sheet under a fresh `/tmp` directory

**Interfaces:**
- Consumes: the complete 1,262-face M3 baseline and production schema-2 implementation.
- Produces: reproducible detection/embedding parity, timing, identity-change, visual-review, and zero-inference replay evidence.

- [x] **Step 1: Preserve or reconstruct the baseline before candidate execution**

Verify the known isolated baseline has frame indices `0..299`, all statuses `ok`, 1,262 faces, schema 1, and the expected source-video SHA-256. If the temporary baseline is unavailable, reconstruct it with the unchanged `main` implementation in a separate clean worktree/cache before using the optimized branch. Never overwrite the repository cache or schema-1 baseline.

- [x] **Step 2: Run the production candidate over frames 0-299**

Use a fresh temporary cache/output/CSV with stride 1 and batch size 8. Capture process wall/user/system time and the CLI preflight/final summaries. Confirm preflight reports 300 inference frames and the candidate cache metadata reports schema 2 and the approved pipeline/halo.

- [x] **Step 3: Compare candidate cache records against baseline**

Using greedy maximum-IoU one-to-one matching at IoU >= 0.5, write comparison JSON containing exact per-frame counts, baseline/candidate totals, matches/losses/additions, IoU mean/median/p05/p95/minimum, embedding cosine similarity/distance mean/median/p05/p95/p99/extrema, nearest gallery-owner changes, and threshold-0.30 assignment changes. Derive expectations from the two independent caches, not from production mapping helpers.

- [x] **Step 4: Enforce the hard R1 gate**

Expected minimum result: all 1,262 baseline faces matched, zero baseline losses, and no unexplained visible-face regression. Generate a contact sheet for every added/lost detection and manually classify each. Any unexplained visible loss rejects the switch and stops the task.

- [x] **Step 5: Review R2 changes**

Enumerate every changed thresholded identity assignment with frame, face, old/new nearest owner, old/new distance, and crop. Record whether the prototype's five changes reproduce; do not alter threshold or normalization at this gate.

- [x] **Step 6: Run a warm schema-2 replay**

Repeat the same 300-frame window with a separate output/CSV and changed threshold. Require `to_infer=0`, zero detector/Facenet build and forward calls, identical cached boxes/embeddings, and successful relabelling/video publication.

- [x] **Step 7: Record performance against the explicit target**

Report detector build/forward, Facenet build/forward, preprocessing/alignment, FaceCache persistence, video I/O/render, CLI elapsed, and process wall time. The acceptance target is at least 50% lower steady perception wall time than the historical M3 rate without a material R1/R2 or recovery regression.

- [x] **Step 8: Run final verification after evidence publication**

Run:

```bash
.venv/bin/python -m pytest -m "not slow" -v
.venv/bin/python -m pytest -m slow -v
.venv/bin/python -m pip check
.venv/bin/python label_video.py --help
git diff --check
git status --short
```

Expected: all tests pass, dependency/CLI/diff checks pass, only intended source/tests/docs are changed, and no video, cache, model weight, reference photo, CSV, contact sheet, or comparison binary is tracked.

- [x] **Step 9: Report the amended M4 STOP gate**

Report what changed, exact tests, 300-frame counts/IoU/embedding evidence, all identity changes, timings, warm replay, remaining uncertainty, and how the result advances R1/R2. Stop before the full 3,044-frame run and wait for Eli's approval.

- [ ] **Step 10: Commit only if explicitly requested after owner review**

If Eli asks after accepting the gate, rerun the complete fast and slow suites immediately
before committing, stage the implementation, tests, approved docs, and Markdown evidence
only, and use the plain-language summary
`feat: cut video face processing time without losing sampled faces`. Immediately rerun the
same complete suites after the commit and report both results. Never stage private or
generated media, caches, weights, CSV files, JSON comparisons, or contact sheets.
