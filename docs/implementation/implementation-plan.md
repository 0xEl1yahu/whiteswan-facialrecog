# Character Face Labeller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic CPU-only pipeline that boxes every detected face in the supplied video and labels it as one of five Harry Potter characters or Unknown, while caching every expensive embedding result for reuse.

**Architecture:** The `label_video.py` executable delegates to a focused `face_labeller/` package whose `core.run` coordinates model-free video preflight, gallery indexing, cached RetinaFace/Facenet512 perception, pure matching, evidence logging, rendering, and video output. Matching, threshold changes, pin-strategy changes, preview stride changes, and optional smoothing replay compatible cached embeddings without model loading or inference. `analyse_matches.py` consumes derived CSV/cache evidence during M5 and never mutates approved defaults.

**Tech Stack:** Python 3.11, DeepFace 0.0.101, RetinaFace 0.0.18, TensorFlow/`tf-keras`, Facenet512, OpenCV, NumPy, pytest, gdown.

**Spec:** `docs/design/design-plan.md`

## Global Constraints

- Work in milestone order. Stop after every task and wait for Eli's approval before continuing.
- Do not commit unless Eli explicitly asks; the conditional commit steps below never grant permission themselves.
- Run on CPU only and produce deterministic output for the same input, configuration, and installed versions.
- Always pass `model_name="Facenet512"` and `detector_backend="retinaface"` explicitly.
- Use cosine distance and L2-normalised `float32` embeddings of shape `(512,)`.
- Never cap faces by default. Unknown faces remain boxed. Generate the final video at stride 1.
- Keep `match`, `cosine_distances`, `draw`, and `iou` pure and free of I/O.
- Never scrape or download reference photos; only validate owner-curated files.
- Do not change the approved data contracts or `label_video.py` CLI without further owner approval.
- Cache one gallery embedding per photo and perception results per absolute frame index. Cache keys contain only upstream embedding inputs, never threshold, pin strategy, stride, frame window, or smoothing.
- Use owner-decided `all` pins (D1). For D2, use batch size 8, stride 3 for quick smoke
  tests, stride 2 for review previews, and stride 1 for final output. D3 selected threshold
  `0.305` with `base` normalization on 2026-09-27.
- Profile the real perception path before starting M4's full 3,044-frame run. Keep the
  current per-frame RetinaFace path as the correctness baseline; an event-driven gate or
  tracker may be proposed from evidence but cannot replace it without owner approval
  because a newly appearing Unknown face could otherwise be missed.
- M6 is forbidden without an explicit owner command. M7 packages whatever has been approved if M6 is skipped.

## File Structure

- `label_video.py` — thin executable and compatibility facade.
- `face_labeller/` — contracts, configuration, perception, caches, gallery, recognition,
  rendering, evidence, video planning/streaming, and the single execution core, as approved
  in `docs/superpowers/specs/2026-09-26-label-video-modularization-design.md`.
- `analyse_matches.py` — M5 evidence generation over `matches.csv`, keyed caches, and source video.
- `requirements.txt` — exact compatible versions recorded at M0 and updated only when an approved milestone adds a dependency.
- `pytest.ini` — declares the `slow` marker.
- `tests/test_config.py` — CLI/default and cache-key boundary tests.
- `tests/test_perception.py` — mocked DeepFace shape, filtering, normalization, clipping, batch fallback, and drawing tests.
- `tests/test_gallery.py` — gallery validation, pin strategies, leave-one-out, incremental cache, and lazy-load tests.
- `tests/test_face_cache.py` — frame status, key, persistence, atomic recovery, and cross-stride reuse tests.
- `tests/test_matching.py` — cosine parity, matching, `nearest_name`, confidence, and CSV tests.
- `tests/test_video_integration.py` — marked-slow short video/gallery pipeline tests.
- `tests/test_analysis.py` — M5 histogram, near-threshold selection, and contact-sheet tests.
- `tests/test_tracker.py` — M6-only IoU and smoothing tests.
- `README.md` and `AGENTS.md` — commands, gate evidence, final rationale, and operating guidance.

## Review Focus

- A batch containing one frame that raises must retain successful frames and mark only the true failure; Task 3 tests this fallback.
- A gallery with a changed, added, deleted, invalid, or uppercase-extension image must reconcile entries without re-embedding unchanged photos; Task 2 tests each case.
- A missing/corrupt/mismatched FaceCache must become a clean miss rather than a silent reuse or crash; Task 3 tests each condition.
- Empty detections, partially out-of-frame boxes, and labels near every image edge must not create fake faces, invalid boxes, or mutate input frames; Task 1 tests these cases.
- An unreadable input, invalid FPS/size, or failed writer must exit non-zero with a clear message and leave no claimed successful output; Task 3 tests these failures.

---

### Task 0: M0 — Reproducible CPU Environment

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `tests/test_environment.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: owner-supplied `data/video-source/nimbus.mp4` and Python 3.11.
- Produces: a working `.venv`, pinned dependencies, warmed model weights, and a repeatable slow smoke test used by later milestones.

- [x] **Step 1: Create and populate the Python 3.11 environment**

Run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install "deepface[tensorflow]" tf-keras opencv-python gdown pytest
```

Expected: installation succeeds under Python 3.11; no GPU package or flag is added.

- [x] **Step 2: Declare the slow-test marker**

Create `pytest.ini` with a `slow` marker whose description says it may load models, download weights, or process real media.

- [x] **Step 3: Write the environment smoke test**

In `tests/test_environment.py`, add `test_cpu_stack_embeds_same_video_frame_identically`. It must open frame 0 of `nimbus.mp4`, call `DeepFace.represent` twice with explicit Facenet512/RetinaFace, `enforce_detection=False`, alignment, base normalization, and L2 normalization, filter confidence-zero placeholders, and assert identical face counts, boxes, and embedding arrays across the two calls.

- [x] **Step 4: Run the smoke test and warm weights**

Run: `.venv/bin/python -m pytest tests/test_environment.py -m slow -v`

Expected: PASS on CPU; Facenet512 and RetinaFace weights exist in DeepFace's normal user cache afterward. Record model-load and two-call timings for the gate report.

- [x] **Step 5: Pin the installed compatible versions**

Record exact versions for DeepFace, RetinaFace, TensorFlow, `tf-keras`, OpenCV, NumPy, gdown, and pytest in `requirements.txt`. Recreate/import-check in a temporary Python 3.11 environment if practical; otherwise run `.venv/bin/python -m pip check` and all M0 tests.

- [x] **Step 6: Update commands and report the M0 STOP gate**

Add exact setup/test commands and version facts to `README.md` and `AGENTS.md`. Report how M0 enables R1/R2, imports, determinism evidence, model-load/embedding timings, installed versions, and open questions. Stop for owner approval.

- [x] **Step 7: Commit only if explicitly requested**

If Eli asks for a commit, stage only the M0 files and use `chore: establish reproducible CPU environment`.

History note: the reproducible environment and pinned stack are present in `main` before
the current checkpoint branch.

---

### Task 1: M1 — Single-Frame Perception and Rendering

**Files:**
- Create: `label_video.py`
- Create: `tests/test_config.py`
- Create: `tests/test_perception.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `numpy.ndarray` BGR frames and the approved CLI/defaults.
- Produces: `Config`, `Face`, `Match`, `Gallery`, and `Track` dataclasses; `load_config(args) -> Config`; `build_models(cfg) -> None`; `embed_faces(frames, cfg) -> list[list[Face]]`; and pure `draw(frame, faces, matches, cfg) -> np.ndarray`.

- [x] **Step 1: Write failing contract and configuration tests**

Add tests that assert every section 7 CLI default, `max_faces is None`, the five canonical character labels, fixed model/detector names, `align is True`, `expand_percentage == 0`, valid enum choices, positive stride/batch size, non-negative frame window, and clear parser failures. Define `Match.nearest_name` exactly as approved.

- [x] **Step 2: Verify the configuration tests fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`

Expected: FAIL because `label_video.py` and its contracts do not exist.

- [x] **Step 3: Implement contracts, constants, and `load_config`**

Implement frozen `Face`, `Match`, and `Gallery` plus mutable `Track`, and a frozen `Config` containing every CLI field and fixed upstream preprocessing input. Keep all tunable defaults named and comment each source from the design spec or DeepFace.

- [x] **Step 4: Verify configuration tests pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`

Expected: PASS.

- [x] **Step 5: Write failing lazy-loader and perception tests**

Mock DeepFace and test that `build_models` is idempotent; `embed_faces` explicitly supplies every approved DeepFace argument; batch size 1's flat response becomes one nested frame result; multi-frame responses preserve frame order; confidence-zero placeholders are dropped; `None` landmarks are removed; boxes clip to frame bounds; embeddings become `float32` unit vectors; empty genuine detections stay empty; and a non-unit or wrong-sized embedding raises clearly.

- [x] **Step 6: Verify perception tests fail**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k "model or embed" -v`

Expected: FAIL on missing behavior.

- [x] **Step 7: Implement lazy `build_models` and `embed_faces`**

Call `DeepFace.build_model("Facenet512")` only when this function is invoked by an uncached path, and at most once per process. Implement the approved represent call and result normalization without I/O inside any pure function.

- [x] **Step 8: Verify perception tests pass**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k "model or embed" -v`

Expected: PASS.

- [x] **Step 9: Write failing pure-render tests**

Test output shape/dtype, non-mutation, deterministic character colors, grey Unknown boxes, filled labels kept inside all four edges, zero-face frames, clipped/zero-area boxes, and five landmark dots only when `debug_landmarks` is enabled. Use synthetic `Match` values; M1 does not require a gallery.

- [x] **Step 10: Implement `draw` and pass the render tests**

Run: `.venv/bin/python -m pytest tests/test_perception.py -k draw -v`

Expected: PASS after implementing `draw(frame, faces, matches, cfg) -> np.ndarray` as a pure copy-returning function.

- [x] **Step 11: Produce and inspect the M1 artifact**

Run the real M0 frame through `embed_faces`, construct Unknown matches, render landmarks, and save `output/m1_single_frame.png`. Also run a selected face-less frame and confirm zero boxes. Record model time, perception time, face count, and visual observations.

- [x] **Step 12: Run M1 verification and report the STOP gate**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_perception.py -v`

Expected: PASS. Report how the boxes advance R1 and Unknown rendering advances R2, plus timings and open questions. Stop for owner approval.

- [x] **Step 13: Commit only if explicitly requested**

If Eli asks, commit the M1 files as `feat: add single-frame face perception and rendering`.

History note: M1 is commit `c3794c2`.

---

### Task 2: M2 — Incremental Gallery and Pin Strategies

**Files:**
- Modify: `label_video.py`
- Create: `tests/test_gallery.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: canonical `ref_dir/<Character Name>/*.{jpg,jpeg,png}` photos, `Config`, installed-version metadata, and lazy `build_models`/embedding behavior from Task 1.
- Produces: internal `GalleryPhoto(owner, source_path, file_hash, embedding)` records; `gallery_cache_key(cfg, versions) -> str`; `build_pins(records, strategy) -> Gallery`; `load_gallery(ref_dir, cfg) -> Gallery`; and `leave_one_out_report(records, threshold) -> dict`.

- [x] **Step 1: Write failing gallery validation and pin tests**

Test all five required folders, case-insensitive supported extensions, deterministic path ordering, minimum two valid images per character, warning/skip behavior for an image with no face, failure after skips leave fewer than two, mean-pin re-normalization, all-pin ownership, sorted unique names, and identical per-photo inputs for both strategies.

- [x] **Step 2: Verify the validation/pin tests fail**

Run: `.venv/bin/python -m pytest tests/test_gallery.py -k "validation or pins" -v`

Expected: FAIL.

- [x] **Step 3: Implement gallery discovery, embedding, and pin construction**

Use explicit model/detector/normalization/alignment, `max_faces=1`, `enforce_detection=True`, and L2 normalization. Call `build_models` only immediately before the first cache miss is embedded. Print images found/used/skipped per character.

- [x] **Step 4: Verify validation/pin tests pass**

Run: `.venv/bin/python -m pytest tests/test_gallery.py -k "validation or pins" -v`

Expected: PASS.

- [x] **Step 5: Write failing configuration-keyed cache tests**

Test that the key includes model, detector, normalization, align, `max_faces=1`, expansion, L2 setting, and DeepFace/RetinaFace/TensorFlow/OpenCV versions. Assert photo hashes and pin strategy are not in the configuration key. Test cold build, warm build with zero model loads, one added photo, one modified photo, one deleted photo, normalization switch, and switching back to reuse the earlier variant. Include corrupt/mismatched metadata as a clean miss.

- [x] **Step 6: Implement `cache/gallery_{key}.npz` reconciliation**

Store embeddings, owner, normalized relative source path, per-photo SHA-256, and complete metadata. Write atomically with a temporary file in the same cache directory and `Path.replace`. Never re-embed an unchanged entry.

- [x] **Step 7: Run all gallery cache tests**

Run: `.venv/bin/python -m pytest tests/test_gallery.py -v`

Expected: PASS with mocked model calls proving exact recomputation counts.

- [x] **Step 8: Add and run leave-one-out evidence**

Test `leave_one_out_report` with two synthetic photos per identity and both strategies, then run it over the real two-image-per-character gallery. The held-out photo must never remain among its candidate pins. Record per-character nearest identity, distance, assignment, and aggregate correct/Unknown/wrong counts for mean and all.

- [x] **Step 9: Demonstrate incremental real-cache reuse**

Run the real gallery twice, proving the second run does not load Facenet512. Add one owner-provided photo if available and prove exactly one embed; otherwise demonstrate with a copied test fixture without altering curated data. Prove deletion removes only the corresponding entry.

- [x] **Step 10: Report the M2 STOP gate**

Report how gallery identity pins advance R2, tests, first/warm/incremental timings, per-character counts, leave-one-out results under both strategies, and D1 for Eli. Stop before M3.

- [x] **Step 11: Commit only if explicitly requested**

If Eli asks, commit the M2 files as `feat: add incremental reference gallery`.

History note: M2 is commit `f8faf96`.

---

### Task 3: M3 — Per-Frame Cache and Video Box Pipeline

**Files:**
- Modify: `label_video.py`
- Create: `tests/test_face_cache.py`
- Create: `tests/test_video_integration.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Config`, `Face`, `embed_faces`, absolute video frame indices, and installed-version metadata.
- Produces: `face_cache_key(video_path, cfg, versions) -> str`; `FaceCache.status/get/put_ok/put_failed/missing/flush`; `selected_frame_indices(start, stop, stride) -> list[int]`; `process_batch_with_fallback(indexed_frames, cfg) -> dict[int, tuple[str, list[Face]]]`; and `process_video(cfg, gallery: Gallery | None = None) -> RunSummary`.

- [x] **Step 1: Write failing FaceCache key/status tests**

Assert video SHA-256, upstream perception inputs, and installed versions affect the key; threshold, pin strategy, stride, frame window, smoothing, drawing, CSV, and output path do not. Test explicit absent/ok/failed statuses, including an ok empty-face frame and retry of failed status.

- [x] **Step 2: Verify the FaceCache tests fail**

Run: `.venv/bin/python -m pytest tests/test_face_cache.py -k "key or status" -v`

Expected: FAIL.

- [x] **Step 3: Implement FaceCache metadata and atomic persistence**

Serialize boxes, landmarks, confidences, embeddings, and statuses without changing the public `Face` contract. Validate full metadata on load. Treat missing, truncated/corrupt, or mismatched files as clean misses with warnings. Flush each completed batch atomically so interruption loses at most the active batch.

- [x] **Step 4: Pass key/status/persistence tests**

Run: `.venv/bin/python -m pytest tests/test_face_cache.py -v`

Expected: PASS.

- [x] **Step 5: Write failing selection/reuse/fallback tests**

Assert the requested window's first frame is always selected; later selections are relative to it; a stride-3 cache followed by stride 1 requests only missing absolute indices; repeat runs request none; `--cache-dir` redirects both cache families; `--no-cache` bypasses all cache reads/writes; a batch exception triggers individual retries; successful frames become ok; the true exception becomes failed; and failed frames retry next run.

- [x] **Step 6: Implement selection and batch fallback**

Keep `embed_faces`' approved return type. Catch a batch exception in `process_batch_with_fallback`, retry each frame via `embed_faces([frame], cfg)`, and return explicit statuses for FaceCache persistence.

- [x] **Step 7: Pass reuse/fallback tests**

Run: `.venv/bin/python -m pytest tests/test_face_cache.py tests/test_perception.py -v`

Expected: PASS.

- [x] **Step 8: Write failing video I/O integration tests**

Generate a deterministic ten-frame fixture video. Test unreadable input, invalid FPS/size metadata, writer-open failure, same output frame count/FPS/size, every frame written, Unknown boxes drawn from cached faces, stride reuse, and continuation after one failed frame. Patch perception for fast tests; mark the real-model path slow.

- [x] **Step 9: Implement reader/batcher/writer and progress reporting**

Use `cv2.VideoCapture`, `cv2.VideoWriter` with `mp4v`, absolute frame indices, bounded pending frames, and a `RunSummary` containing elapsed/model/perception times, processed/written frames, faces, cache hits/misses/failures, FPS, and label distribution. Validate writer state before processing. Create output directories safely.

- [x] **Step 10: Pass video integration tests**

Run: `.venv/bin/python -m pytest tests/test_video_integration.py -m "not slow" -v`

Expected: PASS.

- [x] **Step 11: Collect M3 timing and cache evidence**

Run the first 300 frames at stride 3, repeat it warm, then run stride 1 over the same window. Record model load, wall time, processed FPS, faces, cache hit/miss counts, and prove the stride-1 run computes only the missing indices. Separately time a cold stride-1 window for a fair stride comparison.

Execution note (2026-09-26): the cold stride-3 run plus its stride-1 continuation
populated every frame exactly once. A second cold stride-1 run was intentionally omitted
because it would recompute all 300 completed frames under the strict deadline; the combined
timing is retained as a conservative comparison with that limitation stated.

- [x] **Step 12: Report the M3 STOP gate**

Report how per-frame detection and Unknown boxes advance R1, output frame/duration checks, failure behavior, timings, reuse evidence, and D2 batch-size/preview-stride observations. Reaffirm final stride 1. Stop before M4.

- [x] **Step 13: Commit only if explicitly requested**

If Eli asks, commit the M3 files as `feat: add resumable video face cache`.

History note: M3 is commit `f8c9200`.

---

### Task 4: M4 — Matching, CSV Evidence, and Complete Vertical Slice

**Files:**
- Modify: `label_video.py`
- Create: `tests/test_matching.py`
- Modify: `tests/test_video_integration.py`
- Create: `docs/results/m4-perception-profile.md`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: `Face`, `Gallery`, `Config`, cached perception, and DeepFace confidence utilities.
- Produces: pure `cosine_distances(v, pins) -> np.ndarray`; pure `match(face, gallery, threshold) -> Match`; `MatchLogger.log(frame_idx, face_idx, face, match, threshold, frame=None) -> None`; complete `main() -> int`; output video; and `matches.csv`.

- [x] **Step 1: Write failing cosine and match tests**

Assert NumPy cosine distances agree with DeepFace `verification.find_distance` to `1e-5`; nearest pin wins deterministically; exact-threshold distance is Unknown because the rule is `<`; below-threshold assigns the owner; above-threshold keeps `nearest_name` while `name is None`; and confidence calls `find_confidence(distance, "Facenet512", verified, "cosine")` with the correct verified boolean.

- [x] **Step 2: Verify matching tests fail**

Run: `.venv/bin/python -m pytest tests/test_matching.py -k "cosine or match" -v`

Expected: FAIL.

- [x] **Step 3: Implement pure distance and matching functions**

Use `1 - pins @ v`, preserve gallery order for deterministic ties, and never read configuration or perform I/O inside either function.

- [x] **Step 4: Pass cosine and match tests**

Run: `.venv/bin/python -m pytest tests/test_matching.py -k "cosine or match" -v`

Expected: PASS.

- [x] **Step 5: Write failing MatchLogger tests**

Assert the exact approved column order, one row per face, stable face indices, numeric serialization, `nearest_name` retained for Unknown, `assigned_name` rendered as `Unknown`, a fresh header and overwrite (`"w"`) mode for each run, closed/flushed files, and clipped debug crops named from frame/face/nearest/distance without mutating frames.

- [x] **Step 6: Implement MatchLogger and pass its tests**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`

Expected: PASS. The logger consumes `Match.nearest_name`; it must never recalculate distances.

- [x] **Step 7: Write failing vertical-slice integration tests**

Test gallery load, cached perception, matching, labels, CSV rows, output frame count, and non-zero exit on missing input/ref/output failures. Add a fully cached threshold-change test that monkeypatches `DeepFace.build_model` and `DeepFace.represent` to fail if called, then asserts successful relabelling and changed assignments.

- [x] **Step 8: Integrate gallery, matcher, logger, renderer, and CLI**

Make `main()` validate paths, load compatible caches before deciding whether model construction is needed, process all faces without a default cap, and return a process exit code. Every DeepFace call must retain explicit model/detector parameters.

- [x] **Step 9: Run the complete automated suite**

Run:

```bash
.venv/bin/python -m pytest -m "not slow" -v
.venv/bin/python -m pytest -m slow -v
```

Expected: all tests pass; slow tests may use the warmed model and short media fixtures.

- [x] **Step 10: Profile the underlying perception processes before the full run**

Use `cProfile` around the existing CLI on frames 600–629, 1500–1529, and 2700–2729.
These three 30-frame windows sample different thirds of the clip and sit outside the
completed first 300 frames. Use `cache/m3` so all 90 newly profiled frames remain reusable.
For each start frame, run this command with matching zero-padded filenames:

```bash
.venv/bin/python -m cProfile -o output/m4_profile_0600.pstats label_video.py \
  --input data/video-source/nimbus.mp4 --output output/m4_profile_0600.mp4 \
  --ref-dir data/reference-images --start-frame 600 --max-frames 30 \
  --stride 1 --batch-size 8 --cache-dir cache/m3
```

Repeat the same command without changing the cache, replacing the profile and video names
with `output/m4_profile_0600_warm.pstats` and `output/m4_profile_0600_warm.mp4`. Then print
the top 50 cumulative call stacks non-interactively with:

```bash
.venv/bin/python -c 'import pstats; pstats.Stats("output/m4_profile_0600.pstats").strip_dirs().sort_stats("cumulative").print_stats(50)'
```

Repeat both cold/warm commands and the report command with `0600`/`600` replaced by
`1500`/`1500` and `2700`/`2700` respectively.

Record the exact commands in `docs/results/m4-perception-profile.md`, separating model
load, video I/O/rendering, RetinaFace detection, Facenet512 embedding, face count, and
wall time as far as the installed DeepFace call stack permits. The three cold profile
processes each pay model startup, so report it separately rather than projecting it per
window into the single-process full run. Warm repeats must prove the neural calls
disappear. Do not add a profiling CLI or restructure production code solely for this
measurement.

- [x] **Step 11: Report the runtime finding and select the smallest safe path**

Extrapolate stride-1 and stride-2 runtime from all measured windows, state the uncertainty,
and rank the actual hotspots. STOP before the full-video run and ask Eli to choose one of:
(a) continue with the correct per-frame baseline; (b) amend this plan with a measured,
R1-preserving optimization such as better embedding batching or safe CPU thread tuning;
or (c) amend the spec and plan for periodic RetinaFace, a cheap face/activity gate, or
tracking between detections. For option (c), report expected missed-face latency and false-
negative risk because it weakens the requirement to attempt detection on every final frame.
The heavy models must remain resident once loaded; an idle gate must not repeatedly pay the
measured model-startup cost. Do not implement either optimization path until its tests,
acceptance criteria, and owner approval are added to this plan.

- [x] **Step 12: Complete the approved modularization and preflight checkpoint**

Execute `docs/superpowers/plans/2026-09-26-label-video-modularization.md` through its STOP
gate. Preserve the CLI, approved data contracts, cache keys/schemas, CSV schema, output
behavior, lazy model loading, and zero-inference replay. Do not begin the full 3,044-frame
run until Eli accepts the checkpoint report.

Execution note (2026-09-26): the modular package, thin compatibility facade, model-free
preflight, atomic outputs, cache compatibility, and cold/warm parity were completed and
merged to `main` in PR #1 before the runtime optimization work began.

- [x] **Step 13: Generate and validate the full stride-1 vertical slice**

First execute the approved exact-resize black-margin optimization plan at
`docs/superpowers/plans/2026-09-26-retinaface-black-margin-optimization.md`. Its complete
300-frame comparison against the 1,262-face M3 baseline is a mandatory STOP gate. Eli must
accept its detection/embedding/identity/runtime evidence before this step continues.

After that approval, run the approved CLI over all 3,044 frames at stride 1. Verify output
frame count, FPS, dimensions, duration tolerance of one frame, CSV schema/row count, cache
completion, and end-of-run label distribution. Manually inspect representative crowd,
wide, motion, profile, and scene-cut frames for boxes and labels.

Execution note (2026-09-26): the optimized run completed all 3,044 frames at stride 1 in
33m03s, wrote 5,353 evidence rows, and recorded zero failed frames. Output metadata matches
the source. Named-interval and crowd-frame review, including four visible Ron
misassignments and the distant-crowd recall limitation, is recorded in
`docs/results/m4-full-run.md`.

- [x] **Step 14: Prove zero-inference relabelling**

Run again with a different threshold and separate output/CSV paths. Capture logs/tests proving no Facenet512 load, RetinaFace call, or Facenet512 inference occurred and that cached detections/embeddings were reused.

Execution note (2026-09-26): threshold, gallery, and smoothing replays reported all 3,044
frames cached, zero frames to infer, `model=0.000s`, and `perception=0.000s`. Integration
tests fail explicitly if either model or perception is invoked during the warm replay.

Consistency correction (owner-approved 2026-09-27): execute
`docs/superpowers/plans/2026-09-27-gallery-perception-consistency.md` before M5. Reference
photos must share the optimized `embed_faces` path with video faces. Gallery cache schema 2
adds the perception pipeline and detector halo, while the complete video FaceCache remains
compatible and must replay with zero video inference. Regenerate the unsmoothed video/CSV,
preserve AAC audio, enumerate and inspect changed threshold decisions, and publish the
evidence without changing the provisional threshold or normalization.

- [x] **Step 15: Report the M4 STOP gate**

Report how the stride-1 video and Unknown handling satisfy R1/R2, all test results, full/warm timings, output validation, label distribution, visual findings, and open questions. Stop before tuning.

Execution note (2026-09-26): the gate report is `docs/results/m4-full-run.md`; the owner
reviewed the Harry discrepancy evidence and subsequently supplied additional references.

- [x] **Step 16: Commit only if explicitly requested**

If Eli asks, commit the M4 files as `feat: complete cached face labelling pipeline`.

Checkpoint note: Eli requested commits and a PR to `main`; the runtime optimization,
full-run evidence, and tests are committed as a distinct M4 checkpoint in this branch.

---

### Task 5: M5 — Evidence-Based Threshold and Normalization Analysis

**Files:**
- Create: `analyse_matches.py`
- Create: `tests/test_analysis.py`
- Modify: `README.md`
- Create during runs: `output/analysis/` artifacts and `output/tuning-report.md` (gitignored unless owner requests otherwise)

**Interfaces:**
- Consumes: source video, `matches.csv`, Gallery/FaceCache metadata, and separate base/Facenet2018 sample-run CSVs.
- Produces: `load_match_rows(path)`, `build_distance_histograms(rows)`, `select_near_threshold(rows, threshold, margin, limit)`, `build_contact_sheet(video, rows, output, cell_size, columns)`, comparison summaries, and a written recommendation without changing defaults.

- [x] **Step 1: Write failing pure-analysis tests**

Test strict CSV schema validation, empty input, grouping by `nearest_name`, fixed deterministic histogram bins supplied by configuration, stable near-threshold ordering by absolute margin then frame/face index, per-character limits, and summary counts for assigned/Unknown results.

- [x] **Step 2: Verify analysis tests fail**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`

Expected: FAIL.

- [x] **Step 3: Implement CSV analysis and report data generation**

Keep tunables as analysis CLI arguments with documented defaults. Use the standard library, NumPy, and OpenCV; do not add a dataframe dependency unless owner-approved evidence shows it is necessary.

- [x] **Step 4: Write failing contact-sheet tests**

Use a tiny fixture video and rows containing valid, clipped, zero-area, missing-frame, and repeated-frame crops. Assert deterministic layout, labels, dimensions, safe skipping, and no model imports/calls.

- [x] **Step 5: Implement contact-sheet extraction and pass tests**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`

Expected: PASS.

- [x] **Step 6: Generate baseline evidence**

Read the M4 CSV and compatible cache metadata to produce distance histograms by nearest character, label counts, and a near-threshold contact sheet. Record obvious false positives, false negatives, uncertain crops, and detector misses separately.

- [x] **Step 7: Run normalization A/B on the same deterministic sample**

Select sample frame indices once and persist them in the analysis report. Run base and Facenet2018 on exactly those frames and the full gallery, creating separate configuration-keyed gallery/FaceCaches and CSVs. Repeat both runs warm to prove each normalization variant is reused rather than overwritten.

- [x] **Step 8: Write the recommendation without changing defaults**

Create `output/tuning-report.md` containing sample selection, versions, cache keys, threshold evidence, normalization comparison, errors observed, timings, limitations from two photos per character, and a recommendation for D3. Do not modify threshold or normalization defaults.

Execution note (2026-09-27): the completed report recommends retaining threshold `0.30`
and normalization `base`. The 190-crop threshold review contains no confirmed true extras,
so it does not support raising the threshold. Facenet2018 kept identical geometry for all
767 sampled detections but lost 72 correct labels and gained 14, a net loss of 58. The
candidate cold run took 331.746 seconds and its warm replay took 3.670 seconds. Production
defaults and the accepted output remain unchanged pending D3.

Follow-up evidence (2026-09-27): an isolated full-cache A/B replay compared `0.30` with
`0.31`. All 5,353 upstream rows were identical and 111 assignments changed from Unknown;
visual review found 110 correct new names and one new Harry-to-Ron error. The intermediate
`0.305` threshold accepts 62 correct names without that observed error. The audio-preserved
candidate and exact evidence are documented in `docs/results/m5-threshold-ab.md`. Eli chose
`0.305` with `base`; the production default and one-shot runner now record that D3 decision.

- [x] **Step 9: Report the M5 STOP gate**

Report how evidence improves R2 without sacrificing Unknown coverage under R1, tests, artifacts, timings, recommendation, and open questions. Stop all work until Eli decides D3 and either commands M6 or skips to M7.

- [ ] **Step 10: Commit only if explicitly requested**

If Eli asks, commit source/tests/docs only as `feat: add evidence-based match analysis`; do not commit private images, caches, or generated video.

---

### Task 6: M6 — Optional Temporal Smoothing and Delivery Encoding

> Do not start this task unless Eli explicitly commands M6 after the M5 gate.

Execution ruling (2026-09-26): Eli explicitly authorized M6 while independently sourcing
new gallery images. M5/D3 was open at that point, so M6 did not change the then-provisional
threshold or normalization. Full-clip evidence kept smoothing opt-in because lifetime voting reduced
named coverage, while automatic audio restoration was accepted for the one-shot runner.

**Files:**
- Create: `face_labeller/tracking.py`
- Modify: `face_labeller/video.py`, `label_video.py`, `scripts/run_full_pipeline.sh`
- Create: `tests/test_tracker.py`
- Modify: `tests/test_video_integration.py`, `tests/test_run_full_pipeline_script.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: per-frame `Face`/`Match` values, `Config.iou_min`, `Config.track_ttl`, and the unsmoothed cached pipeline.
- Produces: pure `iou(a, b) -> float`; `Tracker.update(frame_idx, faces, matches) -> list[tuple[Face, Match, int]]`; `--smooth` rendering; and documented ffmpeg audio/H.264 delivery commands.

- [x] **Step 1: Write failing IoU tests**

Test identical, disjoint, edge-touching, zero-area, and one known-overlap pair with exact expected values.

- [x] **Step 2: Implement pure IoU and pass tests**

Run: `.venv/bin/python -m pytest tests/test_tracker.py -k iou -v`

Expected: PASS.

- [x] **Step 3: Write failing tracker tests**

Test deterministic greedy association at/under the IoU boundary, multiple faces, new IDs, expiry after exactly `track_ttl` unseen frames, majority vote, Unknown tie winner, no mutation of input faces/matches, and deterministic track ordering.

- [x] **Step 4: Implement Tracker and pass tests**

Run: `.venv/bin/python -m pytest tests/test_tracker.py -v`

Expected: PASS.

- [x] **Step 5: Integrate smoothing as a downstream cache replay**

Assert with a regression test that toggling `--smooth`, `iou_min`, or `track_ttl` does not change Gallery/FaceCache keys or call models. Preserve unsmoothed CSV evidence unless the spec explicitly requires displayed-name logging.

- [x] **Step 6: Add and verify ffmpeg delivery**

The owner additionally approved automatic audio restoration in the one-shot runner. Stage
the video and CSV, combine the labelled video with input AAC without re-encoding, verify
both streams, and publish only after success. Document the separate H.264 browser-delivery
command. Verify streams and duration with `ffprobe`.

- [x] **Step 7: Produce before/after evidence and report M6**

Render the same short cache-backed clip with and without smoothing, record replay time and visible flicker differences, run the full non-slow suite, and report results. Stop for owner review.

- [x] **Step 8: Commit only if explicitly requested**

If Eli asks, commit as `feat: add optional temporal smoothing and delivery guidance`.

Checkpoint note: Eli requested commits and a PR to `main`; M6 remains a distinct commit
from the M4 runtime optimization.

---

### Task 7: M7 — Packaging and Reproducibility

> Checkpoint status (updated 2026-09-27): not started as the final delivery gate. M5/D3 is
> resolved at threshold `0.305` with `base` normalization. M7 still must reconcile and
> verify the final deliverables before claiming final delivery.

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `AGENTS.md`
- Verify: `.gitignore`
- Verify generated, gitignored deliverables: `output/nimbus_labelled.mp4`, `output/matches.csv`

**Interfaces:**
- Consumes: every owner-approved milestone and D1-D3 decision.
- Produces: reproducible documentation, clean tracked source, pinned dependencies, verified final video/CSV, and a final gate report.

- [ ] **Step 1: Reconcile final approved decisions and commands**

Record D1-D3, final stride 1, chosen batch size, threshold, normalization, optional M6 status, exact input/output commands, and cache-replay commands. Remove provisional wording only where Eli has decided.

- [ ] **Step 2: Complete README requirements**

Cover setup, video download, two-image minimum and incremental gallery growth, run commands, cache keys/reuse, architecture, design rationale, runtime/label results, known failures, privacy/gitignore behavior, audio/H.264 status, and what to do with more time.

- [ ] **Step 3: Verify reproducibility from pinned dependencies**

Run `.venv/bin/python -m pip check`, imports, all non-slow tests, approved slow tests, and `python label_video.py --help`. If practical, install `requirements.txt` into a temporary Python 3.11 environment and rerun imports/non-model unit tests.

- [ ] **Step 4: Verify final deliverables**

Programmatically compare input/final-output frame count, FPS, dimensions, and duration; validate every CSV row and required column; confirm the final run used stride 1; record SHA-256 hashes and file sizes; and manually spot-check representative frames.

- [ ] **Step 5: Audit repository cleanliness and exclusions**

Run `git status --short`, `git diff --check`, and `git check-ignore` for the video, reference images, weights if local, caches, debug crops, CSV, and output video. Ensure no private/generated binary is staged or tracked.

- [ ] **Step 6: Run the final verification suite**

Run:

```bash
.venv/bin/python -m pytest -m "not slow" -v
.venv/bin/python -m pytest -m slow -v
```

Expected: all applicable approved tests pass. Report exact counts, skips, durations, final runtime, label distribution, known limitations, and how the deliverables satisfy R1/R2.

- [ ] **Step 7: Commit only if explicitly requested**

If Eli asks, create the requested packaging commit without adding any ignored/private artifact.
