# Gallery Perception Consistency Implementation Plan

> **Archived completed plan.** Implemented in commit `0c7549a` and merged in PR #2.
> Checkboxes below are historical execution records; current status lives in
> [`docs/STATUS.md`](../../STATUS.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed reference photos through the same optimized RetinaFace alignment and Facenet512 path as video faces, invalidate only incompatible gallery caches, and regenerate the accepted video and CSV from the existing complete FaceCache.

**Architecture:** `face_labeller.perception.embed_faces` remains the single detection, local-alignment, and embedding implementation. Gallery indexing calls it with a copied `Config` whose `max_faces` is exactly one, then converts an empty result into the gallery's existing skip/error behavior. Gallery cache schema 2 includes the perception-pipeline name and detector halo; the independent schema-2 video FaceCache key is unchanged, so the 5,353 cached video embeddings are replayed without RetinaFace or Facenet512 inference.

**Tech Stack:** Python 3.11, DeepFace 0.0.101, RetinaFace 0.0.18, Facenet512, TensorFlow/`tf-keras`, OpenCV, NumPy, pytest, ffmpeg/ffprobe.

**Spec:** `docs/archive/specs/2026-09-26-retinaface-black-margin-optimization-design.md`

## Global Constraints

- Preserve R1: no cap on video faces, stride 1 for the regenerated deliverable, and every cached detection remains boxed.
- Preserve R2's provisional decision rule: Facenet512, cosine distance, base normalization, `all` pins, and strict `distance < 0.30`; do not tune M5/D3 values in this correction.
- Run on CPU only and keep output deterministic for the same inputs, configuration, and installed versions.
- Always pass `model_name="Facenet512"` and `detector_backend="retinaface"` explicitly at the perception boundary.
- Do not change public dataclasses, CLI flags, CSV columns, or FaceCache schema/key.
- Cache one embedding per owner-curated reference photo; never download or commit reference photos, caches, video, CSV, crops, or model weights.
- Reuse all 3,044 successful cached frame analyses. A regenerated run must report `cached=3044`, `to_infer=0`, and `perception=0.000s`.
- Preserve source AAC audio through the one-shot runner and validate video/audio streams before accepting regenerated output.

## Review Focus

- A gallery photo containing no detected face must still be skipped with the existing warning, not converted into a whole-image embedding; Task 1 tests the empty shared-path result.
- A gallery photo containing multiple faces must contribute only the largest detected face; Task 1 tests the forced `max_faces=1` configuration.
- A pre-fix schema-1 gallery cache must not be silently reused; Task 1 tests the schema, pipeline, and halo key changes.
- A gallery-only change must not invalidate or rewrite the video FaceCache identity; Task 1 retains the schema-2 FaceCache golden key and Task 2 proves zero-inference replay.
- Publishing must not replace accepted output or CSV if gallery regeneration, rendering, audio remux, or stream verification fails; Task 2 runs the staged one-shot runner and validates the resulting streams and evidence.

---

### Task 1: Share the Optimized Perception Path With Gallery Indexing

**Files:**
- Modify: `face_labeller/gallery.py`
- Modify: `tests/test_gallery.py`
- Modify: `tests/test_face_cache.py`

**Interfaces:**
- Consumes: `perception.embed_faces(frames: list[np.ndarray], cfg: Config) -> list[list[Face]]` and immutable `Config`.
- Produces: `_embed_gallery_photo(path: Path, cfg: Config) -> np.ndarray` using `dataclasses.replace(cfg, max_faces=1)`; gallery cache schema 2 metadata containing `perception_pipeline` and `detector_black_halo`.

- [x] **Step 1: Write failing shared-path gallery tests**

Add tests proving `_embed_gallery_photo` passes the decoded BGR photo to `perception.embed_faces`, forces `max_faces=1` without mutating the caller's config, returns the one validated embedding, and raises `ValueError("DeepFace returned no usable face")` for an empty result. Update gallery fixtures to fake the perception boundary rather than DeepFace's superseded end-to-end detector path.

- [x] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_gallery.py -k "shared_perception or no_usable_face" -v`

Expected: FAIL because gallery indexing still calls `DeepFace.represent(... detector_backend="retinaface")` directly.

- [x] **Step 3: Implement the shared gallery path**

Import `replace` from `dataclasses`. In `_embed_gallery_photo`, load the image, call `perception.embed_faces([image], replace(cfg, max_faces=1))`, require exactly one returned frame and one face, and return that face's validated embedding. Preserve the current error text so `load_gallery` keeps its warning/skip behavior.

- [x] **Step 4: Run gallery behavior tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_gallery.py -v`

Expected: all gallery tests pass; warm-cache tests prove `perception.embed_faces` is not called for unchanged photos.

- [x] **Step 5: Write failing gallery cache-identity tests**

Change the hand-derived gallery metadata expectation to schema 2 with `perception_pipeline="retinaface_exact_resize_crop_v1"` and `detector_black_halo=32`. Add parameterized assertions that changing either field changes the gallery key while the existing FaceCache schema-2 golden key remains unchanged.

- [x] **Step 6: Run the cache tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_gallery.py tests/test_face_cache.py -k "cache_key or upstream_config" -v`

Expected: FAIL because gallery metadata is schema 1 and omits the shared perception inputs.

- [x] **Step 7: Implement gallery schema 2 metadata**

Set `GALLERY_CACHE_SCHEMA_VERSION = 2` and add `cfg.perception_pipeline` and `cfg.detector_black_halo` to `_gallery_cache_metadata`. Do not alter `FACE_CACHE_SCHEMA_VERSION`, `face_cache_metadata`, or video cache files.

- [x] **Step 8: Verify Task 1**

Run: `.venv/bin/python -m pytest tests/test_gallery.py tests/test_face_cache.py tests/test_perception.py -v`

Expected: all focused tests pass.

---

### Task 2: Regenerate From Cached Video and Publish Evidence

**Files:**
- Modify: `docs/archive/specs/2026-09-26-retinaface-black-margin-optimization-design.md`
- Modify: `docs/design/design-plan.md`
- Modify: `docs/implementation/implementation-plan.md`
- Modify: `README.md`
- Create: `docs/results/m4-gallery-perception-consistency.md`
- Run, ignored: `scripts/run_full_pipeline.sh`, `cache/gallery_*.npz`, `output/nimbus_labelled.mp4`, `output/matches.csv`

**Interfaces:**
- Consumes: the complete schema-2 video FaceCache, the 13 current owner-curated references, and `scripts/run_full_pipeline.sh`.
- Produces: a schema-2 shared-path gallery cache, regenerated unsmoothed video and CSV with preserved AAC audio, comparison evidence, hashes, and updated reproducibility documentation.

- [x] **Step 1: Document the approved consistency contract before regeneration**

Amend the optimization design and main design/implementation documents to state that gallery and video inputs share `perception.embed_faces`, gallery cache schema 2 carries the pipeline/halo identity, and gallery changes never invalidate compatible video perception.

- [x] **Step 2: Preserve the accepted pre-fix CSV outside the repository**

Copy `output/matches.csv` to a fresh exact temporary directory. Validate its 5,353 rows and documented label totals before using it as comparison evidence. Do not stage or commit it.

- [x] **Step 3: Run the one-shot pipeline**

Run: `scripts/run_full_pipeline.sh`

Expected: the gallery is re-embedded once under schema 2; preflight reports `cached=3044 to_infer=0`; video summary reports `perception=0.000s`; ffmpeg restores AAC audio; stream verification succeeds before publication.

- [x] **Step 4: Compare old and regenerated assignments**

Compare the preserved and regenerated CSVs by `(frame_idx, face_idx)`. Record total rows, per-label totals, nearest-owner changes, threshold-assignment changes, transition counts, and the exact changed rows. Confirm boxes, detector confidences, and cached video embeddings were not recomputed or altered.

- [x] **Step 5: Review changed decisions visually**

Create a gitignored contact sheet from the source video for every changed row, grouped by contiguous scene where practical. Record whether each transition is correct character-to/from-Unknown, a likely error, or genuinely ambiguous. Do not silently modify CSV labels.

- [x] **Step 6: Validate regenerated media and evidence**

Use OpenCV and ffprobe to verify 3,044 readable frames, 1920x1080 dimensions, approximately 29.97 fps, duration within one frame of the source, one video stream, one AAC audio stream, and a 12-column/5,353-row CSV. Record SHA-256 hashes and file sizes.

- [x] **Step 7: Publish the evidence report and update user documentation**

Write `docs/results/m4-gallery-perception-consistency.md` with the measured 13-photo embedding shifts, full-cache assignment comparison, visual review, zero-inference proof, timings, media validation, hashes, and the remaining provisional M5 threshold. Update README current counts and link the report; retain earlier M4/M6 reports as explicitly historical evidence.

- [x] **Step 8: Verify Task 2**

Run:

```bash
.venv/bin/python -m pytest -m "not slow" -v
.venv/bin/python -m pytest -m slow -v
.venv/bin/python -m pip check
.venv/bin/python label_video.py --help
git diff --check
git status --short
```

Expected: all tests and checks pass; only intended source/tests/docs are tracked changes; all private and generated artifacts remain ignored.

---

### Task 3: Commit and Update the Pull Request to `main`

**Files:**
- Commit: Task 1 source/tests and Task 2 documentation/evidence
- Modify: `scripts/run_full_pipeline.sh`
- Modify: `tests/test_run_full_pipeline_script.py`
- Modify: `docs/results/m4-full-run.md`
- Exclude: all reference images, caches, output media, CSVs, contact sheets, temporary comparisons, and weights

**Interfaces:**
- Consumes: verified Tasks 1-2 and the existing `investigation/video-runtime` branch/PR.
- Produces: one plain-language correction commit pushed to the existing pull request against `main`.

Final review correction: the one-shot runner's completion message and the old M4 report
must link to the current gallery-consistency evidence rather than presenting the historical
M6 report as current. A runner test asserts the exact new report path.

- [x] **Step 1: Audit the staged set**

Run `git status --short`, `git diff --check`, `git diff --stat`, and `git check-ignore` for the regenerated video, CSV, gallery cache, FaceCache, references, and contact sheet. Stage only the plan, source, tests, tracked documentation, and Markdown evidence.

- [x] **Step 2: Run the complete pre-commit suite**

Run `.venv/bin/python -m pytest -m "not slow" -v` and `.venv/bin/python -m pytest -m slow -v` immediately before committing.

Expected: every test passes.

- [x] **Step 3: Commit the correction**

Commit message: `fix: compare gallery and video faces consistently`

- [x] **Step 4: Run the complete post-commit suite**

Repeat the fast and slow suites immediately after the commit.

Expected: every test passes on the exact committed tree.

- [x] **Step 5: Push and verify the existing PR**

Push `investigation/video-runtime`, verify the existing PR targets `main`, update its title/body if the new consistency evidence is not represented, and confirm required checks are successful. Report the PR URL and leave the branch/workspace in place for review.
