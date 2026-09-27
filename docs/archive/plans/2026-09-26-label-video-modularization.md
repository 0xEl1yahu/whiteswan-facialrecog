# Label Video Modularization Implementation Plan

> **Archived completed plan.** Implemented and merged in commit `79431f5` (PR #1).
> Checkboxes below are historical execution records; current status lives in
> [`docs/STATUS.md`](../../STATUS.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `label_video.py` into focused `face_labeller` modules, establish `core.run` as the single execution coordinator, and add model-free video/frame preflight without changing approved CLI, data, cache, or output behavior.

**Architecture:** Keep `label_video.py` as a thin executable and compatibility facade over an acyclic `face_labeller/` package. Move leaf responsibilities first, add tested video metadata/frame planning before expensive work, then cut streaming and orchestration over to `video.py` and `core.py` after their dependencies are stable.

**Tech Stack:** Python 3.11, dataclasses, argparse, pathlib, OpenCV, NumPy, DeepFace 0.0.101, RetinaFace 0.0.18, Facenet512, pytest 9.1.1.

**Spec:** `docs/archive/specs/2026-09-26-label-video-modularization-design.md`

## Global Constraints

- This is an M4 refactor checkpoint. Do not begin the pending full 3,044-frame run, M5, or M6.
- Keep `python label_video.py ...` and every approved CLI flag/default unchanged.
- Keep existing `Face`, `Match`, `Gallery`, `Config`, and `RunSummary` fields unchanged.
- Always pass `model_name="Facenet512"` and `detector_backend="retinaface"` explicitly.
- Run on CPU only. Do not add GPU code, flags, dependencies, or benchmarks.
- Preserve R1: no default face cap, Unknown faces stay boxed, every output frame is written, and final stride remains 1.
- Preserve R2: strict `< threshold` matching assigns a character; all other detections are labelled `Unknown`.
- Keep `match`, `cosine_distances`, and `draw` pure and free of I/O.
- Preserve gallery/FaceCache keys, filenames, NPZ schemas, schema versions, and old-cache readability.
- A fully cached gallery and frame replay must not call `DeepFace.build_model` or `DeepFace.represent`.
- Never scrape/download reference images or commit videos, photos, model weights, caches, CSVs, crops, or rendered outputs.
- Do not commit per task. If Eli explicitly requests a commit after the final STOP gate, make one M4 refactor commit containing implementation and tests, and run the complete relevant suite immediately before and after it.

## File Structure

- Modify: `label_video.py` — thin executable and compatibility re-exports only.
- Create: `face_labeller/__init__.py` — package marker; no broad wildcard API.
- Create: `face_labeller/contracts.py` — records and fixed domain constants.
- Create: `face_labeller/config.py` — `Config`, defaults, parser, and argument validation.
- Create: `face_labeller/perception.py` — lazy DeepFace lifecycle, face conversion, embedding, and fallback.
- Create: `face_labeller/cache.py` — stable hashing/version helpers and `FaceCache`.
- Create: `face_labeller/gallery.py` — gallery discovery, cache reconciliation, pins, and diagnostics.
- Create: `face_labeller/recognition.py` — pure distance and matching.
- Create: `face_labeller/rendering.py` — pure deterministic drawing.
- Create: `face_labeller/evidence.py` — CSV logger and debug crops.
- Create: `face_labeller/video.py` — metadata inspection, frame planning, and streaming execution.
- Create: `face_labeller/core.py` — end-to-end coordination and run reporting.
- Create: `tests/test_facade.py` — compatibility/export/delegation tests.
- Create: `tests/test_video_plan.py` — preflight and frame-plan tests.
- Modify: existing tests — import and monkeypatch the module that owns each behavior.
- Modify: `README.md`, `AGENTS.md` — final architecture, commands, and ownership guidance.
- Already amended: authoritative design/implementation documents and approved spec.

## Review Focus

- A valid pre-refactor gallery or FaceCache must load after extraction with the same key and no migration; Tasks 4 and 5 pin golden keys and serialized fields.
- Invalid video metadata or a start frame outside the video must fail before gallery/model calls; Task 2 tests call ordering explicitly.
- A replaced video between preflight and streaming must fail if FPS, dimensions, or frame count differ; Task 6 tests stale-plan rejection.
- Cached `ok` empty-face frames are hits while `failed` and absent selected frames remain pending; Task 2 tests all three states.
- Compatibility exports must not tempt tests to patch ineffective facade globals; Tasks 1 and 3-7 move patches to owning modules and keep a focused facade test.

---

### Task 1: Package Spine, Contracts, Configuration, and Facade

**Files:**
- Create: `face_labeller/__init__.py`
- Create: `face_labeller/contracts.py`
- Create: `face_labeller/config.py`
- Create: `tests/test_facade.py`
- Modify: `label_video.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: existing constants, dataclasses, parser validators, and CLI defaults from `label_video.py`.
- Produces: unchanged `Face`, `Match`, `Gallery`, `GalleryPhoto`, `Track`, `Config`, and `RunSummary`; unchanged `build_parser()`, `parse_args(argv)`, and `load_config(args)`; compatibility imports from `label_video`.

- [x] **Step 1: Write facade and ownership characterization tests**

Add `tests/test_facade.py::test_label_video_exports_supported_api` asserting that these names resolve from `label_video`: `CHARACTER_NAMES`, `MODEL_NAME`, `DETECTOR_BACKEND`, `BOX_COLORS`, `LANDMARK_COLOR`, `Config`, `Face`, `Match`, `Gallery`, `GalleryPhoto`, `Track`, `RunSummary`, `FaceCache`, `MatchLogger`, `build_parser`, `parse_args`, `load_config`, `build_models`, `embed_faces`, `gallery_cache_key`, `face_cache_key`, `build_pins`, `load_gallery`, `leave_one_out_report`, `selected_frame_indices`, `process_batch_with_fallback`, `cosine_distances`, `match`, `draw`, `process_video`, and `main`.

Add `test_approved_contract_fields_are_unchanged` asserting exact dataclass field order:

```python
assert tuple(Face.__dataclass_fields__) == ("box", "landmarks", "embedding", "det_conf")
assert tuple(Match.__dataclass_fields__) == ("nearest_name", "name", "distance", "confidence")
assert tuple(Gallery.__dataclass_fields__) == ("pins", "pin_owner", "names", "meta")
assert tuple(RunSummary.__dataclass_fields__) == (
    "elapsed_seconds", "model_seconds", "perception_seconds",
    "processed_frames", "written_frames", "faces", "cache_hits",
    "cache_misses", "failures", "processing_fps", "fps", "width",
    "height", "label_distribution",
)
```

- [x] **Step 2: Run the characterization tests before moving code**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_facade.py -v`

Expected: existing configuration tests PASS; new facade tests PASS against the monolith and establish the surface to preserve.

- [x] **Step 3: Create the package and move contracts/configuration**

Move fixed domain constants and records to `contracts.py`. Move CLI defaults, `Config`, validators, parser creation, parsing, and config construction to `config.py`. Preserve default-source comments and type annotations.

In `label_video.py`, import and re-export the moved names explicitly. Do not use `from face_labeller import *`, dynamic `__getattr__`, or module alias tricks.

- [x] **Step 4: Move configuration tests to their owning modules**

Update `tests/test_config.py` to import records from `face_labeller.contracts` and configuration behavior from `face_labeller.config`. Keep `tests/test_facade.py` as the only test whose purpose is broad compatibility through `label_video`.

- [x] **Step 5: Verify the first runnable slice**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_facade.py -v
.venv/bin/python -m pytest -m "not slow" -q
.venv/bin/python label_video.py --help
```

Expected: 119 or more fast tests pass, 3 slow tests remain deselected, and help text retains every existing option/default.

---

### Task 2: Video Metadata, Frame Planning, and Early Preflight

**Files:**
- Modify: `face_labeller/contracts.py`
- Create: `face_labeller/video.py`
- Create: `tests/test_video_plan.py`
- Modify: `label_video.py`
- Modify: `tests/test_video_integration.py`

**Interfaces:**
- Consumes: `Config`, input path, OpenCV capture metadata, `FaceCache.status/missing`, and current `selected_frame_indices` behavior.
- Produces: `VideoMetadata`, `FramePlan`, `inspect_video`, `build_frame_plan`, deterministic preflight output, and optional precomputed objects accepted by `process_video`.

- [x] **Step 1: Write failing metadata and pure-plan tests**

Define exact records in the test imports:

```python
@dataclass(frozen=True)
class VideoMetadata:
    input_path: Path
    fps: float
    width: int
    height: int
    frame_count: int

@dataclass(frozen=True)
class FramePlan:
    start_frame: int
    stop_frame: int
    written_frames: int
    selected_indices: tuple[int, ...]
```

Add tests proving:

```python
metadata = VideoMetadata(Path("clip.mp4"), 30.0, 1920, 1080, 10)
assert build_frame_plan(metadata, start_frame=2, max_frames=6, stride=3) == FramePlan(
    start_frame=2,
    stop_frame=8,
    written_frames=6,
    selected_indices=(2, 5),
)
```

Also assert full-window planning, clipping beyond EOF, a one-frame window, stride 1, invalid negative/zero values, and `start_frame >= frame_count` failures.

Mock captures to assert `inspect_video` returns exact metadata, rejects unopened capture/invalid FPS/size/count, and calls `release()` once on every path.

- [x] **Step 2: Run preflight tests and verify the new interfaces are absent**

Run: `.venv/bin/python -m pytest tests/test_video_plan.py -v`

Expected: FAIL because `VideoMetadata`, `FramePlan`, `inspect_video`, and `build_frame_plan` do not exist.

- [x] **Step 3: Implement immutable metadata and pure planning**

Add the two records to `contracts.py`. Implement in `video.py`:

```python
def inspect_video(input_path: Path) -> VideoMetadata: ...

def build_frame_plan(
    metadata: VideoMetadata,
    *,
    start_frame: int,
    max_frames: int | None,
    stride: int,
) -> FramePlan: ...
```

Keep `selected_frame_indices(start, stop, stride)` as a pure public helper in `video.py`; `build_frame_plan` must call it so there is one selection rule.

- [x] **Step 4: Write failing preflight-order and cache-state tests**

Add tests that monkeypatch gallery loading, `build_models`, and `DeepFace.represent` to raise if reached. Assert invalid video metadata and an out-of-range start return non-zero without any forbidden call.

For selected indices `(2, 5, 8)`, populate cache states as `2=ok([])`, `5=failed`, and `8=absent`; assert preflight reports `selected=3`, `cached=1`, and `to_infer=2` and keeps `(5, 8)` in pending order.

- [x] **Step 5: Integrate preflight before gallery work**

In the current executable orchestration, validate cross-path constraints, call `inspect_video`, call `build_frame_plan`, open one `FaceCache`, calculate `pending_indices = tuple(cache.missing(plan.selected_indices))`, and print the approved counts before `load_gallery`.

Extend the current `process_video` signature without breaking direct callers:

```python
def process_video(
    cfg: Config,
    gallery: Gallery | None = None,
    *,
    metadata: VideoMetadata | None = None,
    plan: FramePlan | None = None,
    face_cache: FaceCache | None = None,
) -> RunSummary: ...
```

Require either none or all three optional objects. When supplied, reuse them; when omitted, construct them internally. Verify the streaming capture metadata matches `metadata` before creating/publishing output.

- [x] **Step 6: Verify preflight and unchanged video behavior**

Run:

```bash
.venv/bin/python -m pytest tests/test_video_plan.py tests/test_video_integration.py -m "not slow" -v
.venv/bin/python -m pytest -m "not slow" -q
```

Expected: all fast tests pass; invalid inputs perform no gallery/model work; cache state counts distinguish ok-empty, failed, and absent frames.

---

### Task 3: Pure Recognition, Rendering, and Evidence Modules

**Files:**
- Create: `face_labeller/recognition.py`
- Create: `face_labeller/rendering.py`
- Create: `face_labeller/evidence.py`
- Modify: `label_video.py`
- Modify: `tests/test_matching.py`
- Modify: `tests/test_perception.py`
- Modify: `tests/test_video_integration.py`
- Modify: `tests/test_facade.py`

**Interfaces:**
- Consumes: approved contracts, presentation constants, `Config`, NumPy, OpenCV, and DeepFace confidence conversion.
- Produces: pure `cosine_distances`, pure `match`, `unknown_matches`, pure `draw`, CSV-path validation, and `MatchLogger`.

- [x] **Step 1: Add ownership assertions before extraction**

Extend facade tests after extraction to assert:

```python
assert label_video.match is face_labeller.recognition.match
assert label_video.draw is face_labeller.rendering.draw
assert label_video.MatchLogger is face_labeller.evidence.MatchLogger
```

Move tests that patch confidence calculation to patch `face_labeller.recognition.verification`; move OpenCV drawing patches to `face_labeller.rendering.cv2`; move CSV/crop patches to `face_labeller.evidence.cv2`.

- [x] **Step 2: Extract recognition with no behavior changes**

Move `cosine_distances`, `match`, and `_unknown_matches` to `recognition.py`; rename only the internal helper to public `unknown_matches(faces: Sequence[Face]) -> list[Match]` so `video.py` can consume it without importing a private name. Retain deterministic NumPy `argmin`, strict threshold comparison, nearest-name retention, and the exact confidence call.

- [x] **Step 3: Extract pure rendering**

Move palette/font constants, label placement, and `draw` to `rendering.py`. Preserve input-copy semantics, clipping, edge label placement, colors, text, confidence rounding, landmarks, and zero-area behavior.

- [x] **Step 4: Extract evidence output**

Move `_MATCH_COLUMNS`, CSV-path validation, and `MatchLogger` to `evidence.py`. Preserve staged overwrite behavior, exact column order, Unknown serialization, crop clipping/naming, and no distance recalculation.

- [x] **Step 5: Re-export and verify the pure/output slice**

Explicitly re-export the moved supported names from `label_video.py`. Run:

```bash
.venv/bin/python -m pytest tests/test_matching.py tests/test_perception.py tests/test_facade.py -v
.venv/bin/python -m pytest -m "not slow" -q
```

Expected: matching/drawing/logger tests pass with patches aimed at their owning modules; complete fast suite passes.

---

### Task 4: Perception and FaceCache Extraction

**Files:**
- Create: `face_labeller/perception.py`
- Create: `face_labeller/cache.py`
- Modify: `label_video.py`
- Modify: `face_labeller/video.py`
- Modify: `tests/test_perception.py`
- Modify: `tests/test_face_cache.py`
- Modify: `tests/test_video_integration.py`
- Modify: `tests/test_facade.py`

**Interfaces:**
- Consumes: `Config`, `Face`, input frames, input video bytes, fixed upstream metadata, installed versions.
- Produces: `build_models`, `model_load_seconds`, `embed_faces`, `process_batch_with_fallback`, stable cache-key helpers, and `FaceCache`.

- [x] **Step 1: Add golden cache-identity tests before moving code**

Using a file containing exactly `b"video-fixture"` and fixed versions:

```python
versions = {
    "deepface": "0.0.101",
    "retinaface": "0.0.18",
    "tensorflow": "2.21.0",
    "opencv": "5.0.0.93",
}
```

assert:

```text
gallery_cache_key = 799c07df4ebcc29fefc98c3feb18383d84f269e9f09ff4afbfe270233161f0fc
face_cache_key    = 13fdefcc341fd7d2482b1c2b542ea0d0e4b24eee705f77ff2657f53a22fa9051
```

Keep existing assertions for exact FaceCache NPZ fields, metadata, states, and old-cache load behavior.

- [x] **Step 2: Extract perception and expose read-only timing**

Move lazy DeepFace import, `_MODEL_BUILT`, `_MODEL_LOAD_SECONDS_TOTAL`, `build_models`, DeepFace result normalization, clipping, face conversion, embedding validation, `embed_faces`, and `process_batch_with_fallback` into `perception.py`.

Add:

```python
def model_load_seconds() -> float:
    return _MODEL_LOAD_SECONDS_TOTAL
```

Tests patch `face_labeller.perception._get_deepface`, `time.perf_counter`, and model state directly. Production modules use the accessor rather than importing mutable counters.

- [x] **Step 3: Extract shared cache identity and FaceCache**

Move `_file_sha256`, `_installed_versions`, stable JSON hashing, face-cache metadata/key generation, embedding validation shared by cache decoding, and `FaceCache` into `cache.py`. Preserve the constructor and public methods:

```python
FaceCache(video_path, cfg, versions=None)
status(frame_idx)
get(frame_idx)
put_ok(frame_idx, faces)
put_failed(frame_idx)
missing(frame_indices)
flush()
```

Do not change schema constants, metadata contents, safe stem calculation, dtypes, or atomic replacement.

- [x] **Step 4: Rewire preflight and current streaming code**

Update `video.py` and the still-current orchestration to import `FaceCache` from `cache` and batch fallback/model timing from `perception`. Re-export supported names from the facade.

- [x] **Step 5: Verify lazy loading, cache compatibility, and fallback**

Run:

```bash
.venv/bin/python -m pytest tests/test_perception.py tests/test_face_cache.py tests/test_video_plan.py -v
.venv/bin/python -m pytest tests/test_video_integration.py -m "not slow" -v
.venv/bin/python -m pytest -m "not slow" -q
```

Expected: both golden keys match; existing cache serialization tests pass; batch fallback isolates one failing frame; cached replay makes no model call.

---

### Task 5: Gallery Extraction

**Files:**
- Create: `face_labeller/gallery.py`
- Modify: `label_video.py`
- Modify: `tests/test_gallery.py`
- Modify: `tests/test_facade.py`

**Interfaces:**
- Consumes: owner-curated paths, `Config`, cache helpers, perception embedding/model behavior, and pure recognition/build-pin inputs.
- Produces: `gallery_cache_key`, `build_pins`, `load_gallery`, and `leave_one_out_report` with unchanged behavior and metadata.

- [x] **Step 1: Move gallery tests to intended ownership boundaries**

Update imports to `face_labeller.gallery`. Patch `face_labeller.gallery.cv2.imread`, `face_labeller.gallery._embed_gallery_photo`, and the perception boundary rather than facade globals. Retain the golden gallery key from Task 4.

- [x] **Step 2: Extract gallery discovery, cache, and pin construction**

Move extension filtering, deterministic path ordering, per-photo hashing, gallery metadata/key generation, image embedding, `build_pins`, cache loading/writing, `load_gallery`, and `leave_one_out_report` into `gallery.py`.

Use shared hash/version/embedding validation helpers from `cache.py`; do not generalize the two NPZ formats into a speculative storage framework.

- [x] **Step 3: Preserve gallery cache reconciliation exactly**

Keep one embedding per photo, normalized relative source paths, photo hashes in entries rather than configuration identity, compatible keyed variants, removal of deleted entries, and pin strategy applied only after cache reconciliation.

- [x] **Step 4: Re-export and verify gallery behavior**

Run:

```bash
.venv/bin/python -m pytest tests/test_gallery.py tests/test_facade.py -v
.venv/bin/python -m pytest -m "not slow" -q
```

Expected: minimum-image, invalid-image, mean/all pin, leave-one-out, cold/warm/add/change/delete, corrupt-cache, and zero-model-load replay tests pass.

---

### Task 6: Streaming Video Module and Execution-Core Cutover

**Files:**
- Modify: `face_labeller/video.py`
- Create: `face_labeller/core.py`
- Modify: `label_video.py`
- Modify: `tests/test_video_integration.py`
- Modify: `tests/test_video_plan.py`
- Modify: `tests/test_facade.py`

**Interfaces:**
- Consumes: all stable modules from Tasks 1-5.
- Produces: `video.process_video(...) -> RunSummary`, `core.run(cfg) -> RunSummary`, and thin `label_video.main(argv) -> int`.

- [x] **Step 1: Write failing core sequencing and CLI delegation tests**

Add `test_core_runs_preflight_before_gallery` with ordered spies asserting:

```text
validate paths -> inspect video -> build plan -> open cache -> print plan
-> load gallery -> process video -> print final summary
```

Add `test_main_delegates_to_core` asserting parsed `Config` is passed once to `core.run` and success returns 0. Add `test_main_maps_core_error_to_nonzero` asserting `ERROR: <message>` goes to stderr and return value is 1.

- [x] **Step 2: Move streaming execution into `video.py`**

Move `process_video`, progress reporting, capture/writer lifecycle, pending frame buffers, cache updates, matching, CSV logging, drawing, and output publication from `label_video.py` to `video.py`.

Retain the exact signature introduced in Task 2. Validate that optional `metadata`, `plan`, and `face_cache` are either all supplied or all absent. Preserve bounded pending frames, every-frame output, selected-frame-relative stride, failed-frame handling, label distribution, and timing fields.

- [x] **Step 3: Implement the single execution core**

Implement:

```python
def run(cfg: Config) -> RunSummary: ...
```

Move cross-path validation, metadata/plan/cache construction, deterministic preflight printing, gallery timing/loading, `process_video` invocation, and final completion printing into `core.py`. Use `perception.model_load_seconds()` snapshots to preserve existing gallery/video model timing output without reading a mutable global.

- [x] **Step 4: Reduce `label_video.py` to facade and process adapter**

Keep explicit compatibility imports and:

```python
def main(argv: Sequence[str] | None = None) -> int:
    cfg = load_config(parse_args(argv))
    try:
        run(cfg)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0
```

Retain the `SystemExit(main())` guard. Remove all duplicated implementations after their owning-module tests pass. Ensure no `face_labeller` module imports `label_video`.

- [x] **Step 5: Add stale-plan and direct-caller coverage**

Test that `process_video(cfg, gallery)` still self-plans. Test that passing only one/two preflight objects raises clearly. Test that a streaming capture whose FPS, width, height, or frame count differs from `VideoMetadata` fails before output publication.

- [x] **Step 6: Verify the complete modular application**

Run:

```bash
.venv/bin/python -m pytest tests/test_facade.py tests/test_video_plan.py tests/test_video_integration.py -m "not slow" -v
.venv/bin/python -m pytest -m "not slow" -v
.venv/bin/python label_video.py --help
```

Expected: all fast tests pass; facade is thin; package dependency direction is acyclic; help is unchanged.

---

### Task 7: Cold/Warm Parity, Documentation, and Refactor STOP Gate

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Verify: `docs/design/design-plan.md`
- Verify: `docs/implementation/implementation-plan.md`
- Verify: `docs/archive/specs/2026-09-26-label-video-modularization-design.md`
- Verify: all production/test files from Tasks 1-6
- Generate only in ignored paths: short-run video, CSV, cache, and comparison evidence

**Interfaces:**
- Consumes: completed modular pipeline and existing owner-curated input/gallery/cache assets.
- Produces: architecture documentation, compatibility evidence, timing comparison, and the M4 modularization STOP report.

- [x] **Step 1: Update architecture and developer documentation**

Update README architecture/module descriptions and keep the existing CLI examples. Update AGENTS repository layout and commands so future work imports and patches owning modules. Reaffirm the M4 checkpoint, final stride 1, CPU-only requirement, cache semantics, and M6 prohibition.

- [x] **Step 2: Run the complete automated verification**

Run:

```bash
.venv/bin/python -m pytest -m "not slow" -v
.venv/bin/python -m pytest -m slow -v
.venv/bin/python label_video.py --help
git diff --check
```

Expected: all applicable tests pass. Record exact pass/skip counts and durations. If a slow test requires a missing owner/private fixture, report the skip/block precisely rather than substituting downloaded data.

- [x] **Step 3: Run a short cold/warm parity window**

Use an owner-approved existing short window and isolated ignored cache/output paths. Run once cold and once warm with identical CLI options. Record command, total/window/selected/cached/to-infer counts, model/gallery/perception/elapsed timings, faces, failures, labels, output metadata, and CSV row count.

The warm run must report zero frames to infer and must prove no model build/represent call through existing instrumentation/tests.

- [x] **Step 4: Compare observable artifacts**

Against the equivalent pre-refactor behavior or retained fixture expectations, verify:

- output frame count, FPS, width, height, and duration tolerance;
- exact CSV columns and row ordering;
- cache key/path and metadata compatibility;
- label and box behavior on representative frames; and
- no generated/private artifact is tracked.

Do not claim byte-identical MP4 output because codec/container metadata may vary; compare decoded frames or representative pixels when exact rendering parity is needed.

- [x] **Step 5: Audit repository state**

Run:

```bash
git status --short
git diff --check
git check-ignore data/video-source/nimbus.mp4 cache output
```

Expected: only intended source/test/docs changes are unignored; private/generated assets remain ignored.

- [x] **Step 6: Report the modularization STOP gate**

Report:

- modules created and the final `label_video.py` responsibility;
- how preflight calculates and reports work before gallery/model use;
- how tests and parity evidence preserve R1 and R2;
- fast/slow test counts and durations;
- cold/warm timings and zero-inference evidence;
- cache/CLI/data/CSV compatibility evidence;
- any open questions or deviations; and
- that the full 3,044-frame M4 run has not started.

Stop and wait for Eli's approval before M4 Step 13.

- [x] **Step 7: Commit only if explicitly requested**

If Eli asks after reviewing the STOP report, run the full relevant test suite immediately before committing, stage only the modularization source/tests/docs, and use a plain-language message such as:

```bash
git commit -m "refactor: split the video labeller into focused modules"
```

Run the same suite immediately after the commit and report both results.
