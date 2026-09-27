# Label Video Modularization Design

> **Archived approved design.** Implemented and merged in commit `79431f5` (PR #1).
> Current architecture and status live in [`docs/STATUS.md`](../../STATUS.md) and the
> authoritative design plan.

**Status:** Approved by Eli on 2026-09-26

**Date:** 2026-09-26

**Authority:** This document is the owner-approved modularization amendment to
`docs/design/design-plan.md` and `docs/implementation/implementation-plan.md`. The
authoritative documents link to this specification and retain authority for all
unamended requirements.

## 1. Intent

Split the 1,423-line `label_video.py` pipeline into focused modules while keeping one
obvious execution core and preserving the existing command:

```bash
python label_video.py --input ... --output ... --ref-dir ...
```

The refactor also adds a model-free preflight stage. Given the existing `--input` video
path and frame-window options, preflight validates the video and calculates the exact
absolute frame indices selected for processing. After opening the compatible FaceCache,
the application reports which selected frames are already cached and which require
inference before loading a gallery or model.

This work is an amendment inside the unfinished M4 milestone. It must reach its own STOP
gate before the full 3,044-frame M4 run. It does not begin M5 or M6.

## 2. Goals

1. Give each production module one clear responsibility and an explicit dependency
   direction.
2. Make `face_labeller.core.run(cfg)` the single end-to-end execution coordinator.
3. Keep `label_video.py` as the executable deliverable and a compatibility facade.
4. Fail on invalid video metadata or frame windows before gallery embedding or frame
   inference can begin.
5. Calculate and report total, window, selected, cached, and inference-needed frame
   counts before expensive work.
6. Preserve R1: every detected face is boxed, Unknown faces are retained, no default face
   cap is introduced, and the final run uses stride 1.
7. Preserve R2: each box receives the nearest accepted character name or `Unknown` under
   the existing strict-threshold rule.
8. Preserve cache compatibility and zero-inference replay behavior.

## 3. Non-goals

- No change to the approved CLI flags, defaults, or command shape.
- No interactive input prompt, configuration file, GUI, REST API, or directory-wide video
  discovery.
- No change to `Face`, `Match`, `Gallery`, `Config`, or `RunSummary` fields.
- No change to the CSV schema, drawing appearance, matching rule, gallery validation,
  model parameters, or cache-key inputs.
- No FaceCache or gallery-cache migration and no schema-version bump solely because code
  moved.
- No inference optimization claim. Modularization must not be presented as making
  RetinaFace or Facenet512 faster.
- No tracking, temporal smoothing, audio remux, tuning, or other M5/M6 work.
- No reference-image acquisition or generated/private artifact changes.

## 4. Current State

`label_video.py` currently contains 37 top-level functions and 9 classes across 1,423
lines. It owns contracts, CLI parsing, DeepFace model lifecycle, perception, two cache
families, gallery indexing, matching, evidence logging, rendering, video streaming, and
top-level error handling.

The current fast-test baseline is:

```text
119 passed, 3 deselected
```

The command and its imported symbols are exercised directly by the test suite. Several
tests also monkeypatch implementation globals through `label_video`; those patches are
test-internal coupling rather than a supported runtime contract and will move to the
module that owns the behavior.

## 5. Target File Structure

```text
label_video.py
face_labeller/
    __init__.py
    contracts.py
    config.py
    perception.py
    cache.py
    gallery.py
    recognition.py
    rendering.py
    evidence.py
    video.py
    core.py
```

### `label_video.py`

The executable adapter and compatibility facade. It imports the CLI/config entrypoints
and `core.run`, converts an application exception to the existing non-zero process result,
and retains `if __name__ == "__main__"` execution. The core prints the preflight and final
run summaries because it owns the timing context needed to preserve their current fields.

It temporarily re-exports the existing named application interfaces used by tests and
potential local consumers. It does not proxy third-party module objects or mutable module
globals such as `cv2`, `time`, or model-load counters.

### `face_labeller/contracts.py`

Owns dependency-light records and fixed domain constants:

- `Face`
- `Match`
- `Gallery`
- `GalleryPhoto`
- `Track`
- `RunSummary`
- `VideoMetadata`
- `FramePlan`
- canonical character names and fixed model/detector names

The approved records retain their existing fields and semantics. `VideoMetadata` and
`FramePlan` are new internal frozen records, not replacements for an approved contract.

### `face_labeller/config.py`

Owns `Config`, default values, argument validators, `build_parser`, `parse_args`, and
`load_config`. Every existing flag and default remains visible in `--help`.

### `face_labeller/perception.py`

Owns lazy DeepFace access, Facenet512 model construction, DeepFace result-shape
normalization, box clipping, embedding validation, `embed_faces`, and batch fallback.
It is the only production module that calls `DeepFace.build_model` or
`DeepFace.represent`.

It exposes a read-only `model_load_seconds() -> float` instrumentation accessor so the
video runner and core can preserve current timing semantics without importing a mutable
module global.

Every call continues to pass `model_name="Facenet512"` and
`detector_backend="retinaface"` explicitly. Model loading remains lazy and at most once
per process.

### `face_labeller/cache.py`

Owns shared deterministic hashing/version helpers, FaceCache metadata and key generation,
the `FaceCache` class, and atomic FaceCache persistence. Gallery-specific serialization
stays in `gallery.py`, while both cache families use the same small hashing/version
utilities.

Cache metadata dictionaries, canonical JSON encoding, filenames, and schema versions must
remain identical for identical pre-refactor inputs.

### `face_labeller/gallery.py`

Owns gallery discovery, per-photo embedding, gallery-cache reconciliation and atomic
persistence, pin construction, `load_gallery`, and `leave_one_out_report`.

It may depend on contracts, config, cache helpers, perception, and recognition. It must
not import the video runner or execution core.

### `face_labeller/recognition.py`

Owns pure `cosine_distances`, pure `match`, and construction of Unknown placeholder
matches used before gallery recognition. It performs no file, video, cache, configuration,
or model I/O.

### `face_labeller/rendering.py`

Owns fixed presentation constants and pure `draw`, including label placement. It returns a
copy and never mutates the source frame.

### `face_labeller/evidence.py`

Owns CSV-path validation, the exact approved column order, `MatchLogger`, staged CSV
publication, and optional debug-crop writes. It consumes completed `Match` values and never
recalculates distances.

### `face_labeller/video.py`

Owns video inspection, pure frame-window planning, capture/writer lifecycle, selected-frame
streaming, FaceCache use, matching, logging, rendering, progress reporting, and
`process_video`.

It does not parse CLI arguments or load the gallery. It receives a complete `Config`, an
optional `Gallery`, and optionally a precomputed metadata/plan/cache bundle from the core.
Direct callers of the existing `process_video(cfg, gallery=None)` form remain supported;
when no plan is supplied, `process_video` constructs the same plan internally.

### `face_labeller/core.py`

Owns the application use case:

```python
def run(cfg: Config) -> RunSummary:
    ...
```

It validates cross-path constraints, performs video preflight, opens the compatible
FaceCache, calculates and reports pending work, loads the gallery, invokes video
processing, and returns the summary. It does not parse raw command-line strings and does
not contain OpenCV or DeepFace algorithms.

## 6. Dependency Direction

The intended dependency graph is acyclic:

```text
label_video.py
    -> config
    -> core

core
    -> contracts, config, cache, gallery, perception, video

video
    -> contracts, config, cache, perception, recognition, rendering, evidence

gallery
    -> contracts, config, cache, perception, recognition

perception / recognition / rendering / evidence / cache
    -> contracts and config only where needed
```

No package module imports `label_video.py`. `contracts.py` imports no other project module.
`core.py` is the outer coordinator; lower-level modules never call back into it.

## 7. Video Preflight and Frame Planning

### Interfaces

```python
def inspect_video(input_path: Path) -> VideoMetadata:
    ...

def build_frame_plan(
    metadata: VideoMetadata,
    *,
    start_frame: int,
    max_frames: int | None,
    stride: int,
) -> FramePlan:
    ...
```

`VideoMetadata` contains the resolved input path, FPS, width, height, and total frame
count. `FramePlan` contains the requested start, exclusive stop, number of frames that will
be written, and the ordered tuple of absolute selected indices.

`inspect_video` opens the path through OpenCV, verifies the capture, reads the metadata,
closes the capture on every path, and raises the same clear errors currently produced by
`process_video` for invalid input, FPS, size, or frame count.

`build_frame_plan` is pure. It validates that `start_frame` lies inside the video, clips the
exclusive stop to the video length, and selects frames relative to the requested window
with `range(start, stop, stride)`. The first requested frame is therefore always selected.

### Cached-versus-pending calculation

After preflight, `core.run` opens the FaceCache using the existing input-video hash and
upstream perception metadata. It calculates:

```python
pending_indices = tuple(face_cache.missing(frame_plan.selected_indices))
```

`failed` entries remain pending because `FaceCache.missing` already treats any non-`ok`
state as work to retry. Empty-face `ok` entries remain cache hits.

Before gallery loading, the core prints a deterministic summary containing:

- input path;
- total frames;
- requested half-open window;
- frames that will be written;
- stride;
- selected frames;
- cached selected frames; and
- frames requiring inference.

The plan is informational and authoritative for the run. It never changes stride,
silently caps work, or skips a selected failed frame.

### Processing handoff

The core passes the precomputed metadata, frame plan, and FaceCache into `process_video` so
the cache is not reopened and the input hash is not recomputed. `process_video` must still
open the capture used for streaming and verify that its FPS, dimensions, and frame count
match the preflight metadata before writing output. A mismatch fails rather than processing
against a stale plan.

## 8. End-to-End Data Flow

```text
CLI arguments
  -> Config
  -> core.run
       -> inspect_video
       -> build_frame_plan
       -> FaceCache + pending indices
       -> print preflight summary
       -> load_gallery
       -> process_video
            -> cached faces or perception
            -> match
            -> MatchLogger
            -> draw
            -> VideoWriter
       -> print final summary
       -> RunSummary
  -> CLI exit code
```

Gallery loading remains before frame inference so that a gallery failure cannot leave a
claimed successful labelled output. Both gallery and perception retain lazy model loading:
a fully cached gallery plus fully cached frame plan must not load Facenet512.

## 9. Compatibility Requirements

### CLI

The complete approved CLI remains unchanged. In particular, `--input` remains required and
is the sole way to select the source video. Preflight is automatic and does not require a
new flag.

### Public Python surface

During this refactor, `label_video.py` continues to expose the currently imported names:

- contracts and constants;
- config/parser functions;
- perception, cache, gallery, matching, rendering, and evidence entrypoints;
- `process_video`; and
- `main`.

New production code and migrated tests import from owning package modules. A dedicated
facade test proves the supported names still resolve from `label_video`.

Mutable implementation details such as `_MODEL_LOAD_SECONDS_TOTAL` are not compatibility
contracts. Tests that need them patch or inspect `face_labeller.perception` directly.

### Caches and artifacts

For identical inputs and installed versions:

- `gallery_cache_key` and `face_cache_key` return the same strings as before;
- gallery and face cache filenames are unchanged;
- NPZ fields, metadata JSON, dtypes, statuses, and schema versions are unchanged;
- old valid caches load without migration;
- `matches.csv` columns and row semantics are unchanged; and
- output video frame selection, count, FPS, dimensions, annotations, and codec are
  unchanged.

## 10. Error Handling and Atomicity

- Video preflight errors occur before gallery/model work.
- Cross-path validation still rejects input/output identity and unsafe CSV destinations.
- A batch inference failure still retries each selected frame individually.
- An individually failing frame is cached as `failed`, written unannotated, and retried on
  the next run.
- Gallery images that cannot produce a valid face remain warnings; the two-valid-images
  minimum remains enforced per character.
- FaceCache, gallery cache, CSV, and video output keep their existing staged/atomic
  publication behavior.
- Temporary output is removed on failure. Existing successful outputs are replaced only
  after a successful run.
- Preflight capture and processing capture are both released on success and failure.

## 11. Testing Strategy

Every extraction slice must leave the CLI runnable and the applicable test suite passing.
Tests move with the behavior they verify; they do not retain monkeypatches through the
facade when the true owner is a package module.

### Characterization and facade tests

- Pin the existing CLI flags/defaults and approved dataclass fields.
- Pin cache keys and serialized field sets using deterministic fixtures.
- Assert every supported compatibility name resolves from `label_video`.
- Assert `label_video.main` delegates to the core and preserves success/failure exit codes.

### Preflight tests

- Missing or unreadable video.
- Invalid FPS, dimensions, and frame count.
- Start frame equal to or beyond total frame count.
- Full video, clipped window, one-frame window, and non-zero start.
- Strides 1, 2, and 3 with selection relative to the window start.
- Empty-face `ok`, `failed`, and absent cache states when calculating pending work.
- Capture release on every success and failure path.
- No gallery, `build_model`, or `represent` call before preflight succeeds.
- Processing fails if reopened video metadata differs from the plan.

### Module tests

- Existing perception tests patch `face_labeller.perception`.
- Existing cache tests patch `face_labeller.cache`.
- Existing gallery tests patch `face_labeller.gallery` and perception at their ownership
  boundaries.
- Matching and rendering remain pure and independently testable.
- Evidence tests retain exact CSV and crop behavior.
- Video integration tests patch component interfaces rather than unrelated globals.

### Regression commands

```bash
.venv/bin/python -m pytest -m "not slow" -v
.venv/bin/python -m pytest -m slow -v
.venv/bin/python label_video.py --help
git diff --check
```

After automated verification, run the same short cold and warm window before and after the
refactor and compare metadata, cache hit/miss/failure counts, CSV contents, and representative
rendered frames. The existing M4 full-video run remains blocked until this refactor's STOP
gate is accepted.

## 12. Vertical Delivery Slices

### Slice 0: Authority and characterization

Approve this specification, amend the authoritative design/implementation documents, and
add compatibility characterization where current behavior is not already pinned.

Exit: documented single-file requirements are replaced, cache/CLI/data boundaries are
explicit, and the existing suite passes.

### Slice 1: Package spine and stable contracts

Create the package; move contracts and configuration; retain facade re-exports.

Exit: the command and supported imports behave exactly as before, with no model or video
behavior moved yet.

### Slice 2: Video preflight vertical slice

Add metadata inspection, pure frame planning, and pending-work reporting. Invoke it from
the current executable orchestration while the remaining pipeline is still monolithic;
the execution-core cutover in Slice 6 transfers that unchanged sequence to `core.run`.

Exit: a valid invocation reports its planned frame work before expensive operations; an
invalid invocation performs no gallery or model work.

### Slice 3: Pure recognition, rendering, and evidence

Move the downstream stateless logic and evidence writer behind their stable interfaces.

Exit: matching, drawing, CSV, crop, threshold, and cached replay behavior remain identical.

### Slice 4: Perception and FaceCache

Move DeepFace integration, batch recovery, cache identity, and FaceCache persistence.

Exit: explicit model/detector arguments, lazy loading, old-cache reuse, failed-frame retry,
and cross-stride reuse all pass their tests.

### Slice 5: Gallery indexing

Move gallery validation, incremental embedding cache, pin construction, and leave-one-out
reporting.

Exit: cold, warm, add/change/delete, pin-strategy, and minimum-gallery behavior remain
unchanged.

### Slice 6: Execution-core cutover

Move streaming video execution into `video.py`, make `core.run` the sole coordinator, and
reduce `label_video.py` to its executable/facade role.

Exit: short cold and warm end-to-end runs match the monolith's observable outputs and the
full applicable suite passes.

### Slice 7: Documentation and M4 refactor STOP gate

Update diagrams, commands, module ownership, and developer guidance. Report what changed,
test results, timings, compatibility evidence, effects on R1/R2, and open questions.

Exit: Eli accepts the refactor before the pending full 3,044-frame stride-1 M4 run begins.

## 13. Risks and Mitigations

### Import-cycle risk

Mitigation: enforce the dependency direction in section 6; contracts remain at the bottom
and no module imports the executable facade.

### Monkeypatch and global-state drift

Mitigation: move tests to the true owner of model state and third-party dependencies; keep
facade tests focused on supported names and CLI delegation.

### Cache invalidation caused by refactoring

Mitigation: characterize keys and NPZ schemas before moving code, then load a pre-refactor
fixture after extraction. Module names never enter cache identity.

### Duplicate preflight work

Mitigation: pass the opened FaceCache and computed plan into video processing. The video is
reopened for streaming, but its full hash is not recalculated.

### Stale plan after input replacement

Mitigation: processing rechecks video metadata against preflight. Cache identity continues
to use the full video hash, so a replacement with matching superficial metadata selects a
different cache.

### Big-bang behavioral regression

Mitigation: move one responsibility cluster per slice, retain the executable after every
slice, and run focused tests plus the complete fast suite at every STOP point.

## 14. Rollback

Each slice is independently reviewable. If a slice fails its behavioral gate, revert only
that slice while retaining the last accepted runnable state. Because caches and artifact
schemas are not migrated, rollback requires no data conversion and existing caches remain
usable by the pre-refactor code.

## 15. Acceptance Criteria

The modularization is complete when:

1. `label_video.py` is a thin executable/compatibility facade and contains no pipeline
   algorithms.
2. `face_labeller.core.run(cfg)` visibly describes the end-to-end sequence.
3. A preflight summary is emitted before gallery/model work and accurately reports the
   configured frame window, selected-frame count, cache hits, and pending inference; the
   underlying `FramePlan` retains the exact ordered absolute indices.
4. Invalid video metadata or frame windows trigger no model/gallery work.
5. The existing CLI and approved data contracts are unchanged.
6. Existing valid gallery and FaceCache files remain reusable without migration.
7. A fully cached relabel still loads no model and performs no inference.
8. All applicable fast and slow tests pass.
9. A short cold/warm output comparison shows no unexplained changes to video metadata,
   CSV evidence, labels, boxes, or cache accounting.
10. The STOP report states how the refactor preserves R1/R2, gives timings and evidence,
    and receives owner approval before the full M4 run.
