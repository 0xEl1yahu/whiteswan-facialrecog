# SPEC: Character Face Labeller (White Swan Data ML Assessment)

## 0. Objective
Core requirements (from the brief; every milestone serves these two):
  R1. Draw bounding boxes around ALL the faces in the video.
  R2. Label the boxes with the respective character's name when possible, otherwise
      "Unknown".

Given an input video, produce an output video where every detected face has a bounding box,
labelled with a character name when recognisable, otherwise "Unknown".
R1 implications: max_faces defaults to None (no cap); no face is dropped for being
unrecognised (Unknown faces are still boxed); every output frame carries boxes. The final
deliverable uses stride 1 so detection is attempted on every frame. Stride remains
configurable for development and preview runs; at stride > 1, skipped frames reuse the last
boxes and can lag on entrances, exits, cuts or fast motion. This trade-off is reported at M3.
Characters: Harry Potter, Ron Weasley, Hermione Granger, Prof. McGonagall, Prof. Severus Snape.
Required stack: Python + DeepFace, detector = RetinaFace, recogniser = Facenet512.
Accuracy bar: "reasonable given the model's capabilities". Not perfect.

Deliverables: `label_video.py`, `requirements.txt`, `README.md`, output video, `matches.csv`.

Priorities, in order:
1. A first vertical slice (M1-M4: video in, labelled video + CSV out) that is built and
   provable, with tests and evidence at every gate.
2. Data efficiency: every expensive result (face detection, embeddings) is computed once,
   cached, and reused. Matching and tuning replay from cache and never re-run the models.
3. Only then tuning (M5), and enhancements/extras (M6+) on the owner's explicit command.

## 1. Scope
In scope: detection, recognition against a reference gallery, annotation, video I/O,
match logging, optional temporal smoothing, optional audio re-mux.
Non-goals: training or fine-tuning models, vector databases (`DeepFace.register/search/
identify/build_index`), demography (`DeepFace.analyze`), anti-spoofing, web UI, REST API.

## 2. Constraints
- Python 3.11 (`python3.11`, uv-managed on the dev machine), in a project `.venv/`.
  Why: TensorFlow 2.21 (current) ships wheels for CPython 3.10-3.13 only, so the machine's
  default `python3` (3.14) cannot install the stack. 3.11 sits inside that window with
  headroom on both sides, is already installed, and is the most exercised version across
  the CV stack (deepface, retina-face, opencv). 3.10 remains valid but is older and slower.
- Install: `pip install "deepface[tensorflow]" tf-keras opencv-python gdown`.
  `tf-keras` is REQUIRED: TF >= 2.16 uses Keras 3, and deepface raises ValueError at import
  if `tf_keras` is missing; the `[tensorflow]` extra does not pull it in.
  Audited against deepface 0.0.101 and retina-face 0.0.18. Pin exact versions in
  requirements.txt after the first working run.
- Detector and model are fixed: `detector_backend="retinaface"`, `model_name="Facenet512"`.
  Always pass both explicitly (DeepFace defaults are opencv / VGG-Face).
- Distance metric: cosine.
- CPU only (slow is acceptable). GPU is out of scope: do not add GPU code paths, flags or
  benchmarks.
- Deterministic: the same input and config must produce the same output.
- Reference photos are human-curated by the owner. Agents must NOT scrape or download face
  images. Agents build tooling that validates the gallery.
- Video source: Google Drive file id 1CM1IWUN59ZWml9MwgrvSHXz_9AirIiuU, saved as
  `data/video-source/nimbus.mp4` (download via gdown, documented in the README; gitignored).
  Clip facts: Philosopher's Stone, 1920x1080 H.264, 30000/1001 fps, 3044 frames, ~101.6 s,
  AAC audio track.

Repo layout:
```
docs/design/design-plan.md      approved design specification
docs/implementation/           milestone implementation plans
label_video.py                  executable and compatibility facade
face_labeller/                  focused pipeline modules and execution core
tests/                          pytest suite (unit + slow integration)
data/video-source/nimbus.mp4    input clip            (gitignored)
data/reference-images/<Name>/   owner-curated gallery (gitignored)
cache/                          keyed gallery + per-frame embedding caches (gitignored)
output/                         labelled video, matches.csv, debug/ (gitignored)
```

## 3. Agent operating rules
1. Work one milestone at a time (section 9). EVERY milestone ends in a STOP gate: halt and
   report what was built, test results, timings, and open questions, so the owner can test
   the system before the next milestone starts. Work runs up to M5; M6 starts only on the
   owner's explicit command, after the slice is verified.
2. Do not change the data contracts (section 5) or the CLI (section 7) without owner approval.
3. Do not resolve OWNER decisions (section 12). Use the provisional default and flag it.
4. Do not hard-code magic numbers. Every tunable goes in config, with its source noted.
5. Where this spec says "verify", check the behaviour in the installed DeepFace source or with
   a test. Do not assume.
6. Keep pure functions pure: no I/O inside `match`, `cosine_distances`, `draw` or `iou`.
7. Build the smallest provable thing first. No enhancement, tuning knob or extra lands
   before the vertical slice (M1-M4) passes its gates.

## 4. Architecture
Single process, two phases, stages passing explicit records.

INDEXING (once, cached)
  data/reference-images/<Name>/*.jpg -> RetinaFace -> align -> Facenet512
    -> per-photo embeddings (`cache/gallery_{key}.npz`) -> pin strategy applied at load -> Gallery

INFERENCE (streaming, per frame)
  Reader -> Perception -> Matcher -> [Tracker] -> Renderer -> Writer
               |              \-> MatchLogger (matches.csv)
               \-> FaceCache (cache/faces_<video>_<key>.npz)

State lives ONLY in: Gallery (read-only after load), FaceCache, Tracker, Writer, Logger.
Hot path: Perception (RetinaFace dominates runtime).

Data efficiency (compute once, reuse everywhere):
- Each gallery cache stores ONE embedding PER PHOTO. Gallery caches are keyed only by
  inputs that affect embeddings, and compatible variants coexist, so switching away from
  a normalization and back never re-embeds unchanged photos. The pin strategy (D1) is
  applied when the gallery loads and is not part of the embedding key.
- Perception results (boxes, landmarks, det_conf, embeddings) are cached PER FRAME INDEX.
  Each run computes only the frames it needs that are not already cached, so a stride-1 run
  after a stride-3 run, or a full run after a 300-frame window, reuses what exists.
  Threshold, pin strategy and smoothing changes cost seconds, not a detection pass.
- M5 analysis reads the caches and matches.csv. It never calls the models on frames that
  are already cached.

Model facts that constrain the design (verified in deepface 0.0.101 / retina-face 0.0.18
source, 2026-09-26):
- RetinaFace internally rescales detector input to a 1024 px short side (long side capped
  at 1980). Naively downscaling the unpadded frame is rejected because it changed small-face
  recall. The owner-approved M4 path instead reproduces DeepFace's exact padded resize and
  removes only grid-aligned black margin while retaining a 32 px detector halo. For 1080p
  this changes the detector tensor from 1820x1024 to 988x576 without changing the resized
  content pixels. See
  `docs/superpowers/specs/2026-09-26-retinaface-black-margin-optimization-design.md`.
- DeepFace calls RetinaFace with threshold=0.9, hard-coded and NOT configurable through any
  represent()/extract_faces() argument (models/face_detection/RetinaFace.py:48). Real
  detections have face_confidence >= 0.9. face_confidence is rounded to 2 dp.
- With enforce_detection=False and no face in the frame, represent() returns the whole frame
  as a "face" with face_confidence == 0. This MUST be filtered out.
- Result fields: embedding, facial_area, face_confidence (+ face if return_face=True).
  facial_area always has x, y, w, h, left_eye, right_eye; RetinaFace adds nose, mouth_left,
  mouth_right. Each landmark is an (int, int) tuple, or None if outside the image (drop
  None). Left/right are the PERSON's left/right, not the viewer's.
- Alignment levels the eyes; the face is resized to 160x160 for Facenet512.
- normalization options: base (no-op, default), raw, Facenet, Facenet2018, VGGFace,
  VGGFace2, ArcFace. M5 compares base vs Facenet2018.
- Facenet512 = Inception-ResNet v1 with a 512-d bottleneck (Sandberg weights). Weights are
  downloaded once to ~/.deepface/weights/ on first build_model.
- Embeddings are NOT L2-normalised by default (represent(l2_normalize=False)). Pass
  l2_normalize=True, and still assert unit norm on our side.
- Default threshold: find_threshold("Facenet512", "cosine") == 0.30.
- find_confidence(distance, model_name, verified, distance_metric) returns 0-100
  (51-100 = same person, 0-49 = different).
- represent() accepts a list of images. DETECTION IS NOT BATCHED (a Python loop per image);
  only the Facenet512 forward pass is batched. Batch size therefore buys little speed;
  RetinaFace cost per frame dominates.
- SHAPE TRAP: a list of length 1 returns a flat list[dict] (same as a single image), not
  list[list[dict]] (representation.py:225). embed_faces must special-case it.
- max_faces keeps the N largest faces (by w*h) per image.
- return_face=True returns the preprocessed model input (160x160, normalised), NOT the raw
  crop. Debug crops are cut from the original frame using facial_area instead.
- DeepFace.build_model("Facenet512") is enough; task defaults to "facial_recognition".
- deepface 0.0.101 supports TF and PyTorch backends; TF wins when installed. We install TF
  only.
- No randomness in the inference path (training=False). Bit-exact repeatability across
  thread counts is not guaranteed by TF; M0 verifies it with a repeat-run embedding check.

## 5. Data contracts
```python
@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]         # x, y, w, h in frame pixels
    landmarks: dict[str, tuple[int, int]]  # keys as returned by DeepFace; may be empty
    embedding: np.ndarray                  # shape (512,), float32, L2-normalised
    det_conf: float                        # 0 < det_conf <= 1

@dataclass(frozen=True)
class Match:
    nearest_name: str          # owner of the nearest pin, even when assigned Unknown
    name: str | None      # None means Unknown
    distance: float       # cosine distance to the nearest pin
    confidence: float     # from verification.find_confidence (verify its scale)

@dataclass(frozen=True)
class Gallery:
    pins: np.ndarray        # shape (n_pins, 512), float32, L2-normalised
    pin_owner: list[str]    # pin index -> character name (supports mean or per-photo pins)
    names: list[str]        # unique character names, sorted
    meta: dict              # embedding config/version key, strategy, source paths + hashes

@dataclass
class Track:              # used only in M6
    id: int
    box: tuple[int, int, int, int]
    last_seen: int        # frame index
    votes: Counter[str]   # name -> count (Unknown counted as "__unknown__")
```

## 6. Module specs

The original single-file layout is amended by the owner-approved modularization design in
`docs/superpowers/specs/2026-09-26-label-video-modularization-design.md`. The public command
remains `python label_video.py ...`; `label_video.py` becomes a thin executable and
compatibility facade over the `face_labeller/` package. Existing data contracts, CLI flags,
cache formats, matching rules, and output formats remain unchanged.

Module ownership is:

- `face_labeller/contracts.py`: domain records and fixed domain constants.
- `face_labeller/config.py`: `Config`, CLI parsing, validation, and defaults.
- `face_labeller/perception.py`: lazy model loading, DeepFace conversion, and batch fallback.
- `face_labeller/cache.py`: shared cache identity helpers and `FaceCache` persistence.
- `face_labeller/gallery.py`: gallery discovery, cache reconciliation, pins, and diagnostics.
- `face_labeller/recognition.py`: pure cosine distance and matching.
- `face_labeller/tracking.py`: pure IoU and optional stateful association/name voting.
- `face_labeller/rendering.py`: pure deterministic drawing.
- `face_labeller/evidence.py`: CSV logging and optional debug crops.
- `face_labeller/video.py`: video inspection, frame planning, and streaming execution.
- `face_labeller/core.py`: the single end-to-end execution coordinator.

Dependencies point inward toward contracts; no package module imports `label_video.py`, and
lower-level modules never call back into `core.py`.

`load_config(args) -> Config`
  Merge CLI args with defaults (section 7). Record the source of each default in comments.

`build_detector_model() -> Any` and `build_models(cfg) -> None`
  Lazily build resident RetinaFace and Facenet512 models immediately before uncached work
  needs them. Track their load times separately and expose their sum to the video summary.
  A fully compatible cache replay imports/builds neither model.

`embed_faces(frames: list[np.ndarray], cfg) -> list[list[Face]]`
  Reproduce DeepFace's padded RetinaFace resize, crop only grid-aligned black margin with a
  32 px halo, and call the resident RetinaFace model at threshold 0.9 without upscaling.
  Restore boxes/landmarks to full-frame coordinates and align each face locally on the
  original frame through DeepFace's `extract_face`. Flatten all aligned crops in the frame
  batch into one `DeepFace.represent` call with Facenet512 and
  `detector_backend="skip"`; reassemble one ordered list per input frame. Validate unit
  512-dimensional float32 embeddings. If no faces exist, skip Facenet512. Any failure
  propagates to the existing per-frame fallback and retry/cache semantics.

`load_gallery(ref_dir: Path, cfg) -> Gallery`
  Structure: ref_dir/<Character Name>/*.{jpg,jpeg,png}. The folder name is the label.
  Decode each image and pass it through the same optimized `embed_faces` perception path
  as video input, using an immutable config copy with max_faces=1. An empty detection is an
  invalid reference: log a warning and skip the image rather than embedding the whole
  photo.
  Fail loudly if any required character folder has fewer than 2 valid images. Two images
  per character are the initial minimum; additional owner-curated images may be added
  incrementally, with 5-10 varied, clear images per character the target.
  Cache ONE embedding per photo to cache/gallery_<key>.npz (embeddings, owner name, source
  path, file hash). Reconcile the active cache against the current gallery: embed only new
  or changed photos and remove entries for deleted photos without re-embedding unchanged
  ones. The configuration key contains every non-photo upstream embedding input: model,
  detector, normalization, align, max_faces=1, expand_percentage, perception pipeline,
  detector black halo, L2-normalization setting, and installed versions of deepface,
  retina-face, tensorflow and opencv. Gallery cache schema 2 separates the old direct
  DeepFace path from the shared optimized path. Photo hashes live in the cache entries, not
  the configuration key, so gallery additions and removals do not discard unchanged
  embeddings. Compatible keyed caches coexist, so a later return to a previous
  configuration reuses its embeddings.
  Apply the pin strategy at load time, per cfg.pin_strategy ("mean" or "all"):
  "all" uses the per-photo embeddings directly; "mean" averages per character and
  re-normalises. The strategy is NOT part of the cache key.
  Print a per-character report: images found, images used, images skipped.

`FaceCache`
  A per-frame map: frame_idx -> (status, faces). Faces hold box, landmarks, det_conf,
  embedding.
  Key (everything upstream of match, and nothing downstream):
    schema 2 + sha256 of the input video + detector + model + normalization + align +
    max_faces + expand_percentage + perception pipeline + detector black halo + installed
    versions of deepface, retina-face, tensorflow, opencv.
  NOT in the key: stride, start/max frames, threshold, pin strategy, smoothing.
  The full key inputs are stored in the cache meta and checked on load; any mismatch is a
  miss (new cache), never a silent reuse.
  Frame status is explicit:
    ok      processed; faces may be an empty list (a genuine "no faces" frame)
    failed  represent() raised; always retried on the next run
    absent  never processed
  main() asks the cache which of the needed frames are absent or failed, and sends only
  those to embed_faces. Written incrementally so an interrupted run resumes from the last
  completed batch.

`inspect_video(input_path: Path) -> VideoMetadata`
  Open the selected `--input` path without loading a gallery or model; validate capture,
  FPS, dimensions, and frame count; return immutable metadata; and release the capture on
  every path.

`build_frame_plan(metadata, start_frame, max_frames, stride) -> FramePlan`
  Purely calculate the half-open output window and ordered absolute selected indices. The
  first frame in the requested window is always selected. After the compatible FaceCache
  opens, `core.run` reports total/window/written/selected/cached/inference-needed counts
  before gallery or model work. Failed cache entries remain inference-needed; ok empty-face
  entries remain hits.

`cosine_distances(v: np.ndarray, pins: np.ndarray) -> np.ndarray`
  Pure numpy: 1 - pins @ v (both L2-normalised). Returns shape (n_pins,).
  Unit test: must agree with deepface.modules.verification.find_distance(a, b, "cosine")
  on single pairs to 1e-5.

`match(face: Face, gallery: Gallery, threshold: float) -> Match`
  Nearest pin by cosine distance. nearest_name = pin_owner[i] regardless of the threshold.
  If distance < threshold, name = nearest_name, otherwise name = None. confidence via
  verification.find_confidence(distance, "Facenet512", verified, "cosine").

`iou(a, b) -> float` and `Tracker.update(frame_idx, faces, matches) -> list[tuple[Face, Match, int]]`
  (M6 only)
  Greedy IoU association (iou >= cfg.iou_min). New track if unmatched.
  Expire after cfg.track_ttl frames unseen. Displayed name = majority vote over the track's
  history; Unknown wins ties.

`draw(frame, faces, matches, cfg) -> np.ndarray`
  Pure: returns a copy. Box colour is fixed per character (deterministic palette);
  Unknown is grey. Label text on a filled background: "Name 84%" or "Unknown".
  Landmark dots drawn only when cfg.debug_landmarks is set. Keep labels inside the frame.

`MatchLogger`
  Writes a CSV with columns:
  frame_idx, face_idx, x, y, w, h, det_conf, nearest_name, distance, threshold,
  assigned_name, confidence. nearest_name comes directly from Match, including when
  assigned_name is Unknown; the logger never repeats distance calculation.
  With cfg.debug_crops, saves crops to debug/crops/<frame>_<face>_<nearest>_<dist>.png.

`process_video(cfg, gallery=None, *, metadata=None, plan=None, face_cache=None)`
  Reader: cv2.VideoCapture. Read fps, width, height and frame count; fail loudly if not opened.
  When the core supplies preflight metadata, a frame plan, and an open FaceCache, verify the
  streaming capture metadata against the plan and reuse those objects. Direct callers may
  omit all three and receive the same internally planned behavior.
  Writer: cv2.VideoWriter with the same fps and size, fourcc "mp4v".
  Loop: the first frame in the requested window is processed, then every cfg.stride-th
  frame relative to it. Processed frames are grouped into batches of cfg.batch_size for
  embed_faces. If a batch raises, retry its frames individually so one bad frame does not
  discard valid work from the others. Unprocessed frames reuse the most recent annotations.
  EVERY frame is written, so the output has the same frame count and duration as the input.
  The final deliverable uses stride 1; larger strides are development/preview modes.
  Progress: log frames processed, faces found and processing fps every N frames.
  End-of-run summary: total time, fps, faces, and the label distribution.

`core.run(cfg) -> RunSummary`
  Validate cross-path constraints, inspect the input, build the frame plan, open the
  compatible FaceCache, report pending work, load the gallery, invoke `process_video`, and
  print the existing final timing/result summary. It is the only end-to-end execution
  coordinator. `label_video.main()` parses arguments, calls the core, and maps errors to the
  existing process exit codes.

M6 delivery: `scripts/run_full_pipeline.sh` stages the OpenCV video and CSV, re-muxes the
input AAC audio with ffmpeg, verifies video/audio streams with ffprobe, and only then
publishes both artifacts. The README also documents an optional H.264 re-encode for browser
playback.

## 7. CLI and config
```
python label_video.py --input data/video-source/nimbus.mp4 \
  --output output/nimbus_labelled.mp4 --ref-dir data/reference-images/
  [--stride 1] [--batch-size 8] [--threshold <default: find_threshold>]
  [--pin-strategy mean|all (default: all)] [--normalization base|Facenet2018]
  [--max-faces N (default: no cap, per R1)] [--start-frame 0] [--max-frames N]
  [--cache-dir cache/] [--no-cache]    # owner-approved 2026-09-26
  [--smooth] [--iou-min 0.3] [--track-ttl 15]
  [--csv output/matches.csv] [--debug-crops] [--debug-landmarks]
```
`--start-frame/--max-frames` let gates and tests run on a short window (e.g. the first 300
frames) without a full pass. `--cache-dir/--no-cache` control the gallery and FaceCache.
Every default is shown in --help, with its source noted. Stride stays configurable, but the
final deliverable is generated with stride 1 so detection is attempted on every frame.

## 8. Error handling
- Missing input, ref_dir or character folder: exit non-zero with a clear message.
- A reference image with no face: skip it with a warning (never crash).
- A batch that fails in represent(): retry its frames individually. If an individual frame
  still fails, log it, cache it as failed, write it un-annotated, and continue. Failed
  frames are retried on the next run.
- Output directory is created if it is missing.

## 9. Milestones and gates
Every milestone ends in a STOP gate where the owner tests the system. M1-M4 are the vertical
slice and must be provable end-to-end before anything else. Work runs to M5, then halts.
M6 starts only on the owner's explicit command. M7 packages whatever has been approved.

M0 Environment (prerequisite)
  Build: .venv on python3.11, install (including tf-keras), record exact versions, warm the
  model weights.
  Accept: `import deepface` works; a smoke test embeds one video frame on CPU; embedding the
  same frame twice gives identical vectors (determinism check).
  STOP: owner review.

M1 Single frame end-to-end
  Build: build_models, embed_faces, draw with debug_landmarks.
  Accept: one annotated PNG. Boxes are on faces; the 5 dots sit on eyes, nose and mouth
  corners; a face-less frame yields zero boxes.
  STOP: owner review.

M2 Gallery
  Build: load_gallery, cache, per-character report.
  Accept: every character has at least 2 valid images and at least 1 resulting pin; a second
  run hits the cache, does not load Facenet512, and embeds nothing; adding one photo embeds
  only that photo; deleting one photo removes only its cache entry; a leave-one-out sanity
  check (each photo vs the gallery built without it) is reported under BOTH "mean" and
  "all". Additional photos can be added incrementally without re-embedding unchanged ones.
  STOP: owner review (owner decides pin strategy, D1).

M3 Boxes across the full video (no names)
  Build: Reader/Writer loop, stride, batching, FaceCache, progress logging.
  Accept: output frame count and duration match the input; timings reported for
  stride 1 and stride 3 on the first 300 frames; a second run with the same key hits the
  FaceCache, does not load Facenet512 and runs no detection; a stride-1 run after a stride-3
  run computes only the missing frames. Stride 1 is retained for the final deliverable;
  larger strides remain available for development and preview runs.
  STOP: owner review (owner decides batch size and any preferred preview stride, D2; final
  stride 1 is already decided).

M4 Names (completes the vertical slice)
  Build: cosine_distances, match, labels, MatchLogger.
  Before the full run, profile short uncached windows from multiple parts of the clip and
  separate RetinaFace, Facenet512, model-load, and I/O/render costs as far as the installed
  stack permits. Reuse those cached frames. The per-frame RetinaFace path remains the
  correctness baseline. Any event-driven gate or tracking substitute must report its
  missed-face latency/false-negative risk and requires owner approval because it can weaken
  R1; heavyweight models remain loaded while idle rather than restarting per trigger.
  STOP after profiling: owner chooses the per-frame baseline, an R1-preserving measured
  optimization, or a separately specified event-driven trade-off before the full run.
  Accept: unit tests pass; after the runtime checkpoint is reviewed, the full video is
  labelled at stride 1; the CSV is written with nearest_name retained for Unknown
  assignments; relabelling at a different threshold replays from the FaceCache without
  loading Facenet512 or making inference calls.
  Before the full-video run, complete the approved modularization and video-preflight plan
  in `docs/superpowers/plans/2026-09-26-label-video-modularization.md`. Its STOP gate must
  prove CLI/data/cache compatibility and short cold/warm output parity. This is an M4
  refactor checkpoint, not a new product milestone.
  Before the full-video run, execute the owner-approved exact-resize black-margin
  optimization in
  `docs/superpowers/plans/2026-09-26-retinaface-black-margin-optimization.md`. Its 300-frame
  gate must match all 1,262 M3 baseline faces at IoU >= 0.5, manually explain additions,
  enumerate identity changes, beat steady perception wall time by at least 50%, and prove
  a schema-2 warm replay performs zero inference. Stop again for owner review before all
  3,044 frames.
  STOP: owner review. The slice is verified here before any tuning.

M5 Tune from evidence
  Build: an analysis script or notebook over matches.csv and the FaceCache (distance
  histogram by nearest_name); a contact sheet of crops near the threshold; an A/B of
  normalization base vs Facenet2018 on the same sampled frames. This deliberately embeds
  the gallery and sample frames once under the alternate normalization; both keyed cache
  variants are retained, so switching between them does not repeat that work.
  Accept: a written recommendation for the threshold and normalization, with evidence.
  No values are changed without owner approval.
  STOP: owner decides (D3). Work halts here until the owner commands M6.

M6 Polish (optional, owner command only)
  Build: Tracker plus --smooth; ffmpeg audio re-mux and H.264 commands in the README.
  Accept: visibly less label flicker on a before/after clip; tracker unit tests pass.

M7 Package
  README (section 11), pinned requirements, final output video, clean repo.

## 10. Testing
Unit (pytest, no model download needed):
- cosine_distances vs verification.find_distance parity.
- match: synthetic pins, correct nearest name; above threshold -> None.
- match: nearest_name is retained when the assigned name is None.
- iou: identical boxes = 1, disjoint = 0, known overlap value.
- draw: output shape and dtype equal the input, input not mutated.
- Tracker: association, expiry, majority vote, tie -> Unknown.
- Pin strategy: "mean" and "all" derived from the same synthetic per-photo cache.
- Gallery cache: adding or changing one photo embeds only that photo; deleting one removes
  only its entry; changing an upstream embedding input selects a different keyed cache;
  changing the perception pipeline or detector halo selects a different keyed cache;
  changing pin strategy does not.
- Gallery/video consistency: gallery photos call the shared optimized perception boundary
  with max_faces=1; an empty result is skipped and never becomes a whole-photo embedding.
- Lazy model loading: fully cached gallery and FaceCache replay does not call build_model.
- Exact detector input: 1080p produces the pixel-identical 988x576 grid crop; odd,
  portrait, and extreme-aspect frames retain valid symmetric geometry.
- Split perception: direct RetinaFace coordinates map back to the source frame, edge faces
  align locally, and one skip-mode Facenet512 batch preserves frame/detection order.
- FaceCache key: changing the threshold, pin strategy, stride or frame window does not
  change it; changing model, normalization, max_faces, align, perception pipeline, detector
  halo, or a library version does.
- FaceCache status: a zero-face frame is stored as ok and not recomputed; a frame whose
  represent() raises is stored as failed and is retried on the next run.
- FaceCache reuse: after a stride-3 run, a stride-1 run over the same window requests
  only the frames not already cached.
- Batch recovery: one failing frame is isolated by individual retries while the other
  frames in its batch are cached as ok.
- Video preflight: invalid metadata/window values fail before gallery/model work; exact
  selected indices are calculated relative to the requested start; cached ok/failed/absent
  states produce the correct inference-needed set; reopened metadata must match the plan.
- Compatibility facade: established application names continue to resolve from
  `label_video`, while tests patch implementation dependencies in their owning modules.
Integration (marked slow):
- A 10-frame clip runs end-to-end, and the output frame count equals the input's.
- A gallery built from a fixture folder with one face-less image skips that image.

## 11. README must contain
Setup and install; downloading the video (gdown) to data/video-source/; how to build
data/reference-images/ (folder = label, initially at least 2 valid images per character,
growing toward 5-10 clear, varied photos, from the Philosopher's Stone era to match the
clip); run commands; caching behaviour; the
architecture diagram; design rationale (why RetinaFace, Facenet512, cosine, the gallery
approach, stride over downscaling); results summary with runtime and label distribution;
known failure modes; what I'd do with more time.

## 12. Open decisions (OWNER: Eli)
D1 Pin strategy: DECIDED by Eli on 2026-09-26: "all" (per-photo pins, nearest wins).
   M2 leave-one-out gave identical aggregate results for "mean" and "all" with two photos
   per character: nearest identity 10/10, 4 assigned, 6 Unknown, 0 wrong at threshold 0.30.
   "all" preserves both examples instead of averaging their variation. Storage remains
   one entry per photo, and the strategy is applied at load without re-embedding.
D2 Stride / batch size. DECIDED by Eli on 2026-09-26: batch size 8; stride 3 for quick
   engineering smoke tests; stride 2 for reviewable previews; and stride 1 for the final
   output, so detection is attempted on every final frame. At approximately 30 fps,
   stride 2 can carry a box for at most one skipped frame (about 33 ms), while halving the
   selected detection work. Only the embedding pass is batched because detection loops per
   frame, so stride—not batch size—drives runtime. The CLI default remains stride 1.
D3 Threshold and normalization, chosen from the M5 evidence.
   Provisional: 0.30 (library default), base.

## 13. Known failure modes (document them; don't over-engineer)
Profile or turned faces; motion blur; low light; small faces in wide shots;
occlusion (hair, hats, hands); age mismatch between the gallery and the clip;
extras who resemble a lead (mitigated by the threshold and Unknown).

## 14. Definition of done
All deliverables exist; unit tests pass; the stride-1 output video has the same frame count
and duration as the input, with boxes and labels; the README explains how to reproduce it;
decisions D1-D3 are recorded with their evidence.

```
data/reference-images/ ─► load_gallery ─► cache/gallery_<key>.npz (per photo) ─► pins ─┐
nimbus.mp4 ─► Reader ─► embed_faces ─► FaceCache ─► match ─► [Tracker] ─► draw ─► Writer ─► output/
                (RetinaFace+Facenet512)                 └─► MatchLogger ─► matches.csv ─► M5 tuning
gates:  M0 ─► M1 ─► M2 (D1) ─► M3 (D2) ─► M4 [slice verified] ─► M5 (D3) ║ M6 on command ─► M7
```
