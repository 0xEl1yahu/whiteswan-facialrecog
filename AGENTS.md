# AGENTS.md

Guidance for coding agents (Claude Code, Codex, etc.) working in this repo.

## What this is
White Swan Data ML assessment: label every face in a video with a bounding box and a
Harry Potter character name (Harry, Ron, Hermione, McGonagall, Snape) or "Unknown".
Stack is fixed: Python + DeepFace, detector RetinaFace, recogniser Facenet512, cosine distance.

## Core requirements (never lose sight of these)
- **R1:** Draw bounding boxes around **all** the faces in the video. Unknown faces are still
  boxed, and the number of faces per frame is never capped by default. Stride remains
  configurable for development, but the final deliverable uses stride 1 so detection is
  attempted on every frame.
- **R2:** Label each box with the character's name when possible, otherwise "Unknown".

Every milestone must move these forward, and every gate report must say how.

**Read [docs/STATUS.md](docs/STATUS.md) and
[docs/design/design-plan.md](docs/design/design-plan.md) in full before doing any work.**
STATUS is the source of current milestone/artifact state; the design plan is the normative
requirements source. This file only summarises how to work with them.
The approved implementation sequence is
[docs/implementation/implementation-plan.md](docs/implementation/implementation-plan.md).
Read both documents before starting a milestone.

## Operating rules (from docs/design/design-plan.md section 3)
1. Work one milestone at a time (M0-M7, section 9). At each STOP gate, halt and report:
   what was built, test results, timings, open questions. Do not start the next milestone
   until the owner (Eli) approves.
2. Do not change the data contracts (section 5) or the CLI (section 7) without owner approval.
3. Respect the recorded owner decisions: `all` pins, batch size 8, final stride 1, and
   D3's `0.305` threshold with `base` normalization. Flag any choices still awaiting a gate.
4. No magic numbers. Every tunable lives in config with its source noted in a comment.
5. "Verify" means check the installed DeepFace source or write a test. Do not assume.
6. `match`, `cosine_distances`, `draw`, `iou` stay pure: no I/O.
7. Never scrape or download face images. Reference photos are curated by the owner;
   agents only build tooling that validates the gallery.
8. Always pass `detector_backend="retinaface"` and `model_name="Facenet512"` explicitly.
9. Output must be deterministic and must run on CPU.

## Priorities
1. Vertical slice first (M1-M4), provable at every gate. M5/D3 is complete; M6 was
   explicitly approved. M7 remains the final packaging gate.
2. Data efficiency: compute detections/embeddings once, cache by their true upstream inputs,
   and replay. Never load models or re-run inference when compatible gallery and frame data
   are cached. Adding a reference image embeds only that image; changing stride computes
   only missing frame indices; threshold and pin-strategy changes reuse all embeddings.
3. CPU only. Do not add GPU code, flags or benchmarks.

## Repo layout
```
docs/design/design-plan.md   spec (source of truth)
docs/implementation/         approved implementation plans
docs/STATUS.md               current decisions, gallery, artifacts, and milestone
docs/archive/                completed plans/specs and historical investigations
label_video.py               thin CLI and compatibility facade
face_labeller/contracts.py   stable dataclasses and fixed model/name constants
face_labeller/config.py      CLI parser, defaults, and validation
face_labeller/cache.py       cache identity and persistent FaceCache
face_labeller/perception.py  lazy DeepFace models, detection, and embeddings
face_labeller/gallery.py     reference discovery, per-photo cache, and pins
face_labeller/recognition.py pure cosine matching
face_labeller/tracking.py    pure IoU + optional temporal display-name smoothing
face_labeller/rendering.py   pure box, label, and landmark drawing
face_labeller/evidence.py    matches.csv and debug-crop publication
face_labeller/video.py       video preflight, frame plan, and streaming execution
face_labeller/core.py        single end-to-end run coordinator
tests/                       tests patch the module that owns each behavior
data/video-source/nimbus.mp4 input clip, Philosopher's Stone (gitignored)
data/reference-images/<Name>/ owner-curated gallery (gitignored; owner supplies before M2)
cache/                       keyed gallery caches (one embedding/photo) + FaceCache (gitignored)
output/                      labelled video, matches.csv, debug crops (gitignored)
scripts/run_full_pipeline.sh one-shot setup, run, audio restore, and stream verification
```

## Environment
- Python 3.11 (`python3.11`). The default `python3` is 3.14, which TensorFlow does not
  support (wheels exist for 3.10-3.13 only).
- `.venv/` install: `.venv/bin/python -m pip install -r requirements.txt`.
  `tf-keras` is mandatory, or deepface raises at import. M0 pins deepface 0.0.101,
  retina-face 0.0.18, tensorflow/tf-keras 2.21.0, opencv-python 5.0.0.93, numpy 2.4.6,
  gdown 6.4.0 and pytest 9.1.1.
- Spec was audited against deepface 0.0.101 / retina-face 0.0.18. See
  docs/design/design-plan.md
  section 4 for verified model facts (notably: a batch of 1 returns a flat list; embeddings
  are not L2-normalised by default; the RetinaFace 0.9 threshold is not configurable).

## Commands
- `.venv/bin/python -m pytest tests/test_environment.py -m slow -v` for the M0 CPU and
  repeat-embedding smoke test.
- `.venv/bin/python -m pytest tests/test_config.py tests/test_perception.py -v` for M1.
- `.venv/bin/python -m pytest tests/test_face_cache.py tests/test_video_integration.py -v`
  for M3 cache and video integration.
- `.venv/bin/python -m pytest tests/test_matching.py -v` for M4 matching and CSV logging.
- `.venv/bin/python -m pytest tests/test_tracker.py -v` for M6 IoU and smoothing.
- `.venv/bin/python -m pytest tests/test_core.py tests/test_video_plan.py -v` for preflight
  and orchestration.
- `.venv/bin/python -m pytest -m "not slow"` for unit tests.
- `.venv/bin/python -m pytest -m slow` for integration tests.
- `python label_video.py --input ... --output ... --ref-dir ...`

## Current checkpoint
- The optimized 3,044-frame M4 stride-1 run is complete and cached. M5/D3 selected a
  `0.305` threshold with `base` normalization on 2026-09-27.
- M6 was explicitly approved on 2026-09-26. `--smooth` now performs downstream IoU
  tracking without altering raw CSV evidence or cache identity.
- The one-shot runner restores source AAC audio and verifies both streams before publishing.
- The accepted `0.305` output recovered 62 visually confirmed correct labels with no
  observed new wrong name and preserves source AAC audio; M7 verification is complete.
- Preflight reports the exact window, selected indices, cache hits, and frames to infer
  before gallery/model work.
- Final delivery remains stride 1 and CPU only.

## Git
- Commit only when asked. Branch off `main` for milestone work.
- Keep each milestone in its own commit; never squash multiple milestones together.
- Commit a milestone's tests with the implementation they verify. Run that milestone's
  complete test set immediately before the commit and again immediately after it, and
  report both results.
- Give each milestone commit a plain-language one-line summary that a non-technical
  stakeholder can understand.
- Never commit video files, reference photos or model weights.
