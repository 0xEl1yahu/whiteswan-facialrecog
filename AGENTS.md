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

**The spec is [design-plan.md](design-plan.md). Read it in full before doing any work.**
It is the source of truth; this file only summarises how to work with it.

## Operating rules (from design-plan.md section 3)
1. Work one milestone at a time (M1-M7, section 9). At each STOP gate, halt and report:
   what was built, test results, timings, open questions. Do not start the next milestone
   until the owner (Eli) approves.
2. Do not change the data contracts (section 5) or the CLI (section 7) without owner approval.
3. Do not resolve the OWNER decisions D1-D3 (section 12). Use the provisional default and flag it.
4. No magic numbers. Every tunable lives in config with its source noted in a comment.
5. "Verify" means check the installed DeepFace source or write a test. Do not assume.
6. `match`, `cosine_distances`, `draw`, `iou` stay pure: no I/O.
7. Never scrape or download face images. Reference photos are curated by the owner;
   agents only build tooling that validates the gallery.
8. Always pass `detector_backend="retinaface"` and `model_name="Facenet512"` explicitly.
9. Output must be deterministic and must run on CPU.

## Priorities
1. Vertical slice first (M1-M4), provable at every gate. Work runs to M5 then halts;
   M6 only on the owner's explicit command.
2. Data efficiency: compute detections/embeddings once, cache by their true upstream inputs,
   and replay. Never load models or re-run inference when compatible gallery and frame data
   are cached. Adding a reference image embeds only that image; changing stride computes
   only missing frame indices; threshold and pin-strategy changes reuse all embeddings.
3. CPU only. Do not add GPU code, flags or benchmarks.

## Repo layout
```
design-plan.md               spec (source of truth)
label_video.py, tests/       to be built
data/video-source/nimbus.mp4 input clip, Philosopher's Stone (gitignored)
data/reference-images/<Name>/ owner-curated gallery (gitignored; owner supplies before M2)
cache/                       keyed gallery caches (one embedding/photo) + FaceCache (gitignored)
output/                      labelled video, matches.csv, debug crops (gitignored)
```

## Environment
- Python 3.11 (`python3.11`). The default `python3` is 3.14, which TensorFlow does not
  support (wheels exist for 3.10-3.13 only).
- `.venv/` install: `pip install "deepface[tensorflow]" tf-keras opencv-python gdown`.
  `tf-keras` is mandatory, or deepface raises at import. Pin versions after the first run.
- Spec was audited against deepface 0.0.101 / retina-face 0.0.18. See design-plan.md
  section 4 for verified model facts (notably: a batch of 1 returns a flat list; embeddings
  are not L2-normalised by default; the RetinaFace 0.9 threshold is not configurable).

## Commands
None yet. Add build/test/run commands here as milestones land, e.g.:
- `pytest -m "not slow"` for unit tests (no model download)
- `pytest -m slow` for integration tests
- `python label_video.py --input ... --output ... --ref-dir ...`

## Git
- Commit only when asked. Branch off `main` for milestone work.
- Never commit video files, reference photos or model weights.
