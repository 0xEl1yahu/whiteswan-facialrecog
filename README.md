# Character Face Labeller

Draws a box around every face in a video clip from *Harry Potter and the Philosopher's
Stone* and labels it Harry Potter, Ron Weasley, Hermione Granger, Prof. McGonagall,
Prof. Severus Snape, or **Unknown**. It also writes `matches.csv`, a log of every match
decision with its distance and confidence.

Built for the White Swan Data ML assessment. The stack is set by the brief: Python,
[DeepFace](https://github.com/serengil/deepface), RetinaFace detection, Facenet512
embeddings, cosine distance. Everything runs on CPU.

> **Status: M3 resumable video box pipeline complete and awaiting owner review.**
> The
> source-of-truth spec is [docs/design/design-plan.md](docs/design/design-plan.md), and the
> milestone sequence is [docs/implementation/implementation-plan.md](docs/implementation/implementation-plan.md).
> This README is filled in as each milestone lands.
> Results, timings and the pinned versions are added at the gates listed below.

## How it works

```
data/reference-images/ ─► RetinaFace ─► align ─► Facenet512 ─► cache/gallery_{key}.npz (one embedding per photo)
                                                                        │ pin strategy applied at load
nimbus.mp4 ─► read frame ─► RetinaFace + Facenet512 ─► FaceCache ─► nearest pin (cosine) ─► draw ─► output/nimbus_labelled.mp4
                                                                        └─► output/matches.csv
```

1. **Gallery.** Each reference photo is detected, aligned and turned into a 512-d
   embedding. Embeddings are cached **one per photo**, so adding a photo re-embeds only
   that photo. The owner-selected default keeps one pin per photo (`all`); switching to one
   averaged pin per character (`mean`) for comparison needs no re-embedding.
2. **Video.** Every `stride`-th frame goes through RetinaFace and Facenet512. The results
   are cached in the FaceCache, so later runs with a new threshold or pin strategy replay
   from the cache and never re-run detection.
3. **Match.** Each face takes the name of its nearest gallery pin if the cosine distance is
   below the threshold (default 0.30, DeepFace's value for Facenet512), and "Unknown"
   otherwise.
4. **Render.** Every input frame is written, so the output has the same frame count and
   duration as the input. Frames skipped by `stride` reuse the last annotations.

## Setup

Requires **Python 3.11**. TensorFlow ships wheels for Python 3.10–3.13 only, so 3.14
will not work.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`tf-keras` is needed because current TensorFlow ships with Keras 3, and DeepFace refuses to
import without the legacy `tf_keras` package. `requirements.txt` pins the M0-verified
Python 3.11 stack: DeepFace 0.0.101, RetinaFace 0.0.18, TensorFlow/`tf-keras` 2.21.0,
OpenCV 5.0.0.93, NumPy 2.4.6, gdown 6.4.0, and pytest 9.1.1.

The Facenet512 and RetinaFace weights download automatically to `~/.deepface/weights/` on
first use.

## Data

Neither the video nor the reference photos are committed.

**Video**
```bash
mkdir -p data/video-source
gdown 1CM1IWUN59ZWml9MwgrvSHXz_9AirIiuU -O data/video-source/nimbus.mp4
```
The clip is 1920×1080, 29.97 fps, 3,044 frames (about 101.6 s), with AAC audio.

**Reference photos**

Photos are chosen by hand. Nothing in this repo scrapes or downloads face images.
```
data/reference-images/
  Harry Potter/        *.jpg | *.jpeg | *.png
  Ron Weasley/
  Hermione Granger/
  Prof. McGonagall/
  Prof. Severus Snape/
```
The folder name is the label. Start with at least two valid photos per character and add
more incrementally, growing toward 5–10 clear, varied photos from *Philosopher's Stone* so
the actors are the same age as in the clip. A photo with no detectable face is skipped with
a warning. Fewer than two usable photos for any required character stops the run.

Gallery embeddings are stored in `cache/gallery_<key>.npz`, one record per photo. The key
contains only model/preprocessing settings and installed library versions. Each record has
its relative path and SHA-256, so unchanged photos are reused while added or modified
photos alone are embedded and deleted photos alone are removed. `mean` and `all` pins are
derived from the same records and never cause re-embedding.

## Usage

Run the current pipeline:
```bash
python label_video.py \
  --input data/video-source/nimbus.mp4 \
  --output output/nimbus_labelled.mp4 \
  --ref-dir data/reference-images/
```
Run `python label_video.py --help` to see every option and where each default comes from.
Useful options:
- `--start-frame` and `--max-frames` run on a short window.
- `--stride 3` gives a quick engineering smoke test. Use `--stride 2` for review previews:
  it halves detection work while carrying boxes across at most one skipped frame. Final
  output remains stride 1 so every frame is inspected.
- `--batch-size` controls the number of selected frames recovered together (default: 8).
- `--threshold` changes the match cut-off.
- `--pin-strategy mean|all` sets how photos become pins (default: owner-selected `all`).
- `--no-cache` forces a full recompute.

Outputs go to `output/` (the video, `matches.csv`, and optional debug crops). Caches go to
`cache/`. Both folders are gitignored.

## Milestones

Each milestone ends with a stop for the owner to test and review.

| | Milestone | Owner decision |
|---|---|---|
| M0 | Environment, versions, determinism check | |
| M1 | One frame, end to end: boxes and landmarks | |
| M2 | Gallery, per-photo cache, leave-one-out check | D1: `all` selected |
| M3 | Boxes across the full video, FaceCache, timings | D2: batch 8; smoke stride 3; preview stride 2; final stride 1 |
| M4 | Names and `matches.csv`. **First complete working version, checked here.** | |
| M5 | Tuning evidence: distance histograms, near-threshold crops | D3: threshold and normalisation |
| M6 | *Only on owner command:* smoother labels, audio put back, H.264 encode | |
| M7 | Packaging | |

## Tests

```bash
.venv/bin/python -m pytest tests/test_environment.py -m slow -v
.venv/bin/python -m pytest -m "not slow"
.venv/bin/python -m pytest -m slow
```

## Design notes

- **Stride, not downscaling.** RetinaFace resizes every frame so its short side is
  1024 px, so shrinking frames first saves nothing. Processing fewer frames (`stride`) is
  the only real speed-up. Detection runs one frame at a time inside DeepFace, and only the
  embedding step is batched.
- **Cosine distance on unit-length embeddings.** DeepFace does not normalise embeddings
  to unit length by default. The pipeline asks for normalised output and checks it.
- **"Unknown" is a valid answer.** Leaving a face unlabelled is better than naming the
  wrong character. The threshold is set in M5 from measured distances, not guessed.

## Known failure modes

Profile or turned faces, motion blur, low light, small faces in wide shots, faces partly
covered by hair, hats or hands, and extras who resemble a lead character. The results
section, added after M4 and M5, will report how often each happens in this clip.

## Results

### M0 environment gate

- Python 3.11.15; CPU device available and no TensorFlow GPU device exposed.
- DeepFace, RetinaFace, Facenet512, TensorFlow/`tf-keras`, OpenCV, NumPy, gdown, and pytest
  import successfully; `pip check` reports no broken requirements.
- Facenet512 and RetinaFace weights are warmed in DeepFace's normal user cache.
- Video frame 150 contains two real RetinaFace detections. Repeating Facenet512 embedding
  on that same frame produced bit-identical boxes and embedding arrays.
- Warm run timings: Facenet512 load 1.406 s; first detection/embedding 5.570 s; repeated
  detection/embedding 2.074 s.

### M1 single-frame gate

- Frame 150 produced two boxes, each labelled Unknown before gallery matching exists.
- RetinaFace's five landmarks align with the eyes, nose, and mouth corners on both faces.
- Frame 0 produces zero boxes after filtering DeepFace's confidence-zero placeholder.
- Model load: 6.262 s; batched perception of the face frame plus face-less frame: 9.324 s.
- Artifact: `output/m1_single_frame.png` (generated and gitignored).

### M2 gallery gate

- All five canonical folders contain two usable photos: 10 photos total, with no skips.
- A cold real-gallery build took 27.520 s. It produced five `mean` pins and persisted ten
  per-photo embeddings.
- A warm run took 0.007 s and was verified with model access disabled: no Facenet512 load
  and no detection or embedding call occurred.
- In an isolated copy of the curated gallery, adding one photo made exactly one model load
  and one embedding call (5.331 s). Removing it took 0.007 s, removed only that record,
  and made no model call.
- Leave-one-out at the provisional 0.30 threshold produced the same result for `mean` and
  `all`: nearest identity was correct for 10/10 photos, 4 were assigned correctly, 6 were
  Unknown, and 0 were assigned the wrong identity. With only two photos per character,
  holding one out leaves one same-character photo, so this evidence does not yet
  distinguish the two strategies.
- D1: Eli selected `all`, preserving both reference examples as separate pins. The dated
  baseline is retained in `docs/results/m2-gallery-baseline.md` for future comparisons.

### M3 video and FaceCache gate

- The first 300 frames were written at 1920×1080 and 29.97 fps. The output contains 300
  frames and lasts 10.010 s, matching that input window.
- A cold stride-3 preview processed 100 selected frames in 279.678 s, found 418 faces,
  and cached all 100 results with no failures.
- A stride-1 continuation reused those 100 entries and computed only the 200 gaps in
  536.171 s. Together, the two runs populated each of the first 300 frames exactly once;
  they found 1,262 face detections with no failed frames.
- A fully warm stride-1 replay wrote all 300 frames in 2.724 s with 300 cache hits, zero
  misses, zero model-load time, and zero perception time.
- A one-frame uncached timing probe measured 6.014 s of lazy model loading inside 10.888 s
  of perception. The full cold stride-1 run was deliberately not repeated after the cache
  was complete, avoiding 300 redundant RetinaFace calculations.
- Batch size 8 completed without recovery or memory failure. RetinaFace still detects one
  frame at a time, so stride drives runtime. D2 is decided: stride 3 for quick smoke tests,
  stride 2 for review previews, and stride 1 for final output.
- The dated baseline is retained in `docs/results/m3-video-cache-baseline.md`.

*Runtime, frames per second, and label distribution are added after M4.*
