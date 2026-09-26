# Character Face Labeller

Draws a box around every face in a video clip from *Harry Potter and the Philosopher's
Stone* and labels it Harry Potter, Ron Weasley, Hermione Granger, Prof. McGonagall,
Prof. Severus Snape, or **Unknown**. It also writes `matches.csv`, a log of every match
decision with its distance and confidence.

Built for the White Swan Data ML assessment. The stack is set by the brief: Python,
[DeepFace](https://github.com/serengil/deepface), RetinaFace detection, Facenet512
embeddings, cosine distance. Everything runs on CPU.

> **Status: design approved, implementation not started.** The full spec is in
> [design-plan.md](design-plan.md). This README is filled in as each milestone lands.
> Results, timings and the pinned versions are added at the gates listed below.

## How it works

```
data/reference-images/ ─► RetinaFace ─► align ─► Facenet512 ─► cache/gallery.npz (one embedding per photo)
                                                                        │ pin strategy applied at load
nimbus.mp4 ─► read frame ─► RetinaFace + Facenet512 ─► FaceCache ─► nearest pin (cosine) ─► draw ─► output/nimbus_labelled.mp4
                                                                        └─► output/matches.csv
```

1. **Gallery.** Each reference photo is detected, aligned and turned into a 512-d
   embedding. Embeddings are cached **one per photo**, so adding a photo re-embeds only
   that photo. Switching between one averaged pin per character (`mean`) and one pin per
   photo (`all`) needs no re-embedding.
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
source .venv/bin/activate
pip install "deepface[tensorflow]" tf-keras opencv-python gdown
```

`tf-keras` is needed because current TensorFlow ships with Keras 3, and DeepFace refuses to
import without the legacy `tf_keras` package. After the first working run, the exact
versions are pinned in `requirements.txt`, and `pip install -r requirements.txt` replaces
the line above.

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
The folder name is the label. For each character, use 5–10 clear, mostly frontal photos
from *Philosopher's Stone*, so the actors are the same age as in the clip. A photo with no
detectable face is skipped with a warning. A character with no usable photos stops the
run.

## Usage

Once built:
```bash
python label_video.py \
  --input data/video-source/nimbus.mp4 \
  --output output/nimbus_labelled.mp4 \
  --ref-dir data/reference-images/
```
Run `python label_video.py --help` to see every option and where each default comes from.
Useful options:
- `--start-frame` and `--max-frames` run on a short window.
- `--threshold` changes the match cut-off.
- `--pin-strategy mean|all` sets how photos become pins.
- `--no-cache` forces a full recompute.

Outputs go to `output/` (the video, `matches.csv`, and optional debug crops). Caches go to
`cache/`. Both folders are gitignored.

## Milestones

Each milestone ends with a stop for the owner to test and review.

| | Milestone | Owner decision |
|---|---|---|
| M0 | Environment, versions, determinism check | |
| M1 | One frame, end to end: boxes and landmarks | |
| M2 | Gallery, per-photo cache, leave-one-out check | D1: pin strategy |
| M3 | Boxes across the full video, FaceCache, timings | D2: stride and batch size |
| M4 | Names and `matches.csv`. **First complete working version, checked here.** | |
| M5 | Tuning evidence: distance histograms, near-threshold crops | D3: threshold and normalisation |
| M6 | *Only on owner command:* smoother labels, audio put back, H.264 encode | |
| M7 | Packaging | |

## Tests

To be added in M1–M4:
```bash
pytest -m "not slow"   # unit tests, no model download
pytest -m slow         # integration tests on a short clip
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

*Added after M4: runtime, frames per second, and how many faces got each label.*
