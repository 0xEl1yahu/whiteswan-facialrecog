# Character Face Labeller

Draws a box around every face in a video clip from *Harry Potter and the Philosopher's
Stone* and labels it Harry Potter, Ron Weasley, Hermione Granger, Prof. McGonagall,
Prof. Severus Snape, or **Unknown**. It also writes `matches.csv`, a log of every match
decision with its distance and confidence.

Built for the White Swan Data ML assessment. The stack is set by the brief: Python,
[DeepFace](https://github.com/serengil/deepface), RetinaFace detection, Facenet512
embeddings, cosine distance. Everything runs on CPU.

> **Status: M0–M7 implementation and verification are complete on `main`.**
> [Current project status](docs/STATUS.md) records the active decisions and accepted
> artifact hashes. The normative spec is [docs/design/design-plan.md](docs/design/design-plan.md),
> and the only live checklist is
> [docs/implementation/implementation-plan.md](docs/implementation/implementation-plan.md).

## How it works

```
data/reference-images/ ─► RetinaFace ─► align ─► Facenet512 ─► cache/gallery_{key}.npz (one embedding per photo)
                                                                        │ pin strategy applied at load
nimbus.mp4 ─► read frame ─► RetinaFace + Facenet512 ─► FaceCache ─► nearest pin (cosine) ─► draw ─► output/nimbus_labelled.mp4
                                                                        └─► output/matches.csv
```

1. **Gallery.** Each reference photo uses the same optimized detection, local alignment,
   and embedding path as video faces, capped to the largest face for that photo. Embeddings
   are cached **one per photo**, so adding a photo re-embeds only that photo. The
   owner-selected default keeps one pin per photo (`all`); switching to one averaged pin
   per character (`mean`) for comparison needs no re-embedding.
2. **Video.** Every `stride`-th frame uses direct RetinaFace detection on an exact resized
   crop with redundant black margin removed, then locally aligns faces and batch-embeds the
   crops with Facenet512. Results are cached in the FaceCache, so later threshold or pin
   changes replay without either model.
3. **Match.** Each face takes the name of its nearest gallery pin if the cosine distance is
   below the threshold (owner-selected default 0.305), and "Unknown" otherwise. DeepFace's
   underlying Facenet512/cosine library default is 0.30.
4. **Render.** Every input frame is written, so the output has the same frame count and
   duration as the input. Frames skipped by `stride` reuse the last annotations.

Before loading the gallery or either model, the CLI inspects the input and prints the
total frames, requested window, frames written, selected frames, reusable cache entries,
and frames still requiring inference. Supplying a video path is therefore enough to
calculate the exact work plan up front.

### Code structure

`label_video.py` is a thin CLI and backwards-compatible import facade. Implementation is
split by responsibility under `face_labeller/`:

- `contracts.py` and `config.py`: stable data contracts and CLI configuration.
- `cache.py`, `perception.py`, and `gallery.py`: cache identity/storage, DeepFace model
  work, and owner-curated gallery preparation.
- `recognition.py`, `rendering.py`, and `evidence.py`: pure matching, drawing, and CSV/debug
  evidence output.
- `video.py`: metadata inspection, frame planning, and bounded streaming execution.
- `core.py`: the single run coordinator that performs preflight before gallery/model work.
- `analyse_matches.py`: model-free M5 evidence analysis over CSVs, cache metadata, and
  source-video crops.

Tests and extensions should import or patch the owning module. Existing imports from
`label_video` remain supported for the public API.

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

The owner-curated reference gallery is tracked in Git for handoff. The source video,
model weights, caches, and generated outputs stay out of Git.

**Video**
```bash
mkdir -p data/video-source
.venv/bin/python -m gdown 1CM1IWUN59ZWml9MwgrvSHXz_9AirIiuU -O data/video-source/nimbus.mp4
```
The clip is 1920×1080, 29.97 fps, 3,044 frames (about 101.6 s), with AAC audio.

**Reference photos**

Photos are chosen by hand under `data/reference-images/`. A fresh checkout includes the
17 photos used for the accepted result. The
[source note](data/reference-images/REFERENCE_SOURCES.md) records the original examples;
later additions are not fully sourced there. Nothing in this repo scrapes or downloads
face images.
```
data/reference-images/
  Harry Potter/        *.jpg | *.jpeg | *.png
  Ron Weasley/
  Hermione Granger/
  Prof. McGonagall/
  Prof. Severus Snape/
```
The folder name is the label. The pipeline reads `.jpg`, `.jpeg`, and `.png` files; other
image formats in those folders are not used. Keep at least two valid photos per character
and add more incrementally, growing toward 5–10 clear, varied photos from
*Philosopher's Stone* so the actors are the same age as in the clip. A supported photo
with no detectable face is skipped with a warning. Fewer than two usable photos for any
required character stops the run.

Gallery embeddings are stored in `cache/gallery_<key>.npz`, one record per photo. Schema 2
keys the model, normalization/alignment settings, optimized perception-pipeline name,
32 px detector halo, and installed library versions. Each record has its relative path and
SHA-256, so unchanged photos are reused while added or modified photos alone are embedded
and deleted photos alone are removed. `mean` and `all` pins are derived from the same
records and never cause re-embedding.

Video FaceCache schema 2 includes the perception-pipeline name and 32 px detector halo.
The first optimized run therefore creates a new `faces_*.npz`; schema-1 files remain on
disk but are not loaded as optimized results. The gallery's schema-2 correction likewise
leaves older cache files on disk but does not load their old-path embeddings. Gallery-only
changes never invalidate compatible cached video detections or embeddings.

## Usage

For a fresh macOS/Linux checkout, install Python 3.11 and `ffmpeg` (which includes
`ffprobe`), then run the complete setup and full stride-1 pipeline with:

```bash
./scripts/run_full_pipeline.sh
```

The launcher locates the repository root, requires Python 3.11, creates `.venv`, installs
the pinned dependencies, downloads the fixed Nimbus source video if it is absent, validates
the five reference folders already in the checkout, and runs the approved CPU-only command.
`ffmpeg` and `ffprobe` must be available on `PATH`: the launcher restores the source AAC
track, verifies that the staged result has video and audio streams, and only then publishes
the video and CSV. A failed pipeline or remux preserves the previously published artifacts.
It is safe to rerun:
the existing environment, source video, gallery cache, and FaceCache are reused. Reference
photos are never downloaded by the runner. Set `PYTHON_BIN=/path/to/python3.11` only when
Python 3.11 is not available as `python3.11` on `PATH`.

### Dry run: 30 output frames

After [setup](#setup) and the [video download](#data), use this to check the gallery,
CLI, detection, labels, and CSV without processing the full clip. It writes one second of
video starting at frame 190, where Snape appears, and attempts detection on 10 selected
frames. The other 20 frames reuse the latest boxes, so this is a smoke check rather than
the final R1 result. The direct CLI writes video without source audio.

```bash
.venv/bin/python label_video.py \
  --input data/video-source/nimbus.mp4 \
  --output output/dry-run.mp4 \
  --ref-dir data/reference-images \
  --start-frame 190 --max-frames 30 --stride 3 \
  --batch-size 8 --cache-dir cache \
  --csv output/dry-run.csv
```

### Test run: 300 output frames

This processes the first 10 seconds at stride 1. It attempts detection on every frame in
that window, reusing the dry run's 10 cached results and computing the 290 missing frame
indices. This is a video pipeline check, separate from the automated tests below. Check
`output/test-run.mp4` and `output/test-run.csv` before starting the full run.

```bash
.venv/bin/python label_video.py \
  --input data/video-source/nimbus.mp4 \
  --output output/test-run.mp4 \
  --ref-dir data/reference-images \
  --max-frames 300 --stride 1 \
  --batch-size 8 --cache-dir cache \
  --csv output/test-run.csv
```

### Full handoff run: 3,044 output frames

Run `./scripts/run_full_pipeline.sh`. It uses stride 1 and the same `cache/`, so the 300
test frames are reused. It publishes `output/nimbus_labelled.mp4` with copied AAC audio
and `output/matches.csv` after verifying both streams. This is the final R1/R2 run.
The script installs the pinned requirements each time; the processing times below exclude
dependency installation, video download, and first-time model-weight download.

### CPU timings and benchmark

These are elapsed measurements on an Apple MacBook Air M4. The dry and test rows were
measured on 2026-09-29 with the current 17-photo gallery and a separate fresh cache. The
full-clip rows are the recorded M4 cold benchmark and accepted M5 cached replay. CPU speed,
scene density, and cache state change runtime; budget additional time for dependency,
source-video, and model-weight downloads on a fresh machine.

| Run | Frames written / inferred | Cache state | Measured time |
| --- | ---: | --- | ---: |
| Dry run, frames 190–219, stride 3 | 30 / 10 | Fresh gallery and frame cache | 1m05s (64.58s) |
| Test run, frames 0–299, stride 1 | 300 / 290 | Reused the dry run's 10 frames | 6m27s (387.32s) |
| Full benchmark, stride 1 | 3,044 / 3,044 | Historical cold frame run; gallery cached | 33m03s (1,983.09s) |
| Full accepted replay, stride 1 | 3,044 / 0 | All 3,044 frames cached | 24.989s pipeline render |

The dry run produced 237 face rows; the test run produced 1,265. The full cold benchmark
used the earlier two-photo-per-character gallery; video detection is independent of gallery
size. The accepted `0.305` replay with the current gallery retained 5,353 face rows. The
full-run shell script also installs requirements and remuxes AAC, so its total wall time
will exceed the pipeline figures in the table. Running dry → test → full with the same
`cache/` reuses completed inference at each step.

Run `.venv/bin/python label_video.py --help` to see every option and its default source.
Useful options:
- `--start-frame` and `--max-frames` run on a short window.
- `--stride 3` gives a quick engineering smoke test. Use `--stride 2` for review previews:
  it halves detection work while carrying boxes across at most one skipped frame. Final
  output remains stride 1 so every frame is inspected.
- `--batch-size` controls the number of selected frames recovered together (default: 8).
- `--threshold` changes the strict match cut-off (default: evidence-selected `0.305`).
- `--pin-strategy mean|all` sets how photos become pins (default: owner-selected `all`).
- `--no-cache` forces a full recompute.
- `--csv` sets the evidence file (default: `output/matches.csv`).
- `--debug-crops` saves clipped original-frame face crops beside the CSV under `debug/crops/`.
- `--smooth` enables M6 temporal label smoothing. Greedy IoU tracking uses `--iou-min 0.3`
  and expires a track after `--track-ttl 15` unseen frames by default. It changes rendered
  names only; detection, embeddings, and the raw CSV assignments remain unchanged.

Outputs go to `output/` (the video, `matches.csv`, and optional debug crops). Caches go to
`cache/`. Both folders remain gitignored; the reference gallery is tracked.
The CSV writes one row for each displayed face on each output frame, with frame and face
indices, box, detection confidence, nearest pin owner and distance, threshold, assigned
name, and match confidence. Rejected matches display `Unknown` while retaining the
nearest pin owner for later review. With smoothing enabled, this remains the unsmoothed
evidence rather than silently rewriting the model's frame-level decision. A successful run
replaces its CSV; a failed run preserves the prior file. The CLI reports total elapsed time
and gallery/model-load time separately. Compatible gallery and frame caches let a threshold,
tracking, or pin-strategy change relabel the video without running the models.

### Manual audio and H.264 delivery

The Python/OpenCV writer produces video only. The one-shot launcher automatically copies
the source AAC audio into that labelled stream without re-encoding either stream. For a
browser-oriented H.264 copy of the accepted output, run:

```bash
ffmpeg -i output/nimbus_labelled.mp4 \
  -map 0:v:0 -map 0:a:0 -c:v libx264 -preset medium -crf 18 \
  -pix_fmt yuv420p -c:a copy -shortest output/nimbus_labelled.h264.mp4

ffprobe -v error \
  -show_entries stream=index,codec_type,codec_name,duration,r_frame_rate \
  -show_entries format=duration -of json output/nimbus_labelled.h264.mp4
```

This re-encodes only the labelled video and copies the existing AAC track. `ffprobe` should
show one H.264 video stream and one AAC audio stream with matching approximately 101.6 s
durations for the full Nimbus clip.

### M5 threshold and normalization evidence

M5 analyses the accepted stride-1 CSV without loading RetinaFace, Facenet512, or
TensorFlow. Supply the explicit schema-2 cache files printed by preflight; do not select a
cache with an ambiguous wildcard:

```bash
.venv/bin/python analyse_matches.py baseline \
  --csv output/matches.csv \
  --video data/video-source/nimbus.mp4 \
  --face-cache cache/<recorded-base-face-cache>.npz \
  --gallery-cache cache/<recorded-base-gallery-cache>.npz \
  --output-dir output/analysis \
  --supplemental-keys output/analysis/pre-registered-gallery-flips.csv \
  --threshold 0.30 \
  --margin 0.05 \
  --per-character-limit 30 \
  --bin-edges 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.75 1.00 2.00
```

The outputs are `baseline-summary.json`, per-nearest-character histogram CSV data, a
review manifest, a labelled near-threshold contact sheet, and a separate detector-miss
log under `output/analysis/`, plus `output/tuning-report.md`. They are private generated
evidence and remain gitignored. Human review distinguishes correct names, known characters
rejected as Unknown, wrong names, true extras, uncertain crops, and detector misses.

Normalization is compared on six pre-registered 50-frame windows: 80-129, 190-239,
1120-1169, 1560-1609, 2290-2339, and 2920-2969. Run each window through the normal
`label_video.py` CLI once with `--normalization base` and once with
`--normalization Facenet2018`, writing every video and CSV below separate
`output/analysis/<normalization>/` paths. These disposable sample videos do not need audio;
the accepted full output is not replaced. Repeat the candidate commands to prove their
configuration-keyed cache is warm before comparing the CSVs with `analyse_matches.py`.

Threshold `0.30` and normalization `base` were the frozen inputs for this analysis. The
analysis did not change production defaults automatically; the owner subsequently chose
`0.305` with `base` at the M5/D3 STOP gate.

The initial completed M5 evidence recommended retaining both provisional values. The 190-crop
threshold review found 109 correct names, 78 known characters labelled Unknown, two
wrong-name assignments, and one uncertain crop. It contains no confirmed non-character
faces, so raising the threshold cannot be safety-scored. On the fixed 300-frame A/B,
Facenet2018 matched all 767 base detections at IoU 1.0 but lost 72 correct labels and gained
14, a net loss of 58. The cold candidate run took 331.746 seconds; a warm replay took 3.670
seconds with no perception inference. See the gitignored `output/tuning-report.md` and
`output/analysis/threshold-sweep.csv` for the hashes, cache identities, review findings,
and full trade-off table. The later exhaustive threshold review supplied the evidence for
the final D3 choice.

A follow-up exhaustive threshold A/B replay compared `0.30` with `0.31` over the full
cached video. All 5,353 detections were identical and exactly 111 Unknown assignments
changed: visual review found 110 correct new names and one new Harry-to-Ron error. A
`0.305` replay accepts 62 of those correct names without accepting the observed error.
Eli selected `0.305` with `base`; the default, one-shot runner, accepted cache-only replay,
and audio-preserved output validation are recorded in
[the M5 threshold A/B report](docs/results/m5-threshold-ab.md).

## Milestones

Each milestone ends with a stop for the owner to test and review.

| | Milestone | Owner decision |
|---|---|---|
| M0 | Environment, versions, determinism check | |
| M1 | One frame, end to end: boxes and landmarks | |
| M2 | Gallery, per-photo cache, leave-one-out check | D1: `all` selected |
| M3 | Boxes across the full video, FaceCache, timings | D2: batch 8; smoke stride 3; preview stride 2; final stride 1 |
| M4 | Names and `matches.csv`. **First complete working version, checked here.** | |
| M5 | Tuning evidence: distance histograms, near-threshold crops | D3: `0.305`, `base` selected 2026-09-27 |
| M6 | Optional temporal smoothing, automatic audio restoration, H.264 guidance | Owner approved 2026-09-26 |
| M7 | Packaging | Verified and merged to `main` in PR #3 |

Threshold `0.305` and `base` normalization are the production defaults. The accepted
stride-1 output was verified before the M7 merge; the owner's later handoff decision placed
the gallery in the repository.

## Tests

```bash
.venv/bin/python -m pytest tests/test_environment.py -m slow -v
.venv/bin/python -m pytest tests/test_matching.py -v
.venv/bin/python -m pytest tests/test_analysis.py -v
.venv/bin/python -m pytest tests/test_tracker.py -v
.venv/bin/python -m pytest tests/test_core.py tests/test_video_plan.py -v
.venv/bin/python -m pytest -m "not slow"
.venv/bin/python -m pytest -m slow
```

## Design notes

- **Remove black work, not image detail.** Naive unpadded downscaling missed small faces and
  was rejected. The approved path reproduces DeepFace's padded RetinaFace resize exactly,
  then removes only grid-aligned black margin while retaining a 32 px halo. A 1080p frame's
  detector tensor falls from 1820x1024 to 988x576; aligned crops are batch-embedded.
- **Cosine distance on unit-length embeddings.** DeepFace does not normalise embeddings
  to unit length by default. The pipeline asks for normalised output and checks it.
- **"Unknown" is a valid answer.** Leaving a face unlabelled is better than naming the
  wrong character. The threshold is set in M5 from measured distances, not guessed.

## Known failure modes

Profile or turned faces, motion blur, low light, small faces in wide shots, faces partly
covered by hair, hats or hands, and extras who resemble a lead character. The results
below and the linked M4/M5 reports document the observed effects in this clip.

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

The detailed M4 hotspot timings and runtime projections are retained in the historical
`docs/results/m4-perception-profile.md`; the measured full-video result follows below.

### M4 black-margin optimization improvement

The production path now removes redundant black detector margin while preserving the
same resized image pixels, face scale, and RetinaFace feature-grid phase. It then aligns
the detected faces locally and sends all crops in the group through one Facenet512 batch.

| Measure | Before | Optimized path | Improvement |
| --- | ---: | ---: | ---: |
| RetinaFace tensor for a 1920x1080 frame | 1820x1024, 75% black | 988x576, 32 px halo | 69.5% fewer detector pixels |
| 300-frame cold perception | 815.849 s | 265.390 s | 67.5% lower; 3.1x throughput |
| Per selected frame | 2.719 s | 0.885 s | 1.834 s saved |
| Faces detected | 1,262 | 1,265 | 0 lost; 3 real faces added |
| Full cold clip | 2 h 17 m historical estimate | 33 m 03 s measured | about 1 h 44 m saved |
| Fully warm 300-frame replay | 2.724 s | 2.707 s | Still zero model/perception work |

All 1,262 baseline faces matched optimized boxes at IoU >= 0.5 (mean 0.956), and their
embeddings had mean cosine similarity 0.978. Five of 1,262 assignments moved across the
then-provisional 0.30 threshold; all were the correct visible Harry or Snape and were within
0.0188 of the threshold. M5 later selected 0.305 from full-clip evidence.

The owner-approved full run then processed all 3,044 frames at stride 1 in 33m03s. It found
5,353 faces with zero failed frames: 648 received a character label and 4,705 remained
Unknown at the provisional 0.30 threshold. The output video matches the source frame count,
resolution, FPS, and duration. The full result, character timelines, spatial summary,
visual-review findings, validation, and hashes are in
[the M4 full-run report](docs/results/m4-full-run.md); the preceding comparison evidence
remains in [the optimization gate](docs/results/m4-retinaface-black-margin-optimization.md).

Harry's low seven-label count was traced to insufficient gallery coverage rather than the
detector: inspected Harry faces were boxed, but 780 probable Harry detections in a
full-cache diagnostic remained Unknown. The two baseline Harry references are themselves
0.372545 apart, above the 0.30 threshold. The owner subsequently added left/right
three-quarter and profile references before M5 tuning. The historical diagnosis and tested
audio-remux path are in
[the Harry discrepancy investigation](docs/results/m4-harry-discrepancy-investigation.md).

The owner subsequently added two Harry views and one Hermione view. The incremental gallery
refresh embedded only those three images. A 12.839 s full-cache replay raised Harry from 7
to 255 assignments and Hermione from 152 to 174, with all 270 changes moving from Unknown
to the intended owner and no existing named assignment displaced. Those figures are the
historical pre-consistency result.

The gallery/video consistency correction then re-embedded all 13 reference photos through
the optimized video path while reusing all 3,044 video-frame cache entries. The regenerated
output retains 5,353 rows and 4,435 Unknown results: Harry 266, Hermione 168, McGonagall 34,
Snape 262, and Ron 188. Seventy-four borderline assignments changed; visual review found
37 correct named gains, 36 correct labels moving to Unknown, and one incorrect Harry-on-Ron
label corrected to Unknown. No new wrong-name assignment appeared. Full historical
measurements, changed frames, media validation, and then-current hashes are in
[the gallery/video consistency report](docs/results/m4-gallery-perception-consistency.md).

After that historical correction, the owner added one more Hermione view and three more
Ron views. The current supported gallery therefore contains 17 photos: Harry 4, Hermione
4, Ron 5, McGonagall 2, and Snape 2. A cache-only replay retains all 5,353 detections and
now assigns Harry 266, Hermione 198, Ron 214, McGonagall 34, Snape 262, and Unknown 4,379.
These are the frozen inputs to M5; the linked consistency report remains the earlier
13-photo measurement rather than silently rewriting historical evidence.

The accepted D3 replay now uses threshold `0.305` with the same 17-photo gallery. It keeps
all 5,353 detections and assigns Harry 298, Hermione 211, McGonagall 35, Snape 276, Ron
216, and Unknown 4,317. The video contains all 3,044 frames plus copied AAC audio. Current
artifact hashes and media validation live only in [docs/STATUS.md](docs/STATUS.md).

### M6 temporal smoothing and delivery

The cache-backed Harry window at frames 2263–2282 rendered in 0.318 s without smoothing
and 0.234 s with smoothing, with 20/20 FaceCache hits, no model loading, and no perception.
Raw assignments contained 6 Harry labels and 14 Unknown labels; smoothing rendered 9 Harry
labels and 11 Unknown labels while producing byte-identical CSV evidence. Frame 2268 is a
direct visual example: the isolated raw `Unknown` flicker is carried as Harry by the track.
Across the whole refreshed clip, however, lifetime majority voting was too sticky and
reduced named coverage (for example Snape 275 → 60), so smoothing remains opt-in and is not
enabled by the one-shot runner. The complete evidence, limitations, audio verification,
then-current artifact hashes, and review paths are in
[the historical M6 gate report](docs/results/m6-temporal-smoothing-and-audio.md). Current
unsmoothed artifact hashes are in
[docs/STATUS.md](docs/STATUS.md).
