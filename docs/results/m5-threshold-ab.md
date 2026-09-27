# M5 threshold A/B: 0.30 versus 0.31

> **Current D3 decision evidence as of `0d272ab`.** Accepted artifact identity and remaining
> delivery work are tracked in [`docs/STATUS.md`](../STATUS.md).

Date: 2026-09-27
Status: D3 approved and verified at threshold `0.305`, normalization `base`

## Question

Would raising the strict cosine-distance threshold from `0.30` to `0.31` recover
correct character names without assigning target-character names to the wrong person?

This comparison changes only the downstream threshold. RetinaFace detections,
Facenet512 embeddings, `base` normalization, the `all` pin strategy, gallery, and
stride-1 frame plan remain fixed.

## Inputs and isolation

- Evidence branch: `feature/m5-threshold-ab`; promoted to `feature/m5-evidence-tuning` in
  commit `0d272ab`.
- Source video SHA-256:
  `67c428ef51ccd86ccbe55a08a6e00efe5540eb6387bb03ff0fd4aade8fbba353`
- Threshold-0.30 CSV SHA-256:
  `a710e4d2735cb531c4517842e3c69d028c7bff0b8749bb16ae7d38e836bee1be`
- Threshold-0.31 CSV SHA-256:
  `e70454003bb0409d1e42b25c8c435826b741fa4409834a6b7a63c98cb3e6a284`

At this A/B stage, the accepted output was not overwritten. The 0.31 video, CSV, review manifest,
contact sheets, and source-context images are isolated under the gitignored
`output/analysis/threshold-ab-030-031/` directory.

## Full cached replay

The candidate used the normal production CLI with threshold `0.31`:

```bash
.venv/bin/python label_video.py \
  --input data/video-source/nimbus.mp4 \
  --output output/analysis/threshold-ab-030-031/threshold-031.mp4 \
  --ref-dir data/reference-images \
  --stride 1 \
  --batch-size 8 \
  --threshold 0.31 \
  --pin-strategy all \
  --normalization base \
  --cache-dir cache \
  --csv output/analysis/threshold-ab-030-031/threshold-031.csv
```

Observed result:

- 3,044 frames selected, cached, and written.
- 3,044 cache hits and zero cache misses.
- Model time `0.000s`; perception time `0.000s`.
- 5,353 faces and zero failed frames.
- Total replay time `24.834s`.
- Geometry, detector confidence, nearest owner, and distance were identical for
  all 5,353 rows.
- Exactly 111 assignments changed, all from `Unknown` at 0.30 to the existing
  nearest owner at 0.31. There were no unexpected changes.

The candidate video was remuxed with the source AAC track without re-encoding.
`ffprobe` confirmed MPEG-4 video plus AAC audio, 3,044 frames, 1920x1080,
29.97 fps, and a 101.568-second duration. Its SHA-256 is
`7d127d7350f1ad00b67a0d9d7a200509f26690cabc2ac01ea1d41910a306e6e8`.

## Aggregate result

| Assignment | Threshold 0.30 | Threshold 0.31 | Change |
| --- | ---: | ---: | ---: |
| Harry Potter | 266 | 321 | +55 |
| Hermione Granger | 198 | 224 | +26 |
| Prof. McGonagall | 34 | 35 | +1 |
| Prof. Severus Snape | 262 | 283 | +21 |
| Ron Weasley | 214 | 222 | +8 |
| Unknown | 4,379 | 4,268 | -111 |

## Exhaustive visual review of changed assignments

Every one of the 111 changed crops was reviewed, with full-frame context used for
occluded and rear-profile cases.

| Candidate | Correct new names | Still Unknown | New wrong names |
| ---: | ---: | ---: | ---: |
| 0.30 | 0 | 111 | 0 |
| 0.305 | 62 | 49 | 0 |
| 0.307 | 81 | 30 | 0 |
| 0.308 | 89 | 21 | 1 |
| 0.31 | 110 | 0 | 1 |

The sole new error is frame 1337, face 1: the visible person is Harry Potter,
but the nearest gallery owner is Ron Weasley at distance `0.307554`. Threshold
tuning cannot correct that nearest-neighbour error; it can only leave it Unknown
or accept the wrong nearest owner.

No non-character extra appears in the 111-row transition cohort. Therefore 0.31
did not create a target-character false positive on an extra in this clip. The
new error is a Harry-to-Ron identity confusion.

## Interpretation

- `0.31` maximizes coverage in the tested interval: 110 correct labels gained
  for one new wrong label.
- `0.305` is the strongest simple conservative candidate: 62 correct labels
  gained with no observed new wrong label.
- `0.307` gains 81 correct labels with no observed new error, but choosing a
  three-decimal boundary immediately below the first observed error would be a
  tighter fit to this single clip.
- `0.30` remains the most conservative unchanged setting, but the earlier reason
  for retaining it—lack of visual evidence for faces that would flip—is now
  resolved for the entire 0.30-to-0.31 transition cohort.

This is exhaustive for changed assignments in this video, but it is still one
clip with correlated adjacent frames and one manual reviewer. The result should
not be presented as an independent estimate of performance on arbitrary videos.

## Recommendation for D3

If wrong-name avoidance is the priority, adopt `0.305` with `base`
normalization: it improves R2 coverage by 62 boxes in this clip without an
observed new error. If maximum named coverage is the priority and one additional
Harry-to-Ron label is acceptable, use `0.31` with `base` normalization.

Eli selected the conservative recommendation: threshold `0.305` with `base`
normalization.

## Approved 0.305 replay

The production default and one-shot runner now use `0.305`. A full production-CLI replay
omitted `--threshold`, proving the configured default rather than an explicit override:

- 3,044/3,044 selected frames were cache hits; model and perception time were `0.000s`.
- The run wrote 3,044 frames and 5,353 face rows with zero failures in `27.889s`.
- All geometry, detector confidence, nearest owner, and distance fields matched the
  accepted 0.30 baseline exactly.
- Exactly 62 assignments changed, all from Unknown to the reviewed correct nearest owner;
  no observed new wrong name was accepted.
- Counts are Harry 298, Hermione 211, McGonagall 35, Snape 276, Ron 216, and Unknown 4,317.
- CSV SHA-256: `bc29006e7da6aa105a0f22ceed1895a36f1003ccb226719c5f83b3fcaddbcb4b`.

The candidate was remuxed with source audio. `ffprobe` confirmed MPEG-4 video and AAC
audio, 3,044 video frames, 1920x1080, 29.97 fps, and 101.568 seconds. Audio-preserved video
SHA-256: `fa37fd119ff5c0fc21d208df9a1efc6c4351dc32d9d48a804a657b9f343968ec`.

After approval, the same verified CSV and audio-preserved video were promoted to the
accepted `output/` paths. Their identity and current milestone state are maintained in
[`docs/STATUS.md`](../STATUS.md).
