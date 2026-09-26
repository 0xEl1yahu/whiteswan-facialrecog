# M6 Temporal Smoothing and Audio Delivery

Date: 2026-09-26
Branch: `investigation/video-runtime`

## Outcome

M6 is implemented as a downstream, opt-in presentation stage. It does not change face
detection, Facenet512 embeddings, gallery pins, matching, or either cache identity.
`matches.csv` remains the unsmoothed frame-level evidence. Only the name passed to the
renderer changes when `--smooth` is enabled.

The one-shot runner now also stages the video-only output and CSV, copies the source AAC
stream with ffmpeg, verifies video and audio streams with ffprobe, and publishes the staged
artifacts only after verification succeeds. A remux failure leaves the previous video and
CSV in place.

The owner then added two Harry references and one Hermione reference. All three validated
and embedded incrementally; the ten prior gallery photos and all 3,044 cached frame
detections/embeddings were reused.

## Gallery refresh result

Gallery counts are now Harry 4, Hermione 3, and 2 each for McGonagall, Snape, and Ron. The
incremental refresh took 6.854 s, including 3.938 s of model load, and performed no video
perception.

The full unsmoothed relabel replayed all 3,044 cached frames in 12.839 s:

| Raw assignment | Two-reference baseline | Refreshed gallery | Change |
| --- | ---: | ---: | ---: |
| Harry Potter | 7 | 255 | +248 |
| Hermione Granger | 152 | 174 | +22 |
| Prof. McGonagall | 34 | 34 | 0 |
| Prof. Severus Snape | 275 | 275 | 0 |
| Ron Weasley | 180 | 180 | 0 |
| Unknown | 4,705 | 4,435 | -270 |

Every change was `Unknown` to the intended owner: 248 to Harry and 22 to Hermione, with no
existing named assignment displaced. All 248 Harry gains chose the new side-profile image
`a519c18a91bbae1b0e2c6f448d02a51e.jpg` as their nearest pin. A 12-frame contact sheet
sampled those gains across the clip and showed Harry in every sampled red box. The 22
Hermione gains form one coherent sequence at frames 759–788.

The two new Harry pins are deliberately pose-diverse rather than embedding-close. Pairwise
cosine distances among the four Harry references range from 0.372545 to 0.654987, so the
gallery still should not be interpreted as a tight identity cluster. The side profile is
valuable because it covers a view the original two pins did not.

## Tracker behavior

- Association is deterministic greedy IoU, highest overlap first, with track ID and face
  index as stable tie-breaks.
- A match at `iou >= --iou-min` continues a track; the default is 0.3.
- A track expires after exactly `--track-ttl` unseen absolute frames; the default is 15.
- The displayed identity is the majority vote over the track history.
- `Unknown` is used for an Unknown majority and for every tied top vote, including a tie
  between two named identities.
- New faces always receive new monotonically increasing track IDs.

This is label smoothing, not detection interpolation. It cannot create a box on a frame
where RetinaFace found no face and therefore does not weaken or replace stride-1 R1
detection.

## Cache-backed before/after evidence

Window: absolute frames 2263–2282, 20 frames, stride 1, provisional threshold 0.30,
`all` pins, base normalization. This controlled comparison preceded the gallery refresh,
so it isolates tracker behavior from the new reference data.

| Measure | Unsmoothed | `--smooth` |
| --- | ---: | ---: |
| FaceCache hits | 20/20 | 20/20 |
| Cache misses | 0 | 0 |
| Model load | 0.000 s | 0.000 s |
| Perception | 0.000 s | 0.000 s |
| Pipeline elapsed | 0.318 s | 0.234 s |
| Rendered Harry Potter | 6 | 9 |
| Rendered Unknown | 14 | 11 |

The raw CSVs are byte-identical:

```text
0822cbe654c08e2dd7bb02980c92769d8da00db72da5ee765fb3fefd278664b0
```

At absolute frame 2268, the unsmoothed render changes Harry to `Unknown`; the smoothed
render retains Harry from the preceding five matching frames. The side-by-side inspection
artifact is `output/m6/frame-2268-before-after.png` (left: raw, right: smoothed).

Generated review artifacts:

- `output/m6/harry-before.mp4`
- `output/m6/harry-after.mp4`
- `output/m6/harry-before.csv`
- `output/m6/harry-after.csv`
- `output/m6/frame-2268-before-after.png`

These artifacts are gitignored because they contain source-video imagery and derived face
evidence.

## Full-clip smoothing result

The full refreshed-gallery cache was also rendered with `--smooth` in 12.786 s, again with
3,044/3,044 cache hits and zero model or perception work. The raw CSV remained byte-identical
to the unsmoothed replay, but lifetime majority voting reduced the displayed named counts:

| Rendered label | Unsmoothed | Smoothed |
| --- | ---: | ---: |
| Harry Potter | 255 | 131 |
| Hermione Granger | 174 | 99 |
| Prof. McGonagall | 34 | 29 |
| Prof. Severus Snape | 275 | 60 |
| Ron Weasley | 180 | 164 |
| Unknown | 4,435 | 4,870 |

This is worse at full-clip scale despite the improvement in the selected Harry flicker
window. The tracker is therefore implemented and available for experiments, but is not
enabled by the one-shot final runner. The accepted output uses unsmoothed raw assignments.

## Audio verification

The one-shot runner published the refreshed full 3,044-frame labelled output after a real
remux. It completed without re-encoding and ffprobe reported:

| Stream | Codec | Duration |
| --- | --- | ---: |
| Labelled video | MPEG-4 Part 2 (`mpeg4`) | 101.568235 s |
| Source audio | AAC (`aac`) | 101.564082 s |
| Container | MP4 | 101.568235 s |

The accepted output is 61 MB and contains 3,044 readable video frames. `matches.csv` has
5,353 data rows plus its header. README also documents a separate H.264/AAC command for
browser-oriented delivery.

SHA-256:

- labelled video: `f2d68c45e9abc5a8bb6adcd7f91d91cab9547149780fcc74dd890c6b064df400`
- matches CSV: `5abc33b1e892a53661ae0840c46abc17a8ea6354514ffafaf4866b39b191e9ae`
- schema-2 FaceCache: `22d55012f78a87c18f9e67d76a9a783494150522a039e0e4365c8aeee73f2a08`
- refreshed gallery cache: `76bed34b1e30529f2d344b095fa51ae6a31c6cbd6835ee0b96fbcf10b25613da`

## Limitations and open evidence

- A majority vote can preserve a wrong early assignment as well as a correct one. The
  full-clip result also shows that Unknown can dominate long tracks and suppress many valid
  named frames. Smoothing therefore remains opt-in and the raw CSV is the audit source.
- When a prior majority supplies the displayed name, the percentage shown is still the
  current frame's match confidence. Frame 2268 consequently displays `Harry Potter 34%`:
  the name is temporal, while the percentage transparently remains frame-local.
- Full-history voting is sticky during a long continuous track. The approved M6 contract
  does not introduce a rolling-window tunable; `track_ttl` resets history only after a
  sufficiently long disappearance. A bounded voting window is the next architectural
  experiment if smoothing is revisited.
- The owner-curated side profile proved to be the primary Harry accuracy improvement.
  Tracking can reduce local flicker but does not repair gallery coverage or alter the
  underlying assignment evidence.

## R1/R2 impact

- **R1:** unchanged. RetinaFace still runs or replays a cached result on every stride-1
  frame, and every detected face remains boxed. The tracker never suppresses a face.
- **R2:** the new gallery materially improves raw recognition. Tracking improves a selected
  local flicker case, but the current lifetime-vote policy degrades full-clip label coverage;
  it remains available but disabled in the accepted run.
