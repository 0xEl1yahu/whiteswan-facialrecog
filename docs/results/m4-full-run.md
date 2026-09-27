# M4 Full Stride-1 Run

Status: full run complete; STOP for owner review before M5 tuning

Date: 2026-09-26

> Historical baseline: this report records the original two-reference-per-character run.
> The current accepted artifacts were refreshed after the owner added Harry side references,
> a Hermione three-quarter reference, and the gallery-perception consistency correction.
> Current counts, audio validation, and hashes are in
> [the gallery-perception consistency report](m4-gallery-perception-consistency.md).

## Command

```bash
/usr/bin/time -lp .venv/bin/python label_video.py \
  --input data/video-source/nimbus.mp4 \
  --output output/nimbus_labelled.mp4 \
  --ref-dir data/reference-images \
  --stride 1 \
  --batch-size 8 \
  --threshold 0.30 \
  --pin-strategy all \
  --cache-dir cache \
  --csv output/matches.csv
```

The threshold is still provisional pending M5. `Unknown` is therefore a valid result, and
`nearest_name` on an Unknown row is diagnostic evidence rather than an assigned identity.

## Run result

| Measure | Result |
| --- | ---: |
| Source frames selected / written | 3,044 / 3,044 |
| Stride | 1 |
| Face detections | 5,353 |
| Frames containing at least one detection | 2,371 |
| Frames containing no detection | 673 |
| Successful / failed frame analyses | 3,044 / 0 |
| Assigned character rows | 648 |
| Unknown rows | 4,705 |
| Process elapsed | 1,983.09 s (33m03s) |
| Pipeline elapsed | 1,981.737 s |
| Perception | 1,863.421 s |
| Combined video model load | 7.843 s |
| Gallery cache load | 0.098 s |
| Processing rate | 1.63 selected frames/s |

The measured 33m03s result is 13-22 minutes faster than the conservative 45-55 minute
range projected from the 300-frame gate. The full clip contains many sparser scenes than
the opening baseline window.

## Who was assigned

These are raw model assignments at cosine threshold 0.30, not manually corrected counts.

| Assigned label | Detections | Unique frames | First detection | Last detection |
| --- | ---: | ---: | --- | --- |
| Harry Potter | 7 | 7 | frame 2,263 / 01:15.509 | frame 2,863 / 01:35.529 |
| Hermione Granger | 152 | 152 | frame 104 / 00:03.470 | frame 1,371 / 00:45.746 |
| Prof. McGonagall | 34 | 34 | frame 2,932 / 01:37.831 | frame 2,966 / 01:38.966 |
| Prof. Severus Snape | 275 | 275 | frame 190 / 00:06.340 | frame 720 / 00:24.024 |
| Ron Weasley | 180 | 180 | frame 1,157 / 00:38.605 | frame 2,759 / 01:32.059 |
| Unknown | 4,705 | 2,010 | frame 70 / 00:02.336 | frame 3,043 / 01:41.535 |

The exact record for every detection is in `output/matches.csv`. Each row contains the
absolute frame index, face index, pixel box `(x, y, width, height)`, detector confidence,
nearest gallery owner, cosine distance, threshold, assigned label, and confidence.

## Where the named assignments occur

The ranges below are exact contiguous runs of frames carrying each assigned label. A
single-frame range means the label appeared on only that frame; gaps are not silently
bridged.

### Harry Potter

- frames 2,263-2,267 (01:15.509-01:15.642)
- frame 2,274 (01:15.876)
- frame 2,863 (01:35.529)

### Hermione Granger

- frames 104-153 (00:03.470-00:05.105)
- frames 485-556 (00:16.183-00:18.552)
- frame 1,156 (00:38.572)
- frames 1,159-1,178 (00:38.672-00:39.306)
- frames 1,180-1,182 (00:39.373-00:39.439)
- frames 1,189-1,192 (00:39.673-00:39.773)
- frames 1,370-1,371 (00:45.712-00:45.746)

### Prof. McGonagall

- frame 2,932 (01:37.831)
- frames 2,934-2,966 (01:37.898-01:38.966)

### Prof. Severus Snape

- frames 190, 209, 219, and 234 (00:06.340-00:07.808; isolated frames)
- frames 249-299 (00:08.308-00:09.977)
- frames 350-356, 359, and 361-363 (00:11.678-00:12.112)
- frames 392-399, 401-403, 405-410, 415-416, 419, 422-434, and 436-440
  (00:13.080-00:14.681)
- frames 461-465, 467-469, and 472-473 (00:15.382-00:15.782)
- frames 557-570, 572-575, 577-578, and 580-720 (00:18.585-00:24.024)

### Ron Weasley

- frames 1,157-1,158 (00:38.605-00:38.639; visually incorrect: Hermione)
- frames 1,562-1,600, 1,602-1,612, 1,614, 1,616-1,620, 1,623, 1,627, and
  1,631 (00:52.119-00:54.421)
- frames 1,759-1,790 and 1,792-1,793 (00:58.692-00:59.826)
- frames 1,809 and 1,814 (01:00.360-01:00.527; visually incorrect: Harry)
- frames 2,289-2,300, 2,303, 2,306-2,311, 2,314-2,321, 2,324-2,345, and
  2,347 (01:16.376-01:18.312)
- frames 2,694-2,721, 2,740-2,741, and 2,757-2,759 (01:29.890-01:32.059)

## Where the boxes are in the image

For this summary only, the 1920x1080 image was divided into thirds using each box centre.
The CSV retains the exact pixel coordinates.

| Label | Left/top | Left/middle | Centre/top | Centre/middle | Right/middle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Harry Potter | 0 | 0 | 0 | 7 | 0 |
| Hermione Granger | 6 | 116 | 0 | 30 | 0 |
| Prof. McGonagall | 0 | 0 | 0 | 34 | 0 |
| Prof. Severus Snape | 0 | 0 | 62 | 213 | 0 |
| Ron Weasley | 0 | 0 | 0 | 145 | 35 |

Unknown boxes cover the whole image: 29 left/top, 745 left/middle, 83 left/bottom,
59 centre/top, 1,594 centre/middle, 257 centre/bottom, 144 right/top, 1,474 right/middle,
and 320 right/bottom. Overall box extents reach all four image edges after clipping.

## Visual review

One representative frame from every contiguous named interval was inspected, together
with the maximum-density crowd frame and beginning/end examples.

- Hermione, Harry, McGonagall, and Snape were the correct visible identities in every
  sampled named interval.
- Four Ron assignments were visibly wrong: Hermione at frames 1,157-1,158 and Harry at
  frames 1,809 and 1,814. Their cosine distances were 0.288346, 0.280965, 0.269464, and
  0.292442 respectively, all just below the provisional 0.30 assignment threshold.
- Frame 1,968 at 01:05.666 contains the maximum 20 detections. The boxes cover many of the
  visible crowd faces, but very small distant faces remain unboxed. This is an R1 recall
  limitation of the current RetinaFace threshold/scale, not a claim of complete visual
  ground truth coverage.
- There is no manually annotated full-video ground truth. The interval review can expose
  obvious identity errors but cannot certify every box or every Unknown face.

These findings should feed M5 threshold analysis. They do not justify silently correcting
the CSV or changing the approved default before that gate.

The low Harry count was investigated separately. The evidence identifies insufficient
side/pose gallery coverage as the primary cause, with recognition producing Unknown or Ron
after RetinaFace has already boxed Harry. See
[M4 Harry Potter Discrepancy Investigation](m4-harry-discrepancy-investigation.md).

## Output validation

| Artifact | Validation |
| --- | --- |
| Source video | 3,044 frames, 1920x1080, 29.97002997 fps, 101.568133 s |
| Labelled video | 3,044 frames, 1920x1080, 29.97 fps, 101.568235 s |
| Media streams | Source: H.264 video + AAC audio; output: MPEG-4 video only |
| `matches.csv` | Required 12-column schema, 5,353 data rows |
| Schema-2 FaceCache | indices 0-3,043 contiguous, 3,044 `ok`, 0 `failed`, 5,353 faces |

The current OpenCV writer does not copy the source audio stream. Audio preservation is not
part of the approved data/CLI contract, but it is an open delivery limitation if the final
review artifact is expected to retain sound.

SHA-256:

- source video: `67c428ef51ccd86ccbe55a08a6e00efe5540eb6387bb03ff0fd4aade8fbba353`
- labelled video: `dfbef29aed8b62e00dd4ca40826d80c152003122782ab93c733683938566663a`
- matches CSV: `9d49873767213b46bf72dff2c2ff28e1567fdd98d41eccde5fd8d25babd3d9e6`
- schema-2 FaceCache: `22d55012f78a87c18f9e67d76a9a783494150522a039e0e4365c8aeee73f2a08`

Artifact sizes at validation were 58 MiB for the labelled video, 423 KiB for the CSV, and
9.7 MiB for the schema-2 FaceCache.

## How to resume or reproduce

Rerun the command at the top with the same cache directory. Preflight should report
`cached=3044 to_infer=0`; neither RetinaFace nor Facenet512 should run. Use different
`--output` and `--csv` paths when testing a new threshold so the accepted 0.30 artifacts
are preserved.

**STOP:** review this full-run evidence before M5 tuning or changing the threshold.
