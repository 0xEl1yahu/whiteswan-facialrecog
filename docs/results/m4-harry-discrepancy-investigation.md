# M4 Harry Potter Discrepancy Investigation

Status: diagnosis complete; no gallery, threshold, tracker, or output behavior changed

Date: 2026-09-26

## Headline finding

Harry's seven final labels are primarily a **gallery coverage problem expressed as an
assignment problem**, not a RetinaFace detection problem. In the reviewed scenes,
RetinaFace usually boxes Harry correctly, but Facenet512 either leaves the box Unknown
because it is too far from both Harry pins or, in two obvious cases, makes Ron the nearest
owner.

M6 tracking is not involved. It is neither enabled nor implemented: `--smooth` is parsed,
and the `Track` dataclass exists, but no `Tracker.update` implementation exists and the
video path never reads `cfg.smooth`.

## Gallery evidence

The gallery contains only two Harry photos:

1. a 2001 premiere image with no glasses, harsh flash lighting, and an off-axis gaze;
2. a frontal studio image in costume with glasses and even lighting.

Their cosine distance is `0.372545`, above the provisional `0.30` assignment threshold.
The M2 leave-one-out gate therefore identifies the other photo as Harry in both directions
but assigns both held-out images Unknown. The Harry gallery does not currently recognize
its own two examples under the production threshold.

The in-film reference is consistently the closer of the two for inspected video faces,
but many correct Harry detections remain far above 0.30:

| Frame | Visible result | Nearest assignment evidence |
| ---: | --- | --- |
| 104 | Harry boxed | nearest Harry, distance 0.435198, Unknown |
| 249 | Harry boxed | nearest Harry, distance 0.507414, Unknown |
| 520 | Harry boxed | nearest Harry, distance 0.478364, Unknown |
| 1,809 | Harry boxed | nearest Ron, distance 0.269464; nearest Harry pin 0.4003 |
| 2,265 | Harry boxed | nearest Harry, distance 0.260689, assigned Harry |
| 2,863 | Harry boxed | nearest Harry, distance 0.293111, assigned Harry |
| 3,043 | Harry boxed | nearest Harry, distance 0.357719, Unknown |

The same continuously visible face crosses the threshold from frame to frame. For example,
frame 2,262 is Unknown at 0.326098, frames 2,263-2,267 are assigned Harry, and frame 2,268
returns to Unknown at 0.348599. In the later shot, frames 2,843-2,862 are Unknown, frame
2,863 alone is Harry at 0.293111, and frame 2,864 is Unknown again at 0.309875. This is
assignment flicker around an under-representative gallery, not a sequence of detector
failures.

## Full-cache diagnostic

For diagnosis only, the seven correctly assigned Harry video embeddings were used as
temporary search seeds against the completed FaceCache. This did not alter any gallery or
output. It found 789 close detections across 80 temporal runs. Visual inspection of one
midpoint from every run found Harry in 79 runs and Ron in one.

After excluding that Ron run, the 788 probable Harry detections break down as:

| Production result | Diagnostic count |
| --- | ---: |
| Harry Potter | 7 |
| Unknown | 780 |
| Ron Weasley | 1 |

The wrong Ron result in that set is frame 1,809. Frame 1,814 is a second visually obvious
Harry-to-Ron error but sits just outside the seed-search cutoff. This diagnostic is not
manually annotated ground truth, so its counts should not replace the CSV; it establishes
the failure stage. Hundreds of Harry-like video embeddings and the inspected Harry faces
already have valid RetinaFace boxes before recognition fails.

## Why raising the global threshold is not the first fix

Changing only the threshold increases every owner's assignments, including false matches:

| Threshold | Harry assignments | All assignments |
| ---: | ---: | ---: |
| 0.30 | 7 | 648 |
| 0.32 | 33 | 762 |
| 0.35 | 91 | 982 |
| 0.38 | 165 | 1,181 |
| 0.40 | 228 | 1,306 |
| 0.45 | 398 | 1,607 |
| 0.50 | 626 | 1,962 |

This is not evidence that those additional rows are correct. The four already observed
wrong assignments all lie below 0.30. Threshold tuning remains M5 work and should follow,
not substitute for, a better Harry gallery.

## Required data improvement

The owner should add side-view coverage while keeping the actor, age, and film consistent:

- left and right three-quarter views;
- left and right profiles;
- glasses visible where possible;
- Great Hall or similarly warm/low scene lighting;
- neutral and speaking/expression variation;
- sharp face crops without occlusion.

The target remains 5-10 curated photos per character. The agent must not scrape or download
these images. When the owner adds them, only the new gallery photos are embedded; the
completed 3,044-frame FaceCache remains valid. Re-run leave-one-out gallery evidence, then
relabel from cache and compare Harry recall and cross-identity errors before changing the
threshold.

## Detection and tracker limits

This finding does not prove perfect detection. The wide crowd shot at frame 1,968 still
contains distant faces below useful RetinaFace scale. It does show that Harry's headline
shortfall is not explained by missing boxes in the reviewed close and medium shots.

Tracking would not repair the gallery. If M6 is later approved and implemented, it may
reduce frame-to-frame label flicker after a correct assignment, but it cannot reliably
recover an identity whose embeddings remain Unknown or closer to Ron.

## Audio preservation

The source contains H.264 video plus AAC audio. The current OpenCV writer publishes an
MPEG-4 video-only file, so audio is lost. This does not affect R1/R2 or model/cache results,
but it is a delivery-quality issue if the reviewed video is expected to retain sound.

Audio can be restored after labelling without rerunning inference or re-encoding video:

```bash
ffmpeg -i output/nimbus_labelled.mp4 \
  -i data/video-source/nimbus.mp4 \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a copy -shortest \
  output/nimbus_labelled_with_audio.mp4
```

This exact remux was tested in `/tmp`: it produced MPEG-4 video plus AAC audio with a
101.568235-second container duration, matching the labelled video. Integrating it into the
supported pipeline remains M6 work and was not done here.
