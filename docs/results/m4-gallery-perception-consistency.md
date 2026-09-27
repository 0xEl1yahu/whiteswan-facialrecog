# M4 Gallery/Video Perception Consistency Correction

> **Historical snapshot as of `0c7549a`.** This report covers the then-current 13-photo
> gallery and threshold `0.30`. The 17-photo gallery and D3 result supersede its counts and
> hashes. See [`docs/STATUS.md`](../STATUS.md).

Status at the time: complete; regenerated deliverables accepted for PR review

Date: 2026-09-27
Branch: `investigation/video-runtime`

## Outcome

Reference photos and video faces now use the same optimized RetinaFace detection,
crop-local alignment, and Facenet512 embedding implementation. The correction invalidated
only the gallery cache. All 3,044 cached video-frame analyses were reused, so the 33-minute
video perception pass was not repeated.

The regenerated unsmoothed output contains 5,353 boxes and evidence rows. Seventy-four
threshold decisions changed: 37 correct character labels were added, 36 correct borderline
labels became `Unknown`, and one incorrect Harry label on Ron became `Unknown`. No face
changed from one named character to another, and visual review found no new wrong-name
assignment.

Threshold `0.30` and base normalization remain provisional until M5. This correction makes
that future comparison internally consistent; it does not tune either value.

## Cause and implementation

Before this correction, video faces used `face_labeller.perception.embed_faces`, which
removes redundant grid-aligned detector margin, restores RetinaFace coordinates, aligns
each face locally, and embeds the aligned crops with `detector_backend="skip"`. Gallery
photos instead called `DeepFace.represent(... detector_backend="retinaface")` directly.

Gallery indexing now decodes each reference photo and calls the shared perception boundary
with an immutable config copy whose `max_faces` is one. An empty result is still rejected
and skipped with a warning. Gallery cache schema 2 includes:

- `perception_pipeline="retinaface_exact_resize_crop_v1"`;
- `detector_black_halo=32`; and
- all previous model, normalization, alignment, expansion, L2, and version inputs.

The gallery key changed from
`799c07df4ebcc29fefc98c3feb18383d84f269e9f09ff4afbfe270233161f0fc`
to `b2f6da96a721de17cd3c938c01ecf0e7b937304f2ab76ef2ca502be758cf86e3`.
The video FaceCache key and file remained unchanged.

## Reference-photo measurement

All 13 current reference photos produced exactly one usable face through both paths. Their
old-path and shared-path unit embeddings differ by:

| Same-photo cosine distance | Result |
| --- | ---: |
| Mean | 0.005428 |
| Median | 0.004782 |
| p95 | 0.014686 |
| Maximum | 0.015266 |

These values directly measure the gallery-path mismatch. The earlier median `0.016465`
and maximum `0.272496` figures compare old and optimized perception on matched video faces
and must not be used as the size of the gallery mismatch.

## Zero-inference regeneration

The one-shot runner reported:

```text
selected=3044 cached=3044 to_infer=0
complete elapsed=50.484s gallery=22.411s model=9.314s
perception=0.000s processed=3044 written=3044 faces=5353 failures=0
cache_hits=3044 cache_misses=0
```

Gallery regeneration covered 4 Harry, 3 Hermione, and 2 each for McGonagall, Snape, and
Ron, with no skipped photos. The elapsed time includes gallery model construction, cached
video rendering, CSV publication, and the runner's staged delivery work. RetinaFace and
Facenet512 performed no video-frame inference.

The complete FaceCache SHA-256 stayed
`22d55012f78a87c18f9e67d76a9a783494150522a039e0e4365c8aeee73f2a08`,
independently confirming that cached video detections and embeddings were not rewritten.
Old and regenerated CSV rows also have identical frame/face keys, boxes, detector
confidences, and thresholds.

## Assignment comparison

| Raw assignment | Old-path gallery | Shared-path gallery | Change |
| --- | ---: | ---: | ---: |
| Harry Potter | 255 | 266 | +11 |
| Hermione Granger | 174 | 168 | -6 |
| Prof. McGonagall | 34 | 34 | 0 |
| Prof. Severus Snape | 275 | 262 | -13 |
| Ron Weasley | 180 | 188 | +8 |
| Unknown | 4,435 | 4,435 | 0 |
| Total | 5,353 | 5,353 | 0 |

Nearest gallery owner changed for 148 rows. Final threshold assignment changed for 74
rows (1.38%):

- `Unknown` to Harry, 19: frames 88, 91, 97, 109, 131, 184, 185, 547, 549, 1070,
  1466, 1748, 1750, 2272, 2273, 2275, 2981, 2982, 2988;
- `Unknown` to Hermione, 2: frames 344, 347;
- `Unknown` to Ron, 16: frames 1601, 1613, 1615, 1621, 1622, 1624, 1625, 1626,
  1628, 1629, 1630, 1632, 1633, 1634, 1757, 1791;
- Harry to `Unknown`, 8: frames 536, 537, 1205, 1821, 2075, 2076, 2863, 2882;
- Hermione to `Unknown`, 8: frames 759, 766, 767, 770, 788, 1172, 1189, 1190;
- Snape to `Unknown`, 13: frames 362, 363, 392, 394, 401, 405, 419, 424, 438, 461,
  469, 473, 578; and
- Ron to `Unknown`, 8: frames 2299, 2300, 2303, 2342, 2343, 2344, 2345, 2347.

The changed rows' old distances span `0.276675-0.330870`; their new distances span
`0.282783-0.327637`. The absolute distance movement has mean `0.013213`, median
`0.012091`, and maximum `0.031521`.

The Hermione and Ron movements are not nearest-pin switches. Every affected face retained
the same nearest reference photo before and after the correction:

- five Hermione losses at frames 759-788 use `104bbeffb44ecbc9054b00130b288a27.jpg`
  and move 0.009-0.013 farther away; three losses at frames 1172/1189/1190 use the
  Philosopher's Stone reference and move 0.002-0.004 farther away; the two gains at frames
  344/347 use the premiere reference and move 0.026-0.032 closer; and
- all affected Ron faces use the premiere reference. The eight losses at frames 2299-2347
  move 0.011-0.031 farther away, while the sixteen gains at frames 1601-1791 move
  0.007-0.019 closer.

One corrected reference crop can therefore become a better match for one pose and a worse
match for another. The still-provisional hard 0.30 cutoff converts these continuous
distance changes into abrupt named/`Unknown` changes. Additional pose-diverse Hermione and
especially Ron references (Ron currently has only two) should be evaluated before M5.

## Visual review

All 74 changed rows were reviewed in five generated contact-sheet pages under
`output/gallery-consistency-review/` (gitignored):

- all 37 `Unknown`-to-character changes show the stated character;
- 36 character-to-`Unknown` changes remove a correct but borderline label;
- frame 1821 removes an incorrect Harry label from a clearly visible Ron; and
- no change introduces a wrong character name.

The correction therefore trades borderline coverage in both directions while removing
one observed false positive. The repeated adjacent frames are correlated observations,
not 74 independent scenes. There is still no manually annotated full-video ground truth.

## Delivery validation

| Artifact | Validation |
| --- | --- |
| Source video | 3,044 readable frames; H.264; 1920x1080; 29.97002997 fps; video duration 101.568133 s |
| Labelled video | 3,044 readable frames; MPEG-4 Part 2; 1920x1080; 29.97 fps; duration 101.568235 s |
| Audio | AAC copied without re-encoding; output audio duration 101.564082 s |
| `matches.csv` | Required 12-column schema; 5,353 unique `(frame_idx, face_idx)` rows |
| FaceCache | 3,044 cached hits; 0 misses; unchanged SHA-256 |
| Gallery cache | Schema 2; 13 embeddings; shared pipeline/halo metadata present |

The source AAC stream extends slightly beyond the source video stream. The runner's
`-shortest` delivery remux clips copied audio to the labelled video duration rather than
extending the container.

SHA-256:

- source video: `67c428ef51ccd86ccbe55a08a6e00efe5540eb6387bb03ff0fd4aade8fbba353`;
- labelled video: `7b4f35205be35f050284621b6fb286752603d53728440af876217de636b172e7`;
- matches CSV: `6efda09da66d6668bf8af5173b5126aa02e68151189133d3501d8314041b469e`;
- schema-2 FaceCache: `22d55012f78a87c18f9e67d76a9a783494150522a039e0e4365c8aeee73f2a08`;
- schema-2 shared-path gallery cache:
  `550f3709065726c924c22433741f745c36a497a90c53d8b18d2b5698617c27bb`.

Artifact sizes at validation were 63,688,810 bytes for the labelled video, 454,696 bytes
for the CSV, and 27,506 bytes for the gallery cache.

## R1/R2 impact

- **R1:** unchanged. The same 5,353 cached detections are rendered over all 3,044 stride-1
  frames; no box is added, removed, or capped by the gallery correction.
- **R2:** reference and video embeddings now share one preprocessing contract. The observed
  changes remain character/`Unknown` threshold movements, with one reviewed false-positive
  correction and no new wrong-name assignment. M5 can now evaluate threshold and
  normalization without calibrating against a known path mismatch.
