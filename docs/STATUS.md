# Current Project Status

Last updated: 2026-09-29

Branch: `main` (M5/M7 work merged in PR #3)

Last committed implementation: `0d272ab`

This is the single source for current milestone, decision, gallery, and artifact state.
Historical reports preserve the measurements that were true at their named commit; they do
not override this page.

## Milestones and decisions

| Item | Current decision |
| --- | --- |
| M0–M4 | Complete: deterministic CPU pipeline, stride-1 full run, shared optimized perception |
| D1 | `all` gallery pins |
| D2 | Batch size 8; smoke stride 3; preview stride 2; final stride 1 |
| M5 / D3 | Complete: strict cosine threshold `0.305`, normalization `base` |
| M6 | Complete: smoothing available with `--smooth` but disabled by default; audio restored automatically by the one-shot runner |
| M7 | Complete: packaging and verification finished; merged to `main` in PR #3 |

Fixed model contract: RetinaFace detection, Facenet512 recognition, cosine distance, CPU
only. The detector still runs or replays a cached result for every frame in the final
stride-1 delivery.

## Current gallery

The owner-curated gallery has 17 usable photos:

| Character | Photos |
| --- | ---: |
| Harry Potter | 4 |
| Hermione Granger | 4 |
| Prof. McGonagall | 2 |
| Prof. Severus Snape | 2 |
| Ron Weasley | 5 |

## Accepted artifacts

Generated on 2026-09-27 by `scripts/run_full_pipeline.sh` without a threshold override.
The run reused all 3,044 compatible FaceCache entries and performed zero model loading or
perception inference.

| Artifact | Current result |
| --- | --- |
| Source | 3,044 frames; 1920x1080; 29.97002997 fps; H.264 + AAC; SHA-256 `67c428ef51ccd86ccbe55a08a6e00efe5540eb6387bb03ff0fd4aade8fbba353` |
| `output/nimbus_labelled.mp4` | 3,044 frames; 1920x1080; 29.97 fps; MPEG-4 + copied AAC; 101.568235 s; SHA-256 `fa37fd119ff5c0fc21d208df9a1efc6c4351dc32d9d48a804a657b9f343968ec` |
| `output/matches.csv` | 5,353 unique face rows; threshold `0.305`; SHA-256 `bc29006e7da6aa105a0f22ceed1895a36f1003ccb226719c5f83b3fcaddbcb4b` |
| Full FaceCache | 3,044 successful frames, 5,353 faces; SHA-256 `22d55012f78a87c18f9e67d76a9a783494150522a039e0e4365c8aeee73f2a08` |
| Current gallery cache | 17 embeddings; SHA-256 `dd95c0f8ffe4767fa3bca5a5cf65bda3077fb6ce11e8e8171ccd47623d2209d1` |

Accepted assignment totals:

| Assignment | Rows |
| --- | ---: |
| Harry Potter | 298 |
| Hermione Granger | 211 |
| Prof. McGonagall | 35 |
| Prof. Severus Snape | 276 |
| Ron Weasley | 216 |
| Unknown | 4,317 |

Relative to the frozen 17-photo `0.30` M5 baseline, D3 adds 62 visually reviewed correct
names and introduces no observed wrong-name assignment. The exhaustive evidence and its
single-clip limitation are in [the M5 threshold report](results/m5-threshold-ab.md).

## Performance

- Historical unoptimized full-clip projection: approximately 2 h 17 min.
- Measured optimized cold stride-1 run: 33 min 03 s.
- Current accepted cache-only `0.305` replay: 24.989 s, including video render but excluding
  the subsequent sub-second AAC remux.
- Handoff dry run measured 2026-09-29 on the Apple MacBook Air M4: frames 190–219 at
  stride 3, 10 cold detections, 30 output frames, 237 face rows, fresh 17-photo gallery
  and frame cache: 64.58 s wall.
- Handoff test run after that dry run: frames 0–299 at stride 1, 10 cache hits and 290
  detections, 300 output frames, 1,265 face rows: 387.32 s wall (6m27s). These two
  measurements exclude dependency, video, and first-time model-weight downloads.

## Documentation authority

1. This file: current milestone, decisions, gallery, and artifact identity.
2. [`design/design-plan.md`](design/design-plan.md): normative requirements and contracts.
3. [`implementation/implementation-plan.md`](implementation/implementation-plan.md): the
   completed milestone checklist and execution record.
4. [`../README.md`](../README.md): setup and operator workflow.
5. [`results/`](results/): immutable evidence snapshots, each labelled with its commit and
   supersession status.
6. [`archive/`](archive/): completed plans, superseded designs, and investigations; never a
   source of current status.

The owner-curated reference gallery is tracked in Git for handoff. The source video, caches,
CSV evidence, and generated video remain gitignored and must not be committed.
