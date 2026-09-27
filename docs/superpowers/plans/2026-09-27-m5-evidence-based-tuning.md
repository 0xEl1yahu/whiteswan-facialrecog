# M5 Evidence-Based Threshold and Normalization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. M5 is sequential: build the analysis tooling, freeze the sample, generate evidence, review it, and stop for the owner's D3 decision.

**Goal:** Decide whether the production cosine threshold should remain `0.30` and whether Facenet512 normalization should remain `base` or change to `Facenet2018`, using reproducible evidence from the accepted stride-1 run and a fixed 300-frame A/B sample.

**Architecture:** Keep production perception and matching unchanged while M5 runs. A standalone, model-free `analyse_matches.py` reads the existing `matches.csv`, explicit cache metadata, and source video to generate deterministic histograms, review manifests, contact sheets, threshold sweeps, and base-versus-Facenet2018 comparisons. The accepted base FaceCache is replayed; only the six pre-registered `Facenet2018` sample windows and that normalization's gallery variant require new CPU inference. Generated artifacts stay under ignored `output/analysis/`, and no default changes until Eli makes D3 at the STOP gate.

**Tech Stack:** Python 3.11, standard-library `argparse`/`csv`/`json`, NumPy, OpenCV, pytest, the existing DeepFace 0.0.101 + RetinaFace 0.0.18 + Facenet512 pipeline for the A/B runs only.

**Spec:** `docs/design/design-plan.md` section 9 (M5), `docs/implementation/implementation-plan.md` Task 5, and the fixed contracts in `face_labeller/evidence.py` and `face_labeller/cache.py`.

## Global Constraints

- Preserve R1: M5 must not cap detections, suppress Unknown faces, change boxes, or alter the accepted stride-1 deliverable.
- Improve R2 from evidence: judge correct-name coverage, wrong-name errors, and appropriate Unknown decisions separately.
- Keep `Facenet512`, `retinaface`, cosine distance, `all` pins, CPU execution, and the strict `distance < threshold` rule fixed.
- Treat `0.30` and `base` as provisional throughout implementation. Do not change `Config` defaults, the full-run script, accepted video, or accepted CSV before the M5 STOP decision.
- Do not change the stable production CLI, public dataclasses, `matches.csv` schema, gallery-cache schema, or FaceCache schema/key.
- Do not call DeepFace, RetinaFace, or TensorFlow from baseline analysis or contact-sheet generation. Model calls are allowed only through the existing production CLI for the registered A/B sample.
- Reuse the complete base FaceCache. Threshold sweeps are arithmetic replays over recorded distances and must never load a model.
- Keep every analysis tunable as a named CLI option or documented analysis constant. Do not add pandas, plotting libraries, or other dependencies.
- Never commit reference photos, source/output video, caches, CSV evidence, crops, contact sheets, model weights, or reviewer annotations containing private image data.
- A contact sheet can reveal classification errors, but it cannot reveal a face the detector omitted. Record detector misses through a separate source-video/output-video playback audit.

## Frozen Starting Point

Before changing source, verify and record the current private baseline:

- 3,044 source frames and 5,353 face rows.
- 17 supported owner-curated references: Harry 4, Hermione 4, Ron 5, McGonagall 2, and Snape 2.
- Current assignments: Harry 266, Hermione 198, Ron 214, McGonagall 34, Snape 262, and Unknown 4,379.
- `normalization=base`, `threshold=0.30`, `pin_strategy=all`, `stride=1`, and smoothing off.
- The exact source-video SHA-256, CSV SHA-256, gallery-cache path/key, FaceCache path/key, dependency versions, and generation timestamp.

If any count, hash, or cache configuration has changed, do not silently update the expectations. Explain the difference and establish a new baseline before continuing.

## Registered Normalization Sample

Freeze these six inclusive 50-frame windows before either A/B run. They total 300 frames and cover every named character plus varied Unknown faces:

| Window | Frames | Coverage reason |
| --- | ---: | --- |
| S1 | 80-129 | Harry and Hermione in the early sequence |
| S2 | 190-239 | Snape and known borderline decisions |
| S3 | 1120-1169 | Hermione views strengthened by the expanded gallery |
| S4 | 1560-1609 | Ron's first sustained labelled sequence |
| S5 | 2290-2339 | Later Ron views and Unknown competition |
| S6 | 2920-2969 | McGonagall's late sequence |

Persist the manifest and its rationale in `output/tuning-report.md` before running `Facenet2018`. Do not replace difficult windows after seeing the result. Any extra diagnostic window must be reported as post-hoc and excluded from the primary A/B totals.

## Review Focus

- Strictly validate the 12-column match schema and numeric values before drawing conclusions.
- Group distance distributions by `nearest_name`, not `assigned_name`; otherwise rejected faces disappear into one undifferentiated Unknown group.
- Keep threshold effects separate from nearest-neighbour errors. A threshold can reject a wrong nearest character, but it cannot make the nearest character correct.
- Compare A/B detections spatially within each frame before comparing embeddings or labels. Match boxes one-to-one by highest IoU, report unmatched detections, and do not assume `(frame_idx, face_idx)` is stable across independent inference.
- Report both aggregate and per-character results so a global improvement cannot conceal a regression for Hermione, Ron, or the two-photo McGonagall/Snape galleries.
- Give wrong-name assignments more weight than correct-name-to-Unknown flips, but show the raw trade-off table rather than hiding it behind one score.
- Separate `wrong name`, `known character rejected as Unknown`, `true extra correctly Unknown`, `uncertain`, and `detector miss` in visual review.
- Verify both normalization cache variants survive and replay warm; changing normalization must not overwrite the base cache.

## Planned Analysis CLI

Keep the production `label_video.py` interface unchanged. `analyse_matches.py` owns three
analysis-only subcommands:

```text
analyse_matches.py baseline \
  --csv PATH --video PATH --face-cache PATH --gallery-cache PATH \
  --output-dir PATH --threshold FLOAT --margin FLOAT \
  --per-character-limit INT --bin-edges FLOAT [FLOAT ...]

analyse_matches.py compare \
  --base-csv PATH [PATH ...] --candidate-csv PATH [PATH ...] \
  --video PATH --output-dir PATH --iou-min FLOAT

analyse_matches.py report \
  --baseline-summary PATH --review-csv PATH --comparison PATH \
  --normalization-review PATH --detector-misses PATH --timings PATH \
  --candidate-face-cache PATH --candidate-gallery-cache PATH \
  --thresholds FLOAT [FLOAT ...] --recommended-threshold FLOAT \
  --recommended-normalization {base,Facenet2018} \
  --threshold-output PATH --output PATH
```

`baseline` creates summaries, histograms, the review manifest, and its contact sheet.
`compare` creates the spatially matched normalization evidence and changed-decision contact
sheet. `report` validates completed human review fields, calculates the threshold trade-off
table, and assembles the Markdown report. All three commands are deterministic and
model-free.

---

### Task 1: Build Strict, Pure Match-Evidence Loading

**Files:**
- Create: `analyse_matches.py`
- Create: `tests/test_analysis.py`

**Interfaces:**
- Internal immutable `MatchRow` with fields matching `face_labeller.evidence.MATCH_COLUMNS`.
- `load_match_rows(path: Path) -> list[MatchRow]`
- `load_match_rows_many(paths: Sequence[Path]) -> list[MatchRow]`
- `load_cache_metadata(path: Path) -> dict[str, object]`
- `summarize_assignments(rows: Sequence[MatchRow]) -> dict[str, object]`

- [x] **Step 1: Write failing schema and parsing tests**

  Cover the exact header and order, empty CSVs, duplicate `(frame_idx, face_idx)` keys, negative indices, non-positive boxes, non-finite numeric values, invalid thresholds, invalid assigned names, an assigned name inconsistent with the strict threshold rule, and deterministic merging of disjoint window CSVs. Require actionable errors containing the file and row number.

- [x] **Step 2: Verify RED**

  Run: `.venv/bin/python -m pytest tests/test_analysis.py -k "load or schema or summary" -v`

  Expected: FAIL because `analyse_matches.py` does not exist.

- [x] **Step 3: Implement the smallest strict loader**

  Import `MATCH_COLUMNS` rather than duplicating the production header. Parse values into typed immutable rows, retain full floating-point precision, reject duplicate face keys, and return rows sorted by `(frame_idx, face_idx)`. Read only `meta_json` from explicit `.npz` cache paths using `allow_pickle=False`; never discover a cache by taking the first glob match.

- [x] **Step 4: Implement model-free summaries**

  Report row count, unique frames, faces per frame, assignment totals, nearest-owner totals, distance min/median/percentiles/max, and cache/video/gallery identities. Keep JSON keys and ordering deterministic.

- [x] **Step 5: Verify GREEN**

  Run: `.venv/bin/python -m pytest tests/test_analysis.py -k "load or schema or summary" -v`

  Expected: PASS with no imports of DeepFace, RetinaFace, or TensorFlow.

---

### Task 2: Generate Histograms, Review Samples, and Contact Sheets

**Files:**
- Modify: `analyse_matches.py`
- Modify: `tests/test_analysis.py`
- Generate, ignored: `output/analysis/baseline-summary.json`
- Generate, ignored: `output/analysis/distance-histograms.csv`
- Generate, ignored: `output/analysis/near-threshold-review.csv`
- Generate, ignored: `output/analysis/near-threshold-contact-sheet.png`
- Generate manually, ignored: `output/analysis/detector-misses.csv`

**Interfaces:**
- `build_distance_histograms(rows, bin_edges) -> dict[str, np.ndarray]`
- `select_near_threshold(rows, threshold, margin, per_character_limit) -> list[MatchRow]`
- `build_contact_sheet(video_path, rows, output_path, *, cell_size, columns) -> ContactSheetResult`
- `write_review_manifest(rows, path) -> None`

- [x] **Step 1: Write failing histogram and selection tests**

  Test fixed caller-supplied bins, grouping by nearest owner, inclusion of the rightmost boundary, empty groups, and conservation of total rows. Test stable near-threshold ordering by absolute distance from the threshold, then nearest owner, frame, and face index; enforce limits independently per nearest owner.

- [x] **Step 2: Write failing contact-sheet tests**

  Build a tiny fixture video containing valid, edge-clipped, zero-area, missing-frame, and repeated-frame crops. Assert deterministic layout and dimensions, source-frame reuse, safe skip accounting, visible frame/face/nearest/distance/assignment labels, and atomic output publication. Assert the path never imports or calls model code.

- [x] **Step 3: Verify RED**

  Run: `.venv/bin/python -m pytest tests/test_analysis.py -k "histogram or threshold or contact or review" -v`

  Expected: FAIL because the analysis functions are not implemented.

- [x] **Step 4: Implement deterministic evidence generation**

  Expose histogram bin edges, threshold, review margin, per-character limit, cell dimensions, and column count as analysis CLI arguments with documented defaults. The review CSV must copy the source evidence fields and add blank `truth_name`, `review_outcome`, and `notes` columns; allowed outcomes are `correct_named`, `known_as_unknown`, `wrong_name`, `true_unknown`, and `uncertain`.

- [x] **Step 5: Implement safe one-pass crop extraction**

  Seek/read each required frame once in ascending order, clip boxes to the decoded frame, never resize the source before cropping, letterbox crops into fixed cells without distortion, and report every skipped row. Fail if the requested video cannot be opened or if all requested crops are unusable.

- [x] **Step 6: Verify GREEN**

  Run: `.venv/bin/python -m pytest tests/test_analysis.py -k "histogram or threshold or contact or review" -v`

  Expected: PASS.

---

### Task 3: Build Threshold Sweeps and Normalization Comparison

**Files:**
- Modify: `analyse_matches.py`
- Modify: `tests/test_analysis.py`
- Generate, ignored: `output/analysis/threshold-sweep.csv`
- Generate, ignored: `output/analysis/normalization-comparison.json`
- Generate, ignored: `output/analysis/normalization-changes.csv`
- Generate, ignored: `output/analysis/normalization-changes-contact-sheet.png`

**Interfaces:**
- `reassign_at_threshold(row: MatchRow, threshold: float) -> str`
- `score_thresholds(rows, reviewed_truth, candidates) -> list[ThresholdResult]`
- `match_detections_by_iou(base_rows, candidate_rows, iou_min) -> DetectionComparison`
- `compare_normalizations(base_rows, candidate_rows, iou_min) -> NormalizationComparison`

- [x] **Step 1: Write failing threshold-replay tests**

  Prove the rule is strictly `<`, exact-threshold distances remain Unknown, replay changes only `assigned_name`, and candidate thresholds are finite, unique, sorted, and non-negative. For adjudicated rows, report correct named, known-as-Unknown, wrong-name, correctly rejected extra, uncertain/excluded, known-character recall, and wrong-name rate. Do not manufacture ground truth for unreviewed rows.

- [x] **Step 2: Write failing spatial-comparison tests**

  Cover reordered faces, multiple faces competing for one box, deterministic one-to-one highest-IoU matching, unmatched faces on either side, configurable IoU rejection, per-character summaries, distance deltas, nearest-owner changes, and assignment-transition counts. Include a test showing why direct face-index joins would be wrong.

- [x] **Step 3: Verify RED**

  Run: `.venv/bin/python -m pytest tests/test_analysis.py -k "reassign or sweep or iou or normalization" -v`

  Expected: FAIL.

- [x] **Step 4: Implement arithmetic threshold replay**

  Generate a transparent trade-off table over caller-supplied candidate thresholds. The code may summarize evidence but must not automatically select the production threshold; the final recommendation requires the visual review and written rationale.

- [x] **Step 5: Implement spatial A/B comparison**

  Within each frame, form deterministic one-to-one box matches by descending IoU with stable tie-breakers. For matched faces report IoU, base/candidate nearest owner, distance, assignment, and deltas. Report unmatched detections separately; cosine similarity between base and candidate face embeddings is optional only if the explicit cache paths can be mapped safely to those matched detections.

- [x] **Step 6: Verify GREEN and the full analysis suite**

  Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`

  Expected: PASS.

---

### Task 4: Freeze and Generate the Full Baseline Evidence

**Files:**
- Modify: `README.md`
- Generate, ignored: all `output/analysis/baseline-*` artifacts
- Generate, ignored: `output/analysis/near-threshold-review.csv`
- Generate manually, ignored: `output/analysis/detector-misses.csv`

- [x] **Step 1: Record immutable run identity**

  Run read-only checks over the accepted source video, `output/matches.csv`, explicit base FaceCache, and explicit gallery cache. Store hashes, cache metadata, dependency versions, counts, and the frozen 300-frame sample in `output/analysis/baseline-summary.json` and the opening section of `output/tuning-report.md`.

  Canonical command, after resolving and recording the two explicit cache paths:

  ```bash
  .venv/bin/python analyse_matches.py baseline \
    --csv output/matches.csv \
    --video data/video-source/nimbus.mp4 \
    --face-cache <recorded-base-face-cache.npz> \
    --gallery-cache <recorded-base-gallery-cache.npz> \
    --output-dir output/analysis \
    --threshold 0.30 \
    --margin 0.05 \
    --per-character-limit 30 \
    --bin-edges 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.75 1.00 2.00
  ```

  The report must record these as M5 analysis settings, not new production defaults.

- [x] **Step 2: Generate full-clip distributions**

  Use the 5,353-row CSV to produce overall and per-nearest-character distance histograms, distance percentiles, assignment counts, faces-per-frame counts, and the number of assignments at each threshold candidate. Confirm the totals reconcile exactly with the frozen starting point.

- [x] **Step 3: Generate the near-threshold review set**

  Run the deterministic selector around `0.30`, produce the review manifest and contact sheet, and record the actual margin/limit/bin arguments in the report. Include every previously documented gallery-consistency flip even if it falls outside the capped selection, clearly marking those rows as an additional pre-registered evidence set.

- [x] **Step 4: Adjudicate classification outcomes**

  Review each selected crop against the source scene and owner-curated character identities. Fill `truth_name`, `review_outcome`, and notes; leave genuinely unclear crops as `uncertain` rather than forcing a label. Have the script revalidate that all non-uncertain rows have a complete, allowed adjudication before threshold scoring.

- [x] **Step 5: Audit detector misses separately**

  Watch the source and labelled output side-by-side across the six registered sample windows. Record any visible unboxed face with frame range, scene note, and severity in `detector-misses.csv`. Do not count detector misses as threshold or normalization failures, because no embedding/CSV row exists for them.

- [x] **Step 6: Document reproducible M5 commands**

  Add an M5 section to `README.md` explaining baseline analysis, generated artifacts, private-data exclusions, the separate detector audit, and how to reproduce the sample A/B without replacing the accepted output.

---

### Task 5: Run the Fixed 300-Frame Normalization A/B

**Files:**
- Run, ignored: `output/analysis/base/`
- Run, ignored: `output/analysis/facenet2018/`
- Run, ignored: separate configuration-keyed gallery and FaceCache variants under `cache/`
- Modify generated report only: `output/tuning-report.md`

- [x] **Step 1: Create isolated output paths for all six windows**

  For both normalizations, use one CSV and one disposable video per registered window. Every command must use the source video, full current gallery, `stride=1`, `batch-size=8`, `threshold=0.30`, `pin-strategy=all`, smoothing off, the shared cache directory, and the exact `start-frame`/`max-frames=50` pair from the manifest.

  Canonical command shape (substitute `base` or `Facenet2018`, the table's start frame,
  and the matching isolated filenames):

  ```bash
  .venv/bin/python label_video.py \
    --input data/video-source/nimbus.mp4 \
    --output output/analysis/<normalization>/s<window>.mp4 \
    --ref-dir data/reference-images \
    --stride 1 \
    --batch-size 8 \
    --threshold 0.30 \
    --pin-strategy all \
    --normalization <normalization> \
    --start-frame <80|190|1120|1560|2290|2920> \
    --max-frames 50 \
    --cache-dir cache \
    --csv output/analysis/<normalization>/s<window>.csv
  ```

  These sample videos are analysis artifacts and do not need audio remuxing; the accepted
  full output is neither read as an input nor replaced.

- [x] **Step 2: Replay the six base windows**

  Run the production CLI with `--normalization base`. Expected: all 300 frames are cache hits, no RetinaFace/Facenet512 inference occurs, and the concatenated sample CSV is an exact subset of the accepted full-run CSV for those frames. Stop if any row, box, distance, or assignment differs.

- [x] **Step 3: Run the six Facenet2018 windows cold**

  Run the same commands with `--normalization Facenet2018`. Expected: the 17-photo gallery variant is embedded once, exactly the 300 registered frames are inferred once, and all outputs remain isolated under `output/analysis/facenet2018/`. Capture wall-clock and stage timings for each window.

- [x] **Step 4: Prove both normalization variants are warm and retained**

  Repeat all six `Facenet2018` commands. Expected: 300/300 frames are cache hits, perception time is zero, and no model is loaded. Replay one base window again and confirm its earlier cache remains usable. Record both cache keys and file paths.

- [x] **Step 5: Compare geometry before recognition**

  Spatially match detections within each sampled frame. Report face counts, IoU distribution, unmatched boxes, and any ordering changes. If geometry is not effectively identical, separate detection variability from normalization effects and do not attribute all label changes to normalization.

- [x] **Step 6: Compare recognition outcomes**

  Generate per-character and overall nearest-owner changes, distance deltas, threshold transitions, correct-name coverage, wrong-name errors, and Unknown outcomes using the same reviewed truth set. Produce a contact sheet for every changed or unmatched decision and visually adjudicate it.

---

### Task 6: Write the D3 Recommendation and Stop

**Files:**
- Modify: `README.md`
- Modify: `docs/implementation/implementation-plan.md`
- Generate, ignored unless Eli requests publication: `output/tuning-report.md`

- [x] **Step 1: Complete the tuning report**

  Include baseline identity, sample manifest, exact commands, dependency versions, cache keys, cold/warm timings, full-clip histograms, threshold trade-off table, per-character results, normalization comparison, changed-row visual findings, detector misses, gallery limitations, and private-data exclusions.

- [x] **Step 2: Make one explicit recommendation**

  Recommend one threshold and one normalization, with the evidence for and against each. State separately what the choice changes for correct named coverage, wrong-name risk, Unknown handling, runtime/cache cost, and the small two-photo McGonagall/Snape galleries. If evidence is inconclusive, recommend retaining the current value and say what additional owner-curated evidence would resolve it.

- [x] **Step 3: Verify the implementation**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_analysis.py -v
  .venv/bin/python -m pytest -m "not slow" -v
  .venv/bin/python -m pytest -m slow -v
  .venv/bin/python -m pip check
  .venv/bin/python analyse_matches.py --help
  git diff --check
  git status --short
  ```

  Expected: all tests/checks pass; analysis commands are documented; generated/private artifacts remain ignored; production defaults and the accepted output are unchanged.

- [x] **Step 4: Self-review against the M5 acceptance gate**

  Confirm the work provides distance histograms by nearest character, a near-threshold contact sheet, the fixed-frame base/Facenet2018 A/B, warm-cache proof, a written recommendation, explicit R1/R2 impact, and no unapproved D3 change. Check that all aggregate claims can be traced to a generated artifact and exact command.

  Execution note (2026-09-27): self-review found and fixed a shared-pytest-process import
  assertion, added correctly-rejected-extra/excluded columns to the report, and made report
  counts data-driven. Final verification passed 252 non-slow tests, 3 slow tests, `pip
  check`, CLI help, compilation, and `git diff --check`. Generated artifacts remain ignored;
  no production default, accepted CSV, or accepted video was changed.

- [x] **Step 5: Report the M5 STOP gate**

  Present the recommendation, raw trade-offs, tests, timings, artifacts, limitations, and open questions to Eli. Stop. Do not change defaults, regenerate the accepted full video under a new D3 choice, start M7, or claim final delivery until Eli approves the threshold and normalization.

- [ ] **Step 6: Commit only if explicitly requested**

  Follow-up A/B note (2026-09-27): after the initial STOP report, the owner requested an
  isolated worktree and a direct threshold comparison. A full cache-only `0.31` replay
  preserved all 5,353 upstream detection/recognition rows and changed 111 assignments.
  Exhaustive visual review scored 110 correct new names and one new Harry-to-Ron error;
  `0.305` accepts 62 correct names without that observed error. See
  `docs/results/m5-threshold-ab.md`. This adds evidence but does not resolve D3 on the
  owner's behalf.

  Decision note (2026-09-27): Eli subsequently approved threshold `0.305` with `base`
  normalization and requested implementation in this isolated worktree. The config default
  and one-shot runner were updated under focused red/green tests. A production-CLI replay
  without a threshold override reused all 3,044 cached frames, ran no perception inference,
  and changed exactly 62 reviewed Unknown decisions to correct names with no observed new
  wrong name. The audio-preserved candidate and hashes are recorded in the linked report.

  If Eli asks, commit the M5 analysis source, tests, plan, and documentation as one milestone commit with the plain-language summary `feat: add evidence for face-match tuning`. Exclude every private/generated artifact. Run the complete fast and slow suites immediately before and after that commit.

## D3 Decision Execution

- [x] Apply the approved `0.305` threshold to `face_labeller/config.py` and
  `scripts/run_full_pipeline.sh`; retain `base` normalization.
- [x] Generate an isolated cache-only candidate video/CSV without replacing
  `output/nimbus_labelled.mp4` or `output/matches.csv`.
- [x] Verify the 62 assignment changes against the exhaustive reviewed cohort and preserve
  AAC audio in the candidate.
- [x] Leave the detector, recognizer, pin strategy, tracker, smoothing defaults, gallery
  contents, and reference-image format contract unchanged.
- [ ] Complete M7 packaging and final-delivery verification separately.
