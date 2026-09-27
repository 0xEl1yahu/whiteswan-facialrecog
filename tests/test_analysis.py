from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from analyse_matches import (
    M5_SAMPLE_WINDOWS,
    MatchRow,
    build_contact_sheet,
    build_distance_histograms,
    compare_normalizations,
    main,
    load_cache_metadata,
    load_match_rows,
    load_match_rows_many,
    load_review_truth,
    match_detections_by_iou,
    reassign_at_threshold,
    score_thresholds,
    select_near_threshold,
    summarize_assignments,
    write_review_manifest,
)
from face_labeller.contracts import CHARACTER_NAMES
from face_labeller.evidence import MATCH_COLUMNS


def valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_idx": 7,
        "face_idx": 0,
        "x": 10,
        "y": 20,
        "w": 30,
        "h": 40,
        "det_conf": 0.95,
        "nearest_name": "Harry Potter",
        "distance": 0.25,
        "threshold": 0.30,
        "assigned_name": "Harry Potter",
        "confidence": 87.5,
    }
    row.update(overrides)
    return row


def write_matches(
    path: Path,
    rows: list[dict[str, object]],
    *,
    columns: tuple[str, ...] = MATCH_COLUMNS,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_load_match_rows_parses_exact_schema_and_preserves_float_precision(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "matches.csv"
    write_matches(csv_path, [valid_row(distance="0.250000000123")])

    rows = load_match_rows(csv_path)

    assert len(rows) == 1
    assert rows[0].frame_idx == 7
    assert rows[0].box == (10, 20, 30, 40)
    assert rows[0].distance == 0.250000000123
    assert rows[0].assigned_name == "Harry Potter"


def test_load_match_rows_accepts_header_only_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    write_matches(csv_path, [])

    assert load_match_rows(csv_path) == []


def test_load_match_rows_rejects_wrong_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "wrong.csv"
    write_matches(csv_path, [valid_row()], columns=MATCH_COLUMNS[:-1])

    with pytest.raises(ValueError, match=r"wrong\.csv.*header"):
        load_match_rows(csv_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frame_idx": -1}, "frame_idx"),
        ({"face_idx": -1}, "face_idx"),
        ({"w": 0}, "box"),
        ({"h": -2}, "box"),
        ({"det_conf": "nan"}, "det_conf"),
        ({"distance": "inf"}, "distance"),
        ({"threshold": -0.01}, "threshold"),
        ({"confidence": "nan"}, "confidence"),
        ({"nearest_name": "Dumbledore"}, "nearest_name"),
        ({"assigned_name": "Dumbledore"}, "assigned_name"),
        ({"distance": 0.30, "assigned_name": "Harry Potter"}, "strict threshold"),
        ({"distance": 0.29, "assigned_name": "Unknown"}, "strict threshold"),
        (
            {
                "nearest_name": "Ron Weasley",
                "assigned_name": "Harry Potter",
            },
            "assigned_name",
        ),
    ],
)
def test_load_match_rows_rejects_invalid_row_with_file_and_line(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    csv_path = tmp_path / "invalid.csv"
    write_matches(csv_path, [valid_row(**overrides)])

    with pytest.raises(ValueError) as caught:
        load_match_rows(csv_path)

    error = str(caught.value)
    assert "invalid.csv" in error
    assert "row 2" in error
    assert message in error


def test_load_match_rows_rejects_duplicate_face_key(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicates.csv"
    write_matches(csv_path, [valid_row(), valid_row(distance=0.20)])

    with pytest.raises(ValueError, match=r"duplicates\.csv.*row 3.*duplicate"):
        load_match_rows(csv_path)


def test_load_match_rows_many_sorts_rows_and_rejects_cross_file_duplicates(
    tmp_path: Path,
) -> None:
    later = tmp_path / "later.csv"
    earlier = tmp_path / "earlier.csv"
    write_matches(later, [valid_row(frame_idx=20, face_idx=1)])
    write_matches(earlier, [valid_row(frame_idx=10, face_idx=2)])

    rows = load_match_rows_many([later, earlier])

    assert [(row.frame_idx, row.face_idx) for row in rows] == [(10, 2), (20, 1)]

    duplicate = tmp_path / "duplicate.csv"
    write_matches(duplicate, [valid_row(frame_idx=10, face_idx=2)])
    with pytest.raises(ValueError, match=r"duplicate.*10.*2"):
        load_match_rows_many([earlier, duplicate])


def test_load_cache_metadata_reads_only_meta_json(tmp_path: Path) -> None:
    cache_path = tmp_path / "faces.npz"
    metadata = {"normalization": "base", "schema_version": 2}
    np.savez_compressed(
        cache_path,
        meta_json=np.asarray(json.dumps(metadata)),
        embeddings=np.asarray([object()], dtype=object),
    )

    assert load_cache_metadata(cache_path) == metadata


def test_load_cache_metadata_rejects_missing_or_non_object_metadata(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, values=np.asarray([1]))
    with pytest.raises(ValueError, match=r"missing\.npz.*meta_json"):
        load_cache_metadata(missing)

    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, meta_json=np.asarray("[]"))
    with pytest.raises(ValueError, match=r"invalid\.npz.*object"):
        load_cache_metadata(invalid)


def test_summarize_assignments_reports_deterministic_counts_and_distances(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "matches.csv"
    write_matches(
        csv_path,
        [
            valid_row(frame_idx=2, face_idx=0, distance=0.10),
            valid_row(
                frame_idx=2,
                face_idx=1,
                nearest_name="Ron Weasley",
                distance=0.40,
                assigned_name="Unknown",
            ),
            valid_row(
                frame_idx=5,
                face_idx=0,
                nearest_name="Ron Weasley",
                distance=0.20,
                assigned_name="Ron Weasley",
            ),
        ],
    )

    summary = summarize_assignments(load_match_rows(csv_path))

    assert summary == {
        "row_count": 3,
        "frame_count": 2,
        "faces_per_frame": {"min": 1, "median": 1.5, "mean": 1.5, "max": 2},
        "assignments": {"Harry Potter": 1, "Ron Weasley": 1, "Unknown": 1},
        "nearest_names": {"Harry Potter": 1, "Ron Weasley": 2},
        "distances": {
            "min": pytest.approx(0.10),
            "p25": pytest.approx(0.15),
            "median": pytest.approx(0.20),
            "p75": pytest.approx(0.30),
            "p90": pytest.approx(0.36),
            "p95": pytest.approx(0.38),
            "max": pytest.approx(0.40),
        },
    }


def test_summarize_assignments_handles_empty_rows() -> None:
    assert summarize_assignments([]) == {
        "row_count": 0,
        "frame_count": 0,
        "faces_per_frame": None,
        "assignments": {},
        "nearest_names": {},
        "distances": None,
    }


def test_analysis_import_does_not_load_model_libraries() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import analyse_matches; "
                "blocked = {'deepface', 'retinaface', 'tensorflow'} & sys.modules.keys(); "
                "raise SystemExit(f'model libraries loaded: {sorted(blocked)}' if blocked else 0)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert check.returncode == 0, check.stderr


def match_row(**overrides: object) -> MatchRow:
    values = valid_row(**overrides)
    return MatchRow(
        frame_idx=int(values["frame_idx"]),
        face_idx=int(values["face_idx"]),
        x=int(values["x"]),
        y=int(values["y"]),
        w=int(values["w"]),
        h=int(values["h"]),
        det_conf=float(values["det_conf"]),
        nearest_name=str(values["nearest_name"]),
        distance=float(values["distance"]),
        threshold=float(values["threshold"]),
        assigned_name=str(values["assigned_name"]),
        confidence=float(values["confidence"]),
    )


def test_build_distance_histograms_uses_fixed_bins_and_all_character_groups() -> None:
    rows = [
        match_row(frame_idx=1, distance=0.00),
        match_row(frame_idx=2, distance=0.10),
        match_row(frame_idx=3, distance=0.30, assigned_name="Unknown"),
        match_row(
            frame_idx=4,
            nearest_name="Ron Weasley",
            distance=0.20,
            assigned_name="Ron Weasley",
        ),
    ]

    histograms = build_distance_histograms(rows, np.asarray([0.0, 0.1, 0.2, 0.3]))

    assert list(histograms) == list(CHARACTER_NAMES)
    np.testing.assert_array_equal(histograms["Harry Potter"], [1, 1, 1])
    np.testing.assert_array_equal(histograms["Ron Weasley"], [0, 0, 1])
    for name in set(CHARACTER_NAMES) - {"Harry Potter", "Ron Weasley"}:
        np.testing.assert_array_equal(histograms[name], [0, 0, 0])
    assert sum(int(values.sum()) for values in histograms.values()) == len(rows)


@pytest.mark.parametrize(
    "bin_edges",
    [
        np.asarray([0.0]),
        np.asarray([0.0, 0.2, 0.1]),
        np.asarray([0.0, np.nan, 1.0]),
    ],
)
def test_build_distance_histograms_rejects_invalid_bins(
    bin_edges: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="bin edges"):
        build_distance_histograms([match_row()], bin_edges)


def test_select_near_threshold_is_stable_and_limits_each_character() -> None:
    rows = [
        match_row(frame_idx=8, distance=0.29),
        match_row(frame_idx=5, distance=0.31, assigned_name="Unknown"),
        match_row(frame_idx=3, distance=0.28),
        match_row(
            frame_idx=7,
            nearest_name="Ron Weasley",
            distance=0.31,
            assigned_name="Unknown",
        ),
        match_row(
            frame_idx=4,
            nearest_name="Ron Weasley",
            distance=0.29,
            assigned_name="Ron Weasley",
        ),
        match_row(
            frame_idx=2,
            nearest_name="Ron Weasley",
            distance=0.35,
            assigned_name="Unknown",
        ),
    ]

    selected = select_near_threshold(
        rows, threshold=0.30, margin=0.05, per_character_limit=2
    )

    assert [
        (row.nearest_name, row.frame_idx, row.distance) for row in selected
    ] == [
        ("Harry Potter", 5, 0.31),
        ("Harry Potter", 8, 0.29),
        ("Ron Weasley", 4, 0.29),
        ("Ron Weasley", 7, 0.31),
    ]


@pytest.mark.parametrize(
    ("threshold", "margin", "limit"),
    [(-0.1, 0.1, 1), (0.3, -0.1, 1), (0.3, 0.1, 0)],
)
def test_select_near_threshold_rejects_invalid_settings(
    threshold: float, margin: float, limit: int
) -> None:
    with pytest.raises(ValueError):
        select_near_threshold(
            [match_row()],
            threshold=threshold,
            margin=margin,
            per_character_limit=limit,
        )


def test_write_review_manifest_has_source_fields_and_blank_review_columns(
    tmp_path: Path,
) -> None:
    output = tmp_path / "review.csv"
    write_review_manifest([match_row()], output)

    with output.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)

    assert reader.fieldnames == [
        *MATCH_COLUMNS,
        "truth_name",
        "review_outcome",
        "notes",
    ]
    assert rows[0]["distance"] == "0.25"
    assert rows[0]["truth_name"] == ""
    assert rows[0]["review_outcome"] == ""
    assert rows[0]["notes"] == ""
    assert not list(tmp_path.glob(".*review.csv.*"))


def write_fixture_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (16, 12)
    )
    assert writer.isOpened()
    writer.write(np.full((12, 16, 3), (20, 40, 180), dtype=np.uint8))
    writer.write(np.full((12, 16, 3), (180, 80, 20), dtype=np.uint8))
    writer.release()


def test_build_contact_sheet_clips_skips_reuses_frames_and_labels_cells(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "fixture.avi"
    output_path = tmp_path / "sheet.png"
    write_fixture_video(video_path)
    rows = [
        match_row(frame_idx=0, face_idx=0, x=2, y=2, w=6, h=6),
        match_row(frame_idx=0, face_idx=1, x=-3, y=1, w=7, h=8),
        match_row(frame_idx=1, face_idx=0, x=5, y=3, w=6, h=7),
        replace(match_row(frame_idx=1, face_idx=1), w=0),
        match_row(frame_idx=10, face_idx=0),
    ]

    result = build_contact_sheet(
        video_path,
        rows,
        output_path,
        cell_size=(120, 100),
        columns=2,
    )

    assert result.requested == 5
    assert result.written == 3
    assert result.skipped == 2
    assert result.decoded_frames == 2
    assert result.skipped_keys == ((1, 1), (10, 0))
    sheet = cv2.imread(str(output_path))
    assert sheet.shape == (200, 240, 3)
    assert np.any(sheet[72:100, :120] > 0)
    assert not list(tmp_path.glob(".*sheet.png.*"))


def test_build_contact_sheet_preserves_existing_output_when_encoding_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import analyse_matches

    video_path = tmp_path / "fixture.avi"
    output_path = tmp_path / "sheet.png"
    write_fixture_video(video_path)
    output_path.write_bytes(b"previous")
    monkeypatch.setattr(
        analyse_matches.cv2,
        "imencode",
        lambda *_args, **_kwargs: (False, np.asarray([], dtype=np.uint8)),
    )

    with pytest.raises(OSError, match="encode"):
        build_contact_sheet(
            video_path,
            [match_row(frame_idx=0, x=2, y=2, w=6, h=6)],
            output_path,
            cell_size=(120, 100),
            columns=2,
        )

    assert output_path.read_bytes() == b"previous"


def test_build_contact_sheet_rejects_unusable_video_or_all_invalid_rows(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot open video"):
        build_contact_sheet(
            tmp_path / "missing.avi",
            [match_row()],
            tmp_path / "sheet.png",
            cell_size=(120, 100),
            columns=2,
        )

    video_path = tmp_path / "fixture.avi"
    write_fixture_video(video_path)
    with pytest.raises(ValueError, match="no usable crops"):
        build_contact_sheet(
            video_path,
            [replace(match_row(frame_idx=0), w=0)],
            tmp_path / "sheet.png",
            cell_size=(120, 100),
            columns=2,
        )


def test_reassign_at_threshold_is_strict_and_does_not_mutate_row() -> None:
    row = match_row(distance=0.30, assigned_name="Unknown")

    assert reassign_at_threshold(row, 0.30) == "Unknown"
    assert reassign_at_threshold(row, 0.300001) == "Harry Potter"
    assert row.assigned_name == "Unknown"


@pytest.mark.parametrize("threshold", [-0.01, float("nan"), float("inf")])
def test_reassign_at_threshold_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        reassign_at_threshold(match_row(), threshold)


def test_score_thresholds_reports_reviewed_tradeoffs_without_inventing_truth() -> None:
    rows = [
        match_row(frame_idx=1, distance=0.25),
        match_row(
            frame_idx=2,
            nearest_name="Ron Weasley",
            distance=0.35,
            assigned_name="Unknown",
        ),
        match_row(frame_idx=3, distance=0.27),
        match_row(
            frame_idx=4,
            nearest_name="Prof. Severus Snape",
            distance=0.26,
            assigned_name="Prof. Severus Snape",
        ),
        match_row(frame_idx=5, distance=0.28),
        match_row(frame_idx=6, distance=0.29),
    ]
    truth = {
        (1, 0): "Harry Potter",
        (2, 0): "Ron Weasley",
        (3, 0): "Unknown",
        (4, 0): "Hermione Granger",
        (5, 0): "Uncertain",
    }

    results = score_thresholds(rows, truth, [0.26, 0.30])

    first, second = results
    assert first.threshold == 0.26
    assert first.reviewed == 4
    assert first.correct_named == 1
    assert first.known_as_unknown == 2
    assert first.wrong_name == 0
    assert first.correctly_rejected_extra == 1
    assert first.excluded == 2
    assert first.known_character_recall == pytest.approx(1 / 3)
    assert first.wrong_name_rate == 0.0
    assert second.threshold == 0.30
    assert second.reviewed == 4
    assert second.correct_named == 1
    assert second.known_as_unknown == 1
    assert second.wrong_name == 2
    assert second.correctly_rejected_extra == 0
    assert second.excluded == 2
    assert second.known_character_recall == pytest.approx(1 / 3)
    assert second.wrong_name_rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        [0.30, 0.20],
        [0.30, 0.30],
        [-0.1],
        [float("nan")],
        [float("inf")],
    ],
)
def test_score_thresholds_rejects_invalid_candidate_sequence(
    candidates: list[float],
) -> None:
    with pytest.raises(ValueError, match="candidate thresholds"):
        score_thresholds([match_row()], {(7, 0): "Harry Potter"}, candidates)


def test_score_thresholds_rejects_invalid_truth_name() -> None:
    with pytest.raises(ValueError, match="truth"):
        score_thresholds([match_row()], {(7, 0): "Dumbledore"}, [0.30])


def normalization_rows() -> tuple[list[MatchRow], list[MatchRow]]:
    base = [
        match_row(
            frame_idx=1,
            face_idx=0,
            x=0,
            y=0,
            w=10,
            h=10,
            distance=0.20,
        ),
        match_row(
            frame_idx=1,
            face_idx=1,
            x=20,
            y=0,
            w=10,
            h=10,
            nearest_name="Ron Weasley",
            distance=0.25,
            assigned_name="Ron Weasley",
        ),
        match_row(frame_idx=2, face_idx=0, x=40, y=0, w=10, h=10),
    ]
    candidate = [
        match_row(
            frame_idx=1,
            face_idx=0,
            x=20,
            y=0,
            w=10,
            h=10,
            nearest_name="Ron Weasley",
            distance=0.31,
            assigned_name="Unknown",
        ),
        match_row(
            frame_idx=1,
            face_idx=1,
            x=0,
            y=0,
            w=10,
            h=10,
            nearest_name="Hermione Granger",
            distance=0.22,
            assigned_name="Hermione Granger",
        ),
        match_row(frame_idx=3, face_idx=0, x=60, y=0, w=10, h=10),
    ]
    return base, candidate


def test_match_detections_by_iou_handles_reordering_and_unmatched_faces() -> None:
    base, candidate = normalization_rows()

    comparison = match_detections_by_iou(base, candidate, iou_min=0.90)

    assert [
        (match.base.face_idx, match.candidate.face_idx, match.iou)
        for match in comparison.matched
    ] == [(0, 1, 1.0), (1, 0, 1.0)]
    assert [(row.frame_idx, row.face_idx) for row in comparison.unmatched_base] == [
        (2, 0)
    ]
    assert [
        (row.frame_idx, row.face_idx) for row in comparison.unmatched_candidate
    ] == [(3, 0)]


def test_match_detections_by_iou_uses_stable_one_to_one_tie_breaking() -> None:
    base = [
        match_row(frame_idx=1, face_idx=4, x=0, y=0, w=10, h=10),
        match_row(frame_idx=1, face_idx=2, x=0, y=0, w=10, h=10),
    ]
    candidate = [match_row(frame_idx=1, face_idx=7, x=0, y=0, w=10, h=10)]

    comparison = match_detections_by_iou(base, candidate, iou_min=1.0)

    assert len(comparison.matched) == 1
    assert comparison.matched[0].base.face_idx == 2
    assert comparison.unmatched_base[0].face_idx == 4


def test_match_detections_by_iou_respects_configurable_minimum() -> None:
    base = [match_row(frame_idx=1, x=0, y=0, w=10, h=10)]
    candidate = [match_row(frame_idx=1, x=5, y=0, w=10, h=10)]

    assert len(match_detections_by_iou(base, candidate, iou_min=0.30).matched) == 1
    rejected = match_detections_by_iou(base, candidate, iou_min=0.34)
    assert rejected.matched == ()
    assert len(rejected.unmatched_base) == 1
    assert len(rejected.unmatched_candidate) == 1


@pytest.mark.parametrize("iou_min", [-0.01, 1.01, float("nan")])
def test_match_detections_by_iou_rejects_invalid_minimum(iou_min: float) -> None:
    with pytest.raises(ValueError, match="IoU"):
        match_detections_by_iou([], [], iou_min=iou_min)


def test_compare_normalizations_reports_geometry_recognition_and_per_character() -> None:
    base, candidate = normalization_rows()

    comparison = compare_normalizations(base, candidate, iou_min=0.90)

    assert comparison.matched_count == 2
    assert comparison.unmatched_base_count == 1
    assert comparison.unmatched_candidate_count == 1
    assert comparison.iou == {"min": 1.0, "median": 1.0, "mean": 1.0}
    assert comparison.distance_delta == {
        "mean": pytest.approx(0.04),
        "median": pytest.approx(0.04),
        "mean_absolute": pytest.approx(0.04),
        "max_absolute": pytest.approx(0.06),
    }
    assert comparison.nearest_name_changes == 1
    assert comparison.assignment_changes == 2
    assert comparison.assignment_transitions == {
        "Harry Potter -> Hermione Granger": 1,
        "Ron Weasley -> Unknown": 1,
    }
    assert comparison.per_character["Harry Potter"] == {
        "matched": 1,
        "nearest_name_changes": 1,
        "assignment_changes": 1,
        "mean_distance_delta": pytest.approx(0.02),
    }
    assert comparison.per_character["Ron Weasley"] == {
        "matched": 1,
        "nearest_name_changes": 0,
        "assignment_changes": 1,
        "mean_distance_delta": pytest.approx(0.06),
    }


def test_baseline_cli_writes_reproducible_model_free_artifacts(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "fixture.avi"
    csv_path = tmp_path / "matches.csv"
    face_cache = tmp_path / "faces.npz"
    gallery_cache = tmp_path / "gallery.npz"
    supplemental_keys = tmp_path / "supplemental.csv"
    output_dir = tmp_path / "output" / "analysis"
    write_fixture_video(video_path)
    write_matches(
        csv_path,
        [
            valid_row(frame_idx=0, x=2, y=2, w=6, h=6, distance=0.29),
            valid_row(
                frame_idx=1,
                x=4,
                y=2,
                w=6,
                h=6,
                nearest_name="Ron Weasley",
                distance=0.31,
                assigned_name="Unknown",
            ),
            valid_row(
                frame_idx=0,
                face_idx=1,
                x=8,
                y=2,
                w=6,
                h=6,
                nearest_name="Hermione Granger",
                distance=0.80,
                assigned_name="Unknown",
            ),
        ],
    )
    supplemental_keys.write_text(
        "frame_idx,face_idx,source\n0,1,m4-gallery-consistency\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        face_cache,
        meta_json=np.asarray(json.dumps({"normalization": "base", "schema_version": 2})),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        statuses=np.asarray(["ok", "ok"]),
        boxes=np.asarray([[2, 2, 6, 6], [4, 2, 6, 6]], dtype=np.int32),
    )
    np.savez_compressed(
        gallery_cache,
        meta_json=np.asarray(json.dumps({"photo_count": 17, "schema_version": 2})),
        owners=np.asarray(["Harry Potter", "Ron Weasley"]),
    )

    result = main(
        [
            "baseline",
            "--csv",
            str(csv_path),
            "--video",
            str(video_path),
            "--face-cache",
            str(face_cache),
            "--gallery-cache",
            str(gallery_cache),
            "--output-dir",
            str(output_dir),
            "--supplemental-keys",
            str(supplemental_keys),
            "--threshold",
            "0.30",
            "--margin",
            "0.05",
            "--per-character-limit",
            "1",
            "--bin-edges",
            "0.0",
            "0.25",
            "0.30",
            "0.35",
            "1.0",
            "--cell-width",
            "120",
            "--cell-height",
            "100",
            "--columns",
            "2",
        ]
    )

    assert result == 0
    summary = json.loads((output_dir / "baseline-summary.json").read_text())
    assert summary["summary"]["row_count"] == 3
    assert summary["summary"]["assignments"] == {
        "Harry Potter": 1,
        "Unknown": 2,
    }
    assert summary["analysis_settings"] == {
        "bin_edges": [0.0, 0.25, 0.3, 0.35, 1.0],
        "margin": 0.05,
        "per_character_limit": 1,
        "threshold": 0.3,
    }
    assert summary["cache_metadata"]["face"]["normalization"] == "base"
    assert summary["cache_metadata"]["gallery"]["photo_count"] == 17
    assert summary["cache_inventory"] == {
        "face": {"cached_frames": 2, "faces": 2, "statuses": {"ok": 2}},
        "gallery": {
            "owners": {"Harry Potter": 1, "Ron Weasley": 1},
            "photo_count": 2,
        },
    }
    assert summary["inputs"]["csv"]["sha256"]
    assert summary["inputs"]["video"]["sha256"]
    assert len(summary["sample_windows"]) == 6
    assert summary["review"]["primary_selected"] == 2
    assert summary["review"]["supplemental_added"] == 1
    assert summary["review"]["selected"] == 3
    assert summary["review"]["contact_sheet"]["written"] == 3

    with (output_dir / "distance-histograms.csv").open(newline="") as source:
        histogram_rows = list(csv.DictReader(source))
    assert len(histogram_rows) == len(CHARACTER_NAMES) * 4
    assert sum(int(row["count"]) for row in histogram_rows) == 3
    assert (output_dir / "near-threshold-review.csv").is_file()
    assert (output_dir / "near-threshold-contact-sheet.png").is_file()
    assert (output_dir / "detector-misses.csv").read_text().splitlines()[0] == (
        "start_frame,end_frame,scene,severity,notes"
    )
    report = (output_dir.parent / "tuning-report.md").read_text()
    assert "M5 tuning evidence" in report
    assert "80-129" in report
    assert "provisional production values" in report


def test_load_review_truth_validates_completed_human_adjudication(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "review.csv"
    fieldnames = [*MATCH_COLUMNS, "truth_name", "review_outcome", "notes"]
    with review_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        known = valid_row()
        known.update(
            truth_name="Harry Potter",
            review_outcome="correct_named",
            notes="clear frontal view",
        )
        writer.writerow(known)
        rejected = valid_row(
            frame_idx=8,
            nearest_name="Ron Weasley",
            distance=0.31,
            assigned_name="Unknown",
        )
        rejected.update(
            truth_name="Ron Weasley",
            review_outcome="known_as_unknown",
            notes="clear side view",
        )
        writer.writerow(rejected)

    assert load_review_truth(review_path) == {
        (7, 0): "Harry Potter",
        (8, 0): "Ron Weasley",
    }


@pytest.mark.parametrize(
    ("truth_name", "outcome", "notes", "message"),
    [
        ("", "", "", "truth_name"),
        ("Dumbledore", "wrong_name", "reviewed", "truth_name"),
        ("Harry Potter", "uncertain", "reviewed", "uncertain"),
        ("Ron Weasley", "correct_named", "reviewed", "correct_named"),
        ("Harry Potter", "known_as_unknown", "reviewed", "known_as_unknown"),
        ("Unknown", "true_unknown", "reviewed", "true_unknown"),
        ("Harry Potter", "wrong_name", "reviewed", "wrong_name"),
        ("Uncertain", "uncertain", "", "notes"),
    ],
)
def test_load_review_truth_rejects_incomplete_or_inconsistent_review(
    tmp_path: Path,
    truth_name: str,
    outcome: str,
    notes: str,
    message: str,
) -> None:
    review_path = tmp_path / "review.csv"
    row = valid_row()
    row.update(truth_name=truth_name, review_outcome=outcome, notes=notes)
    with review_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[*MATCH_COLUMNS, "truth_name", "review_outcome", "notes"],
        )
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match=message):
        load_review_truth(review_path)


def test_compare_cli_writes_spatial_summary_changes_and_contact_sheet(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "fixture.avi"
    base_csv = tmp_path / "base.csv"
    candidate_csv = tmp_path / "candidate.csv"
    output_dir = tmp_path / "analysis"
    write_fixture_video(video_path)
    write_matches(
        base_csv,
        [
            valid_row(frame_idx=0, x=2, y=2, w=6, h=6, distance=0.20),
            valid_row(
                frame_idx=1,
                x=4,
                y=2,
                w=6,
                h=6,
                nearest_name="Ron Weasley",
                distance=0.25,
                assigned_name="Ron Weasley",
            ),
        ],
    )
    write_matches(
        candidate_csv,
        [
            valid_row(
                frame_idx=0,
                x=2,
                y=2,
                w=6,
                h=6,
                nearest_name="Hermione Granger",
                distance=0.22,
                assigned_name="Hermione Granger",
            ),
            valid_row(
                frame_idx=1,
                x=4,
                y=2,
                w=6,
                h=6,
                nearest_name="Ron Weasley",
                distance=0.31,
                assigned_name="Unknown",
            ),
        ],
    )

    result = main(
        [
            "compare",
            "--base-csv",
            str(base_csv),
            "--candidate-csv",
            str(candidate_csv),
            "--video",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--iou-min",
            "0.90",
            "--cell-width",
            "120",
            "--cell-height",
            "100",
            "--columns",
            "2",
        ]
    )

    assert result == 0
    summary = json.loads(
        (output_dir / "normalization-comparison.json").read_text()
    )
    assert summary["matched_count"] == 2
    assert summary["unmatched_base_count"] == 0
    assert summary["unmatched_candidate_count"] == 0
    assert summary["nearest_name_changes"] == 1
    assert summary["assignment_changes"] == 2
    with (output_dir / "normalization-changes.csv").open(newline="") as source:
        changes = list(csv.DictReader(source))
    assert [row["change_type"] for row in changes] == [
        "recognition",
        "recognition",
    ]
    assert (output_dir / "normalization-changes-contact-sheet.png").is_file()


def test_report_cli_scores_reviewed_truth_and_writes_explicit_recommendation(
    tmp_path: Path,
) -> None:
    baseline_summary = tmp_path / "baseline.json"
    review_csv = tmp_path / "review.csv"
    normalization_review = tmp_path / "normalization-review.csv"
    comparison_path = tmp_path / "comparison.json"
    detector_misses = tmp_path / "detector-misses.csv"
    timings_path = tmp_path / "timings.json"
    candidate_face_cache = tmp_path / "candidate-faces.npz"
    candidate_gallery_cache = tmp_path / "candidate-gallery.npz"
    threshold_output = tmp_path / "analysis" / "threshold-sweep.csv"
    report_output = tmp_path / "tuning-report.md"
    baseline_summary.write_text(
        json.dumps(
            {
                "analysis_settings": {"threshold": 0.3, "margin": 0.05},
                "cache_inventory": {
                    "face": {"cached_frames": 3044, "faces": 5353},
                    "gallery": {"photo_count": 17, "owners": {"Harry Potter": 4}},
                },
                "inputs": {
                    "csv": {"path": "output/matches.csv", "sha256": "csv-hash"},
                    "video": {"path": "data/video-source/nimbus.mp4", "sha256": "video-hash"},
                    "face_cache": {"path": "cache/base-face.npz", "sha256": "face-hash"},
                    "gallery_cache": {"path": "cache/base-gallery.npz", "sha256": "gallery-hash"},
                },
                "review": {"selected": 1},
                "sample_windows": [dict(window) for window in M5_SAMPLE_WINDOWS],
                "summary": {
                    "row_count": 5353,
                    "frame_count": 2371,
                    "assignments": {"Harry Potter": 266, "Unknown": 4379},
                    "nearest_names": {"Harry Potter": 1704},
                    "distances": {"min": 0.13, "median": 0.48, "max": 1.06},
                },
                "versions": {"deepface": "0.0.101"},
            }
        ),
        encoding="utf-8",
    )
    fieldnames = [*MATCH_COLUMNS, "truth_name", "review_outcome", "notes"]
    reviewed = valid_row()
    reviewed.update(
        truth_name="Harry Potter",
        review_outcome="correct_named",
        notes="clear view",
    )
    for path in (review_csv, normalization_review):
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(reviewed)
    comparison_path.write_text(
        json.dumps(
            {
                "matched_count": 767,
                "unmatched_base_count": 0,
                "unmatched_candidate_count": 0,
                "iou": {"min": 1.0, "median": 1.0, "mean": 1.0},
                "distance_delta": {"mean": 0.03, "median": 0.02, "mean_absolute": 0.05, "max_absolute": 0.2},
                "nearest_name_changes": 222,
                "assignment_changes": 86,
                "assignment_transitions": {"Harry Potter -> Unknown": 49},
                "per_character": {"Harry Potter": {"matched": 177}},
            }
        ),
        encoding="utf-8",
    )
    detector_misses.write_text(
        "start_frame,end_frame,scene,severity,notes\n80,95,Hall,medium,profile miss\n",
        encoding="utf-8",
    )
    timings_path.write_text(
        json.dumps({"cold_elapsed_seconds": 331.746, "warm_elapsed_seconds": 3.67}),
        encoding="utf-8",
    )
    np.savez_compressed(
        candidate_face_cache,
        meta_json=np.asarray(json.dumps({"normalization": "Facenet2018"})),
        frame_indices=np.arange(300),
        statuses=np.asarray(["ok"] * 300),
        boxes=np.zeros((767, 4), dtype=np.int32),
    )
    np.savez_compressed(
        candidate_gallery_cache,
        meta_json=np.asarray(json.dumps({"normalization": "Facenet2018"})),
        owners=np.asarray(["Harry Potter"] * 17),
    )

    result = main(
        [
            "report",
            "--baseline-summary",
            str(baseline_summary),
            "--review-csv",
            str(review_csv),
            "--comparison",
            str(comparison_path),
            "--normalization-review",
            str(normalization_review),
            "--detector-misses",
            str(detector_misses),
            "--timings",
            str(timings_path),
            "--candidate-face-cache",
            str(candidate_face_cache),
            "--candidate-gallery-cache",
            str(candidate_gallery_cache),
            "--thresholds",
            "0.29",
            "0.30",
            "0.31",
            "--recommended-threshold",
            "0.30",
            "--recommended-normalization",
            "base",
            "--threshold-output",
            str(threshold_output),
            "--output",
            str(report_output),
        ]
    )

    assert result == 0
    with threshold_output.open(newline="") as source:
        threshold_rows = list(csv.DictReader(source))
    assert [row["threshold"] for row in threshold_rows] == ["0.29", "0.3", "0.31"]
    report = report_output.read_text()
    assert "Retain threshold `0.30` and normalization `base`" in report
    assert "5,353" in report
    assert "767" in report
    assert "profile miss" in report
    assert "Facenet2018" in report
    assert "Correctly rejected extra" in report
    assert "0 uncertain crops" in report
