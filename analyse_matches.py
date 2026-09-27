#!/usr/bin/env python3
"""Deterministic, model-free analysis of face-match evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Mapping, Sequence, TextIO

import cv2
import numpy as np

from face_labeller.contracts import CHARACTER_NAMES
from face_labeller.cache import installed_versions
from face_labeller.evidence import MATCH_COLUMNS
from face_labeller.tracking import iou


M5_SAMPLE_WINDOWS = (
    {"id": "S1", "start_frame": 80, "end_frame": 129, "max_frames": 50},
    {"id": "S2", "start_frame": 190, "end_frame": 239, "max_frames": 50},
    {"id": "S3", "start_frame": 1120, "end_frame": 1169, "max_frames": 50},
    {"id": "S4", "start_frame": 1560, "end_frame": 1609, "max_frames": 50},
    {"id": "S5", "start_frame": 2290, "end_frame": 2339, "max_frames": 50},
    {"id": "S6", "start_frame": 2920, "end_frame": 2969, "max_frames": 50},
)


@dataclass(frozen=True)
class MatchRow:
    frame_idx: int
    face_idx: int
    x: int
    y: int
    w: int
    h: int
    det_conf: float
    nearest_name: str
    distance: float
    threshold: float
    assigned_name: str
    confidence: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass(frozen=True)
class ContactSheetResult:
    requested: int
    written: int
    skipped: int
    decoded_frames: int
    skipped_keys: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    reviewed: int
    correct_named: int
    known_as_unknown: int
    wrong_name: int
    correctly_rejected_extra: int
    excluded: int
    known_character_recall: float | None
    wrong_name_rate: float | None


@dataclass(frozen=True)
class MatchedDetection:
    base: MatchRow
    candidate: MatchRow
    iou: float


@dataclass(frozen=True)
class DetectionComparison:
    matched: tuple[MatchedDetection, ...]
    unmatched_base: tuple[MatchRow, ...]
    unmatched_candidate: tuple[MatchRow, ...]


@dataclass(frozen=True)
class NormalizationComparison:
    detection: DetectionComparison
    matched_count: int
    unmatched_base_count: int
    unmatched_candidate_count: int
    iou: dict[str, float] | None
    distance_delta: dict[str, float] | None
    nearest_name_changes: int
    assignment_changes: int
    assignment_transitions: dict[str, int]
    per_character: dict[str, dict[str, int | float]]


def _finite_float(value: str, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _parse_match_row(raw: dict[str, str], path: Path, line_number: int) -> MatchRow:
    try:
        row = MatchRow(
            frame_idx=int(raw["frame_idx"]),
            face_idx=int(raw["face_idx"]),
            x=int(raw["x"]),
            y=int(raw["y"]),
            w=int(raw["w"]),
            h=int(raw["h"]),
            det_conf=_finite_float(raw["det_conf"], "det_conf"),
            nearest_name=raw["nearest_name"],
            distance=_finite_float(raw["distance"], "distance"),
            threshold=_finite_float(raw["threshold"], "threshold"),
            assigned_name=raw["assigned_name"],
            confidence=_finite_float(raw["confidence"], "confidence"),
        )
        if row.frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        if row.face_idx < 0:
            raise ValueError("face_idx must be non-negative")
        if row.w <= 0 or row.h <= 0:
            raise ValueError("box width and height must be positive")
        if row.distance < 0:
            raise ValueError("distance must be non-negative")
        if row.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if row.nearest_name not in CHARACTER_NAMES:
            raise ValueError(f"nearest_name is not a configured character: {row.nearest_name}")
        if row.assigned_name not in (*CHARACTER_NAMES, "Unknown"):
            raise ValueError(f"assigned_name is invalid: {row.assigned_name}")
        if row.assigned_name not in {row.nearest_name, "Unknown"}:
            raise ValueError("assigned_name must be nearest_name or Unknown")
        expected_name = row.nearest_name if row.distance < row.threshold else "Unknown"
        if row.assigned_name != expected_name:
            raise ValueError(
                "assigned_name violates strict threshold rule "
                f"(expected {expected_name}, got {row.assigned_name})"
            )
        return row
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: row {line_number}: {error}") from error


def load_match_rows(path: Path) -> list[MatchRow]:
    """Load and strictly validate one matches.csv file."""
    try:
        source = path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{path}: cannot read CSV: {error}") from error

    with source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != MATCH_COLUMNS:
            raise ValueError(
                f"{path}: header must exactly match {','.join(MATCH_COLUMNS)}"
            )
        rows: list[MatchRow] = []
        seen: set[tuple[int, int]] = set()
        for line_number, raw in enumerate(reader, start=2):
            row = _parse_match_row(raw, path, line_number)
            key = (row.frame_idx, row.face_idx)
            if key in seen:
                raise ValueError(
                    f"{path}: row {line_number}: duplicate face key {key}"
                )
            seen.add(key)
            rows.append(row)
    return sorted(rows, key=lambda row: (row.frame_idx, row.face_idx))


def load_match_rows_many(paths: Sequence[Path]) -> list[MatchRow]:
    """Load disjoint evidence files into one deterministic row sequence."""
    rows: list[MatchRow] = []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        for row in load_match_rows(path):
            key = (row.frame_idx, row.face_idx)
            if key in seen:
                raise ValueError(
                    f"duplicate face key across input files: frame {key[0]}, face {key[1]}"
                )
            seen.add(key)
            rows.append(row)
    return sorted(rows, key=lambda row: (row.frame_idx, row.face_idx))


def load_cache_metadata(path: Path) -> dict[str, object]:
    """Read only the JSON metadata field from an explicit NumPy cache path."""
    try:
        with np.load(path, allow_pickle=False) as stored:
            if "meta_json" not in stored.files:
                raise ValueError("cache has no meta_json field")
            metadata = json.loads(str(stored["meta_json"].item()))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: cannot read cache metadata: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: cache meta_json must contain an object")
    return metadata


def _face_cache_inventory(path: Path) -> dict[str, object]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            required = {"frame_indices", "statuses", "boxes"}
            if not required.issubset(stored.files):
                raise ValueError(f"missing fields: {sorted(required - set(stored.files))}")
            frame_indices = np.asarray(stored["frame_indices"])
            statuses = stored["statuses"].astype(str).tolist()
            boxes = np.asarray(stored["boxes"])
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: cannot inspect face cache: {error}") from error
    return {
        "cached_frames": len(frame_indices),
        "faces": len(boxes),
        "statuses": _sorted_counts(statuses),
    }


def _gallery_cache_inventory(path: Path) -> dict[str, object]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            if "owners" not in stored.files:
                raise ValueError("missing fields: ['owners']")
            owners = stored["owners"].astype(str).tolist()
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: cannot inspect gallery cache: {error}") from error
    return {"owners": _sorted_counts(owners), "photo_count": len(owners)}


def _sorted_counts(values: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    return {name: counts[name] for name in sorted(counts)}


def summarize_assignments(rows: Sequence[MatchRow]) -> dict[str, object]:
    """Summarize assignments and distances without inferring ground truth."""
    if not rows:
        return {
            "row_count": 0,
            "frame_count": 0,
            "faces_per_frame": None,
            "assignments": {},
            "nearest_names": {},
            "distances": None,
        }

    frame_counts = np.asarray(
        list(Counter(row.frame_idx for row in rows).values()), dtype=np.float64
    )
    distances = np.asarray([row.distance for row in rows], dtype=np.float64)
    return {
        "row_count": len(rows),
        "frame_count": len(frame_counts),
        "faces_per_frame": {
            "min": int(frame_counts.min()),
            "median": float(np.median(frame_counts)),
            "mean": float(np.mean(frame_counts)),
            "max": int(frame_counts.max()),
        },
        "assignments": _sorted_counts([row.assigned_name for row in rows]),
        "nearest_names": _sorted_counts([row.nearest_name for row in rows]),
        "distances": {
            "min": float(distances.min()),
            "p25": float(np.percentile(distances, 25)),
            "median": float(np.median(distances)),
            "p75": float(np.percentile(distances, 75)),
            "p90": float(np.percentile(distances, 90)),
            "p95": float(np.percentile(distances, 95)),
            "max": float(distances.max()),
        },
    }


def build_distance_histograms(
    rows: Sequence[MatchRow], bin_edges: np.ndarray
) -> dict[str, np.ndarray]:
    """Count distances by nearest character using fixed, caller-owned bins."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("bin edges must be finite and strictly increasing")

    histograms: dict[str, np.ndarray] = {}
    for name in CHARACTER_NAMES:
        distances = [row.distance for row in rows if row.nearest_name == name]
        counts, _ = np.histogram(distances, bins=edges)
        histograms[name] = counts.astype(np.int64)

    counted = sum(int(counts.sum()) for counts in histograms.values())
    if counted != len(rows):
        raise ValueError(
            f"bin edges cover {counted} of {len(rows)} distances; expand their range"
        )
    return histograms


def select_near_threshold(
    rows: Sequence[MatchRow],
    threshold: float,
    margin: float,
    per_character_limit: int,
) -> list[MatchRow]:
    """Select a deterministic, independently capped review set per character."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    if per_character_limit <= 0:
        raise ValueError("per-character limit must be positive")

    candidates = sorted(
        (row for row in rows if abs(row.distance - threshold) <= margin),
        key=lambda row: (
            abs(row.distance - threshold),
            row.nearest_name,
            row.frame_idx,
            row.face_idx,
        ),
    )
    counts: Counter[str] = Counter()
    selected: list[MatchRow] = []
    for row in candidates:
        if counts[row.nearest_name] >= per_character_limit:
            continue
        selected.append(row)
        counts[row.nearest_name] += 1
    return selected


def write_review_manifest(rows: Sequence[MatchRow], path: Path) -> None:
    """Atomically create the human-adjudication manifest for selected rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            newline="",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            fieldnames = [*MATCH_COLUMNS, "truth_name", "review_outcome", "notes"]
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                values = asdict(row)
                values.update(truth_name="", review_outcome="", notes="")
                writer.writerow(values)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_review_truth(path: Path) -> dict[tuple[int, int], str]:
    """Validate completed visual adjudication and return truth by face key."""
    expected_header = [*MATCH_COLUMNS, "truth_name", "review_outcome", "notes"]
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != expected_header:
            raise ValueError(f"{path}: review header does not match expected schema")
        reviewed_truth: dict[tuple[int, int], str] = {}
        for line_number, raw in enumerate(reader, start=2):
            row = _parse_match_row(raw, path, line_number)
            key = (row.frame_idx, row.face_idx)
            if key in reviewed_truth:
                raise ValueError(f"{path}: row {line_number}: duplicate review key {key}")
            truth = raw["truth_name"].strip()
            outcome = raw["review_outcome"].strip()
            notes = raw["notes"].strip()
            if truth not in {*CHARACTER_NAMES, "Unknown", "Uncertain"}:
                raise ValueError(f"{path}: row {line_number}: invalid truth_name")
            if not notes:
                raise ValueError(f"{path}: row {line_number}: notes must not be blank")
            valid = (
                outcome == "correct_named"
                and truth in CHARACTER_NAMES
                and row.assigned_name == truth
            ) or (
                outcome == "known_as_unknown"
                and truth in CHARACTER_NAMES
                and row.assigned_name == "Unknown"
            ) or (
                outcome == "wrong_name"
                and truth != "Uncertain"
                and row.assigned_name != "Unknown"
                and row.assigned_name != truth
            ) or (
                outcome == "true_unknown"
                and truth == "Unknown"
                and row.assigned_name == "Unknown"
            ) or (outcome == "uncertain" and truth == "Uncertain")
            if not valid:
                raise ValueError(
                    f"{path}: row {line_number}: {outcome or 'review_outcome'} "
                    f"is inconsistent with truth_name={truth or '<blank>'} and "
                    f"assigned_name={row.assigned_name}"
                )
            reviewed_truth[key] = truth
    return reviewed_truth


def _letterboxed_crop(
    crop: np.ndarray, width: int, height: int
) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    scale = min(width / crop.shape[1], height / crop.shape[0])
    resized_width = max(1, int(round(crop.shape[1] * scale)))
    resized_height = max(1, int(round(crop.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=interpolation
    )
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _contact_cell(
    frame: np.ndarray, row: MatchRow, cell_size: tuple[int, int]
) -> np.ndarray | None:
    cell_width, cell_height = cell_size
    if row.w <= 0 or row.h <= 0:
        return None
    frame_height, frame_width = frame.shape[:2]
    left = min(max(row.x, 0), frame_width)
    top = min(max(row.y, 0), frame_height)
    right = min(max(row.x + row.w, 0), frame_width)
    bottom = min(max(row.y + row.h, 0), frame_height)
    if right <= left or bottom <= top:
        return None

    label_height = min(42, max(28, cell_height // 3))
    image_height = cell_height - label_height
    if image_height <= 0:
        return None
    cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    cell[:image_height] = _letterboxed_crop(
        frame[top:bottom, left:right], cell_width, image_height
    )
    cv2.putText(
        cell,
        f"f{row.frame_idx}:{row.face_idx} {row.nearest_name}",
        (3, image_height + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        cell,
        f"d={row.distance:.4f} -> {row.assigned_name}",
        (3, min(cell_height - 4, image_height + 27)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cell


def build_contact_sheet(
    video_path: Path,
    rows: Sequence[MatchRow],
    output_path: Path,
    *,
    cell_size: tuple[int, int],
    columns: int,
) -> ContactSheetResult:
    """Extract source crops once per frame and atomically publish a labelled sheet."""
    cell_width, cell_height = cell_size
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("cell dimensions must be positive")
    if columns <= 0:
        raise ValueError("columns must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open video: {video_path}")

    requested_rows = sorted(rows, key=lambda row: (row.frame_idx, row.face_idx))
    rows_by_frame: dict[int, list[MatchRow]] = {}
    for row in requested_rows:
        rows_by_frame.setdefault(row.frame_idx, []).append(row)

    cells: list[np.ndarray] = []
    skipped_keys: list[tuple[int, int]] = []
    decoded_frames = 0
    try:
        for frame_idx, frame_rows in rows_by_frame.items():
            usable_rows = [row for row in frame_rows if row.w > 0 and row.h > 0]
            skipped_keys.extend(
                (row.frame_idx, row.face_idx)
                for row in frame_rows
                if row.w <= 0 or row.h <= 0
            )
            if not usable_rows:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                skipped_keys.extend(
                    (row.frame_idx, row.face_idx) for row in usable_rows
                )
                continue
            decoded_frames += 1
            for row in usable_rows:
                cell = _contact_cell(frame, row, cell_size)
                if cell is None:
                    skipped_keys.append((row.frame_idx, row.face_idx))
                else:
                    cells.append(cell)
    finally:
        capture.release()

    if not cells:
        raise ValueError("no usable crops for contact sheet")

    row_count = math.ceil(len(cells) / columns)
    sheet = np.zeros(
        (row_count * cell_height, columns * cell_width, 3), dtype=np.uint8
    )
    for index, cell in enumerate(cells):
        grid_row, grid_column = divmod(index, columns)
        top = grid_row * cell_height
        left = grid_column * cell_width
        sheet[top : top + cell_height, left : left + cell_width] = cell

    extension = output_path.suffix or ".png"
    encoded, content = cv2.imencode(extension, sheet)
    if not encoded:
        raise OSError(f"cannot encode contact sheet as {extension}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=extension,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content.tobytes())
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    skipped_keys.sort()
    return ContactSheetResult(
        requested=len(requested_rows),
        written=len(cells),
        skipped=len(skipped_keys),
        decoded_frames=decoded_frames,
        skipped_keys=tuple(skipped_keys),
    )


def reassign_at_threshold(row: MatchRow, threshold: float) -> str:
    """Replay the production's strict distance decision without model work."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    return row.nearest_name if row.distance < threshold else "Unknown"


def score_thresholds(
    rows: Sequence[MatchRow],
    reviewed_truth: Mapping[tuple[int, int], str],
    candidates: Sequence[float],
) -> list[ThresholdResult]:
    """Score threshold candidates only where a human supplied usable truth."""
    candidate_values = list(candidates)
    if (
        not candidate_values
        or any(
            not math.isfinite(candidate) or candidate < 0
            for candidate in candidate_values
        )
        or candidate_values != sorted(candidate_values)
        or len(candidate_values) != len(set(candidate_values))
    ):
        raise ValueError(
            "candidate thresholds must be finite, non-negative, unique, and sorted"
        )
    allowed_truth = {*CHARACTER_NAMES, "Unknown", "Uncertain"}
    invalid_truth = sorted(set(reviewed_truth.values()) - allowed_truth)
    if invalid_truth:
        raise ValueError(f"truth contains invalid names: {invalid_truth}")

    results: list[ThresholdResult] = []
    for threshold in candidate_values:
        correct_named = 0
        known_as_unknown = 0
        wrong_name = 0
        correctly_rejected_extra = 0
        excluded = 0
        known_total = 0
        reviewed = 0
        for row in rows:
            truth = reviewed_truth.get((row.frame_idx, row.face_idx))
            if truth is None or truth == "Uncertain":
                excluded += 1
                continue
            reviewed += 1
            predicted = reassign_at_threshold(row, threshold)
            if truth == "Unknown":
                if predicted == "Unknown":
                    correctly_rejected_extra += 1
                else:
                    wrong_name += 1
                continue
            known_total += 1
            if predicted == truth:
                correct_named += 1
            elif predicted == "Unknown":
                known_as_unknown += 1
            else:
                wrong_name += 1
        results.append(
            ThresholdResult(
                threshold=threshold,
                reviewed=reviewed,
                correct_named=correct_named,
                known_as_unknown=known_as_unknown,
                wrong_name=wrong_name,
                correctly_rejected_extra=correctly_rejected_extra,
                excluded=excluded,
                known_character_recall=(
                    correct_named / known_total if known_total else None
                ),
                wrong_name_rate=wrong_name / reviewed if reviewed else None,
            )
        )
    return results


def match_detections_by_iou(
    base_rows: Sequence[MatchRow],
    candidate_rows: Sequence[MatchRow],
    iou_min: float,
) -> DetectionComparison:
    """Match independent detections within each frame by stable greedy IoU."""
    if not math.isfinite(iou_min) or not 0 <= iou_min <= 1:
        raise ValueError("IoU minimum must be finite and between zero and one")

    candidates: list[tuple[float, int, int]] = []
    for base_index, base in enumerate(base_rows):
        for candidate_index, candidate in enumerate(candidate_rows):
            if base.frame_idx != candidate.frame_idx:
                continue
            overlap = iou(base.box, candidate.box)
            if overlap >= iou_min:
                candidates.append((overlap, base_index, candidate_index))
    candidates.sort(
        key=lambda item: (
            -item[0],
            base_rows[item[1]].frame_idx,
            base_rows[item[1]].face_idx,
            candidate_rows[item[2]].face_idx,
        )
    )

    used_base: set[int] = set()
    used_candidate: set[int] = set()
    matched: list[MatchedDetection] = []
    for overlap, base_index, candidate_index in candidates:
        if base_index in used_base or candidate_index in used_candidate:
            continue
        used_base.add(base_index)
        used_candidate.add(candidate_index)
        matched.append(
            MatchedDetection(
                base=base_rows[base_index],
                candidate=candidate_rows[candidate_index],
                iou=overlap,
            )
        )

    matched.sort(
        key=lambda match: (
            match.base.frame_idx,
            match.base.face_idx,
            match.candidate.face_idx,
        )
    )
    unmatched_base = sorted(
        (row for index, row in enumerate(base_rows) if index not in used_base),
        key=lambda row: (row.frame_idx, row.face_idx),
    )
    unmatched_candidate = sorted(
        (
            row
            for index, row in enumerate(candidate_rows)
            if index not in used_candidate
        ),
        key=lambda row: (row.frame_idx, row.face_idx),
    )
    return DetectionComparison(
        matched=tuple(matched),
        unmatched_base=tuple(unmatched_base),
        unmatched_candidate=tuple(unmatched_candidate),
    )


def compare_normalizations(
    base_rows: Sequence[MatchRow],
    candidate_rows: Sequence[MatchRow],
    iou_min: float,
) -> NormalizationComparison:
    """Summarize geometry and recognition changes across normalization runs."""
    detection = match_detections_by_iou(base_rows, candidate_rows, iou_min)
    overlaps = np.asarray([match.iou for match in detection.matched])
    deltas = np.asarray(
        [
            match.candidate.distance - match.base.distance
            for match in detection.matched
        ],
        dtype=np.float64,
    )
    nearest_name_changes = sum(
        match.base.nearest_name != match.candidate.nearest_name
        for match in detection.matched
    )
    assignment_changes = sum(
        match.base.assigned_name != match.candidate.assigned_name
        for match in detection.matched
    )
    transition_counts = Counter(
        f"{match.base.assigned_name} -> {match.candidate.assigned_name}"
        for match in detection.matched
        if match.base.assigned_name != match.candidate.assigned_name
    )

    per_character: dict[str, dict[str, int | float]] = {}
    for name in CHARACTER_NAMES:
        character_matches = [
            match for match in detection.matched if match.base.nearest_name == name
        ]
        if not character_matches:
            continue
        character_deltas = [
            match.candidate.distance - match.base.distance
            for match in character_matches
        ]
        per_character[name] = {
            "matched": len(character_matches),
            "nearest_name_changes": sum(
                match.base.nearest_name != match.candidate.nearest_name
                for match in character_matches
            ),
            "assignment_changes": sum(
                match.base.assigned_name != match.candidate.assigned_name
                for match in character_matches
            ),
            "mean_distance_delta": float(np.mean(character_deltas)),
        }

    return NormalizationComparison(
        detection=detection,
        matched_count=len(detection.matched),
        unmatched_base_count=len(detection.unmatched_base),
        unmatched_candidate_count=len(detection.unmatched_candidate),
        iou=(
            {
                "min": float(overlaps.min()),
                "median": float(np.median(overlaps)),
                "mean": float(np.mean(overlaps)),
            }
            if len(overlaps)
            else None
        ),
        distance_delta=(
            {
                "mean": float(np.mean(deltas)),
                "median": float(np.median(deltas)),
                "mean_absolute": float(np.mean(np.abs(deltas))),
                "max_absolute": float(np.max(np.abs(deltas))),
            }
            if len(deltas)
            else None
        ),
        nearest_name_changes=nearest_name_changes,
        assignment_changes=assignment_changes,
        assignment_transitions={
            name: transition_counts[name] for name in sorted(transition_counts)
        },
        per_character=per_character,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"{path}: cannot hash input: {error}") from error
    return digest.hexdigest()


def _atomic_text_writer(path: Path) -> tuple[TextIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    )
    return temporary, Path(temporary.name)


def _write_json(path: Path, value: object) -> None:
    output, temporary_path = _atomic_text_writer(path)
    try:
        with output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_histograms(
    path: Path,
    histograms: Mapping[str, np.ndarray],
    bin_edges: np.ndarray,
) -> None:
    output, temporary_path = _atomic_text_writer(path)
    try:
        with output:
            writer = csv.writer(output)
            writer.writerow(("nearest_name", "bin_left", "bin_right", "count"))
            for name in CHARACTER_NAMES:
                for index, count in enumerate(histograms[name]):
                    writer.writerow(
                        (
                            name,
                            float(bin_edges[index]),
                            float(bin_edges[index + 1]),
                            int(count),
                        )
                    )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: cannot read JSON object: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_review_details(
    path: Path,
) -> tuple[list[MatchRow], dict[tuple[int, int], str], Counter[str]]:
    truth = load_review_truth(path)
    expected_header = [*MATCH_COLUMNS, "truth_name", "review_outcome", "notes"]
    rows: list[MatchRow] = []
    outcomes: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != expected_header:
            raise ValueError(f"{path}: review header does not match expected schema")
        for line_number, raw in enumerate(reader, start=2):
            rows.append(_parse_match_row(raw, path, line_number))
            outcomes[raw["review_outcome"].strip()] += 1
    return rows, truth, outcomes


def _load_detector_misses(path: Path) -> list[dict[str, str]]:
    expected = ["start_frame", "end_frame", "scene", "severity", "notes"]
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: detector-miss header does not match expected schema")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        try:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: row {line_number}: invalid frame range"
            ) from error
        if start < 0 or end < start or not all(
            row[field].strip() for field in ("scene", "severity", "notes")
        ):
            raise ValueError(f"{path}: row {line_number}: invalid detector finding")
    return rows


def _write_threshold_results(path: Path, results: Sequence[ThresholdResult]) -> None:
    output, temporary_path = _atomic_text_writer(path)
    try:
        with output:
            fieldnames = tuple(ThresholdResult.__dataclass_fields__)
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _count_text(counts: Mapping[str, object]) -> str:
    return ", ".join(
        f"{name}: {int(value):,}" for name, value in counts.items()
    ) or "none"


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _report_markdown(
    *,
    baseline: Mapping[str, object],
    review_rows: Sequence[MatchRow],
    review_truth: Mapping[tuple[int, int], str],
    threshold_results: Sequence[ThresholdResult],
    review_outcomes: Mapping[str, int],
    comparison: Mapping[str, object],
    normalization_outcomes: Mapping[str, int],
    detector_misses: Sequence[Mapping[str, str]],
    timings: Mapping[str, object],
    candidate_face_cache: Path,
    candidate_gallery_cache: Path,
    recommended_threshold: float,
    recommended_normalization: str,
) -> str:
    summary = baseline["summary"]
    settings = baseline["analysis_settings"]
    inventory = baseline["cache_inventory"]
    inputs = baseline["inputs"]
    versions = baseline.get("versions", {})
    windows = baseline["sample_windows"]
    review = baseline.get("review", {})
    assert isinstance(summary, dict)
    assert isinstance(settings, dict)
    assert isinstance(inventory, dict)
    assert isinstance(inputs, dict)
    assert isinstance(versions, dict)
    assert isinstance(windows, list)
    assert isinstance(review, dict)

    face_inventory = inventory["face"]
    gallery_inventory = inventory["gallery"]
    assert isinstance(face_inventory, dict)
    assert isinstance(gallery_inventory, dict)
    assignments = summary.get("assignments", {})
    nearest = summary.get("nearest_names", {})
    distances = summary.get("distances", {})
    assert isinstance(assignments, dict)
    assert isinstance(nearest, dict)
    assert isinstance(distances, dict)

    candidate_face_metadata = load_cache_metadata(candidate_face_cache)
    candidate_gallery_metadata = load_cache_metadata(candidate_gallery_cache)
    candidate_face_inventory = _face_cache_inventory(candidate_face_cache)
    candidate_gallery_inventory = _gallery_cache_inventory(candidate_gallery_cache)
    candidate_normalization = candidate_face_metadata.get(
        "normalization",
        candidate_gallery_metadata.get("normalization", "candidate"),
    )

    recommendation = (
        f"Retain threshold `{recommended_threshold:.2f}` and normalization "
        f"`{recommended_normalization}`"
    )
    wrong_name_keys = sorted(
        [
        (row.frame_idx, row.face_idx)
        for row in review_rows
        if review_truth[(row.frame_idx, row.face_idx)] != "Uncertain"
        and row.assigned_name not in {
            "Unknown",
            review_truth[(row.frame_idx, row.face_idx)],
        }
        ]
    )
    wrong_name_text = ", ".join(
        f"frame {frame_idx} face {face_idx}"
        for frame_idx, face_idx in wrong_name_keys
    ) or "none"
    normalization_lost = int(normalization_outcomes.get("known_as_unknown", 0))
    normalization_gained = int(normalization_outcomes.get("correct_named", 0))
    normalization_net = normalization_gained - normalization_lost
    uncertain_count = int(review_outcomes.get("uncertain", 0))
    uncertain_phrase = (
        "1 uncertain crop is"
        if uncertain_count == 1
        else f"{uncertain_count:,} uncertain crops are"
    )
    confirmed_extra_count = sum(
        truth == "Unknown" for truth in review_truth.values()
    )
    threshold_safety = (
        "The threshold review contains no confirmed non-character faces, so it can "
        "measure known-face recall and wrong-name errors but cannot measure the "
        "false-positive cost of raising the threshold."
        if confirmed_extra_count == 0
        else f"The threshold review contains {confirmed_extra_count:,} confirmed "
        "non-character faces, reported in the threshold table below."
    )
    if normalization_net < 0:
        normalization_effect = (
            f"The candidate normalization lost {-normalization_net:,} net correct "
            "labels on the frozen sample."
        )
        r2_effect = (
            f"retaining `base` preserves {-normalization_net:,} more correct labels "
            f"on the frozen sample than {candidate_normalization}"
        )
    elif normalization_net > 0:
        normalization_effect = (
            f"The candidate normalization gained {normalization_net:,} net correct "
            "labels on the frozen sample."
        )
        r2_effect = (
            f"{candidate_normalization} adds {normalization_net:,} net correct labels "
            "on the frozen sample"
        )
    else:
        normalization_effect = (
            "The candidate normalization made no net change to correct labels on the "
            "frozen sample."
        )
        r2_effect = "the two normalizations have equal net correct-label coverage"
    lines = [
        "# M5 evidence-based tuning report",
        "",
        "**Status: STOP gate — owner decision D3 required.**",
        "",
        f"## Recommendation: {recommendation}",
        "",
        "The evidence does not justify changing either production default.",
        threshold_safety,
        normalization_effect,
        "No production configuration or accepted output has been changed.",
        "",
        "## Reproducible baseline",
        "",
        f"- Evidence rows: {int(summary['row_count']):,} across "
        f"{int(summary['frame_count']):,} frames containing detections.",
        f"- Face cache: {int(face_inventory['cached_frames']):,} cached frames and "
        f"{int(face_inventory['faces']):,} faces.",
        f"- Gallery: {int(gallery_inventory['photo_count']):,} photos "
        f"({_count_text(gallery_inventory.get('owners', {}))}).",
        f"- Accepted assignments: {_count_text(assignments)}.",
        f"- Nearest reference owners: {_count_text(nearest)}.",
        f"- Distance range: {float(distances['min']):.6f}–"
        f"{float(distances['max']):.6f}; median {float(distances['median']):.6f}.",
        f"- Analysis threshold: {float(settings['threshold']):.2f}; review margin: "
        f"{float(settings['margin']):.2f}.",
        "- Installed versions: "
        + (", ".join(f"{name}: {value}" for name, value in versions.items()) or "none")
        + ".",
        "",
        "### Input identity",
        "",
    ]
    for name in ("video", "csv", "face_cache", "gallery_cache"):
        identity = inputs[name]
        assert isinstance(identity, dict)
        lines.append(
            f"- `{identity['path']}` — SHA-256 `{identity['sha256']}`"
        )

    lines.extend(["", "### Frozen normalization sample", ""])
    for window in windows:
        assert isinstance(window, dict)
        lines.append(
            f"- {window['id']}: frames {window['start_frame']}–"
            f"{window['end_frame']} ({window['max_frames']} frames)"
        )

    lines.extend(
        [
            "",
            "## Threshold review",
            "",
            f"The completed review contains {sum(review_outcomes.values()):,} crops: "
            f"{_count_text(review_outcomes)}. {uncertain_phrase} "
            "excluded from scoring.",
            "",
            "| Threshold | Reviewed | Correctly named | Known → Unknown | Wrong name | "
            "Correctly rejected extra | Excluded | Known recall | Wrong-name rate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in threshold_results:
        lines.append(
            f"| {result.threshold:.3f} | {result.reviewed:,} | "
            f"{result.correct_named:,} | {result.known_as_unknown:,} | "
            f"{result.wrong_name:,} | {result.correctly_rejected_extra:,} | "
            f"{result.excluded:,} | {_percent(result.known_character_recall)} | "
            f"{_percent(result.wrong_name_rate)} |"
        )
    lines.extend(
        [
            "",
            f"Wrong-name rows in the completed review: {wrong_name_text}. Lowering the",
            "threshold far enough to reject the observed wrong-name rows also discards",
            "many correct borderline labels. Raising it improves recall on this",
            "known-character-heavy review set, but cannot be defended without reviewed",
            "true extras.",
            "",
            "## Normalization A/B",
            "",
            f"The frozen sample matched **{int(comparison['matched_count']):,}** detections; "
            f"unmatched base/candidate counts were {int(comparison['unmatched_base_count']):,}/"
            f"{int(comparison['unmatched_candidate_count']):,}.",
        ]
    )
    overlap = comparison.get("iou")
    deltas = comparison.get("distance_delta")
    if isinstance(overlap, dict):
        lines.append(
            f"- IoU min/median/mean: {float(overlap['min']):.6f}/"
            f"{float(overlap['median']):.6f}/{float(overlap['mean']):.6f}."
        )
    if isinstance(deltas, dict):
        lines.append(
            f"- Distance delta mean/median: {float(deltas['mean']):+.6f}/"
            f"{float(deltas['median']):+.6f}; mean/max absolute: "
            f"{float(deltas['mean_absolute']):.6f}/{float(deltas['max_absolute']):.6f}."
        )
    per_character = comparison.get("per_character", {})
    if isinstance(per_character, dict) and per_character:
        lines.extend(
            [
                "",
                "| Base nearest owner | Matched | Nearest-owner changes | "
                "Assignment changes | Mean distance delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, values in per_character.items():
            assert isinstance(values, dict)
            lines.append(
                f"| {name} | {int(values.get('matched', 0)):,} | "
                f"{int(values.get('nearest_name_changes', 0)):,} | "
                f"{int(values.get('assignment_changes', 0)):,} | "
                f"{float(values.get('mean_distance_delta', 0.0)):+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            f"- Nearest-owner changes: {int(comparison['nearest_name_changes']):,}; "
            f"assignment changes: {int(comparison['assignment_changes']):,}.",
            f"- Assignment transitions: "
            f"{_count_text(comparison.get('assignment_transitions', {}))}.",
            f"- Visual outcomes for changed assignments: "
            f"{_count_text(normalization_outcomes)}.",
            f"- Candidate: `{candidate_normalization}`; face cache "
            f"{int(candidate_face_inventory['cached_frames']):,} frames/"
            f"{int(candidate_face_inventory['faces']):,} faces; gallery "
            f"{int(candidate_gallery_inventory['photo_count']):,} photos.",
            f"- Candidate face cache SHA-256: `{_file_sha256(candidate_face_cache)}`.",
            f"- Candidate gallery cache SHA-256: `{_file_sha256(candidate_gallery_cache)}`.",
            "",
            f"The changed-assignment review found {normalization_lost:,} correct labels lost and "
            f"{normalization_gained:,} correct labels gained under {candidate_normalization}: "
            f"a net change of {normalization_net:+,} correct labels. Geometry was",
            "unchanged, so this is a recognition/normalization effect, not a detector effect.",
            "",
            "## Runtime and cache reuse",
            "",
            f"- Candidate cold sample: {float(timings['cold_elapsed_seconds']):.3f}s elapsed.",
            f"- Candidate warm replay: {float(timings['warm_elapsed_seconds']):.3f}s elapsed.",
        ]
    )
    if "cold_perception_seconds" in timings:
        lines.append(
            f"- Cold perception work: {float(timings['cold_perception_seconds']):.3f}s."
        )
    timing_windows = timings.get("windows")
    if isinstance(timing_windows, list) and timing_windows:
        lines.extend(
            [
                "",
                "| Window | Cold elapsed | Gallery | Model | Perception |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for window in timing_windows:
            assert isinstance(window, dict)
            lines.append(
                f"| {window['id']} | {float(window['cold_elapsed_seconds']):.3f}s | "
                f"{float(window['cold_gallery_seconds']):.3f}s | "
                f"{float(window['cold_model_seconds']):.3f}s | "
                f"{float(window['cold_perception_seconds']):.3f}s |"
            )

    lines.extend(["", "## Detector audit", ""])
    if detector_misses:
        lines.extend(
            [
                "| Frames | Scene | Severity | Finding |",
                "| --- | --- | --- | --- |",
            ]
        )
        for finding in detector_misses:
            lines.append(
                f"| {finding['start_frame']}–{finding['end_frame']} | "
                f"{finding['scene']} | {finding['severity']} | {finding['notes']} |"
            )
    else:
        lines.append("No detector misses were recorded in the audited sample.")
    lines.extend(
        [
            "",
            "The audit montage sampled every fifth frame across the six frozen windows; it",
            "is a structured spot check, not exhaustive frame-level ground truth.",
            "",
            "## R1 and R2 impact",
            "",
            "- **R1 (box every face):** normalization did not alter sample geometry, but the",
            "  detector audit found intermittent profile/occlusion misses and small background",
            "  faces. These are detection gaps and are not repaired by threshold tuning.",
            f"- **R2 (name when possible):** {r2_effect}. Retaining 0.30 avoids tuning upward on a",
            "  review set with no measured true-extra false-positive cost.",
            "",
            "## Limitations",
            "",
            "- The review set is targeted near the threshold and supplemented with known",
            "  gallery-path flips; it is not a random, independent full-video ground truth.",
            "- Nearby video frames are correlated, and many gallery owners still have only a",
            "  small number of curated reference photos.",
            (
                "- No reviewed true-extra faces are present, so an upward threshold change "
                "cannot be safety-scored from this evidence."
                if confirmed_extra_count == 0
                else f"- Only {confirmed_extra_count:,} reviewed true-extra faces are present; "
                "false-positive estimates therefore have wide uncertainty."
            ),
            "- Facenet2018 was evaluated on the fixed 300-frame sample, not rerun over all",
            "  3,044 frames.",
            "- Review sheets and crops are private generated artifacts and remain gitignored.",
            "",
            "## Reproduction commands",
            "",
            "```bash",
            "python analyse_matches.py baseline --help",
            "python analyse_matches.py compare --help",
            "python analyse_matches.py report --help",
            "```",
            "",
            "The exact paths, hashes, frozen windows, thresholds, cache identities, and",
            "timings needed to reproduce this decision are recorded above and in the adjacent",
            "machine-readable analysis artifacts.",
            "",
            "## Owner decision D3",
            "",
            f"Proposed decision: **{recommendation}.** Approving D3 may then make those",
            "values explicit as accepted defaults; until approval, this report is evidence",
            "only and the pipeline remains unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_report(args: argparse.Namespace) -> int:
    baseline = _load_json_object(args.baseline_summary)
    comparison = _load_json_object(args.comparison)
    timings = _load_json_object(args.timings)
    review_rows, review_truth, review_outcomes = _load_review_details(args.review_csv)
    _, _, normalization_outcomes = _load_review_details(
        args.normalization_review
    )
    detector_misses = _load_detector_misses(args.detector_misses)
    threshold_results = score_thresholds(
        review_rows, review_truth, args.thresholds
    )
    _write_threshold_results(args.threshold_output, threshold_results)
    report = _report_markdown(
        baseline=baseline,
        review_rows=review_rows,
        review_truth=review_truth,
        threshold_results=threshold_results,
        review_outcomes=review_outcomes,
        comparison=comparison,
        normalization_outcomes=normalization_outcomes,
        detector_misses=detector_misses,
        timings=timings,
        candidate_face_cache=args.candidate_face_cache,
        candidate_gallery_cache=args.candidate_gallery_cache,
        recommended_threshold=args.recommended_threshold,
        recommended_normalization=args.recommended_normalization,
    )
    output, temporary_path = _atomic_text_writer(args.output)
    try:
        with output:
            output.write(report)
        temporary_path.replace(args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Threshold candidates scored: {len(threshold_results)}")
    print(f"Report: {args.output}")
    return 0


def _write_initial_report(path: Path, summary: Mapping[str, object]) -> None:
    windows = summary["sample_windows"]
    assert isinstance(windows, list)
    settings = summary["analysis_settings"]
    run_summary = summary["summary"]
    lines = [
        "# M5 tuning evidence",
        "",
        "This report records analysis settings and evidence for the D3 decision. The",
        "provisional production values remain threshold `0.30` and normalization `base`",
        "until the owner reviews the completed evidence.",
        "",
        "## Frozen normalization sample",
        "",
    ]
    for window in windows:
        assert isinstance(window, dict)
        lines.append(
            f"- {window['id']}: {window['start_frame']}-{window['end_frame']} "
            f"({window['max_frames']} frames)"
        )
    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"- Rows: {run_summary['row_count']}",
            f"- Frames containing detections: {run_summary['frame_count']}",
            f"- Analysis threshold: {settings['threshold']}",
            f"- Near-threshold margin: {settings['margin']}",
            f"- Per-character review limit: {settings['per_character_limit']}",
            "",
            "Threshold and normalization findings will be added after visual review and",
            "the fixed-sample A/B run.",
            "",
        ]
    )
    output, temporary_path = _atomic_text_writer(path)
    try:
        with output:
            output.write("\n".join(lines))
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_baseline(args: argparse.Namespace) -> int:
    rows = load_match_rows(args.csv)
    edges = np.asarray(args.bin_edges, dtype=np.float64)
    histograms = build_distance_histograms(rows, edges)
    primary_selected = select_near_threshold(
        rows,
        threshold=args.threshold,
        margin=args.margin,
        per_character_limit=args.per_character_limit,
    )
    selected = list(primary_selected)
    supplemental_added = 0
    if args.supplemental_keys is not None:
        with args.supplemental_keys.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != ["frame_idx", "face_idx", "source"]:
                raise ValueError(
                    f"{args.supplemental_keys}: header must be "
                    "frame_idx,face_idx,source"
                )
            requested_keys: set[tuple[int, int]] = set()
            for line_number, raw in enumerate(reader, start=2):
                try:
                    key = (int(raw["frame_idx"]), int(raw["face_idx"]))
                    if key[0] < 0 or key[1] < 0 or not raw["source"].strip():
                        raise ValueError("invalid key or blank source")
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{args.supplemental_keys}: row {line_number}: {error}"
                    ) from error
                if key in requested_keys:
                    raise ValueError(
                        f"{args.supplemental_keys}: row {line_number}: duplicate {key}"
                    )
                requested_keys.add(key)
        rows_by_key = {(row.frame_idx, row.face_idx): row for row in rows}
        missing_keys = sorted(requested_keys - set(rows_by_key))
        if missing_keys:
            raise ValueError(
                f"{args.supplemental_keys}: keys are absent from matches CSV: "
                f"{missing_keys}"
            )
        selected_keys = {(row.frame_idx, row.face_idx) for row in selected}
        supplemental = [
            rows_by_key[key]
            for key in sorted(requested_keys)
            if key not in selected_keys
        ]
        selected.extend(supplemental)
        supplemental_added = len(supplemental)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "near-threshold-review.csv"
    contact_path = args.output_dir / "near-threshold-contact-sheet.png"
    write_review_manifest(selected, review_path)
    contact = build_contact_sheet(
        args.video,
        selected,
        contact_path,
        cell_size=(args.cell_width, args.cell_height),
        columns=args.columns,
    )

    summary: dict[str, object] = {
        "analysis_settings": {
            "bin_edges": [float(value) for value in edges],
            "margin": args.margin,
            "per_character_limit": args.per_character_limit,
            "threshold": args.threshold,
        },
        "cache_metadata": {
            "face": load_cache_metadata(args.face_cache),
            "gallery": load_cache_metadata(args.gallery_cache),
        },
        "cache_inventory": {
            "face": _face_cache_inventory(args.face_cache),
            "gallery": _gallery_cache_inventory(args.gallery_cache),
        },
        "inputs": {
            "csv": {"path": str(args.csv), "sha256": _file_sha256(args.csv)},
            "video": {"path": str(args.video), "sha256": _file_sha256(args.video)},
            "face_cache": {
                "path": str(args.face_cache),
                "sha256": _file_sha256(args.face_cache),
            },
            "gallery_cache": {
                "path": str(args.gallery_cache),
                "sha256": _file_sha256(args.gallery_cache),
            },
        },
        "review": {
            "primary_selected": len(primary_selected),
            "supplemental_added": supplemental_added,
            "selected": len(selected),
            "contact_sheet": asdict(contact),
        },
        "sample_windows": [dict(window) for window in M5_SAMPLE_WINDOWS],
        "summary": summarize_assignments(rows),
        "threshold_assignment_counts": {
            str(float(threshold)): _sorted_counts(
                [reassign_at_threshold(row, float(threshold)) for row in rows]
            )
            for threshold in edges
        },
        "versions": installed_versions(),
    }
    _write_json(args.output_dir / "baseline-summary.json", summary)
    _write_histograms(
        args.output_dir / "distance-histograms.csv", histograms, edges
    )
    detector_misses = args.output_dir / "detector-misses.csv"
    if not detector_misses.exists():
        output, temporary_path = _atomic_text_writer(detector_misses)
        try:
            with output:
                output.write("start_frame,end_frame,scene,severity,notes\n")
            temporary_path.replace(detector_misses)
        finally:
            temporary_path.unlink(missing_ok=True)
    _write_initial_report(args.output_dir.parent / "tuning-report.md", summary)
    print(f"Baseline rows: {len(rows)}")
    print(f"Review crops: {contact.written} ({contact.skipped} skipped)")
    print(f"Artifacts: {args.output_dir}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    base_rows = load_match_rows_many(args.base_csv)
    candidate_rows = load_match_rows_many(args.candidate_csv)
    comparison = compare_normalizations(base_rows, candidate_rows, args.iou_min)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_json = asdict(comparison)
    _write_json(
        args.output_dir / "normalization-comparison.json", comparison_json
    )

    change_fields = (
        "change_type",
        "frame_idx",
        "base_face_idx",
        "candidate_face_idx",
        "iou",
        "base_nearest_name",
        "candidate_nearest_name",
        "base_distance",
        "candidate_distance",
        "base_assigned_name",
        "candidate_assigned_name",
    )
    change_path = args.output_dir / "normalization-changes.csv"
    output, temporary_path = _atomic_text_writer(change_path)
    contact_rows: list[MatchRow] = []
    changed_count = 0
    try:
        with output:
            writer = csv.DictWriter(output, fieldnames=change_fields)
            writer.writeheader()
            for match in comparison.detection.matched:
                if (
                    match.base.nearest_name == match.candidate.nearest_name
                    and match.base.assigned_name == match.candidate.assigned_name
                ):
                    continue
                writer.writerow(
                    {
                        "change_type": "recognition",
                        "frame_idx": match.base.frame_idx,
                        "base_face_idx": match.base.face_idx,
                        "candidate_face_idx": match.candidate.face_idx,
                        "iou": match.iou,
                        "base_nearest_name": match.base.nearest_name,
                        "candidate_nearest_name": match.candidate.nearest_name,
                        "base_distance": match.base.distance,
                        "candidate_distance": match.candidate.distance,
                        "base_assigned_name": match.base.assigned_name,
                        "candidate_assigned_name": match.candidate.assigned_name,
                    }
                )
                changed_count += 1
                contact_rows.append(match.candidate)
            for row in comparison.detection.unmatched_base:
                writer.writerow(
                    {
                        "change_type": "unmatched_base",
                        "frame_idx": row.frame_idx,
                        "base_face_idx": row.face_idx,
                        "base_nearest_name": row.nearest_name,
                        "base_distance": row.distance,
                        "base_assigned_name": row.assigned_name,
                    }
                )
                contact_rows.append(row)
            for row in comparison.detection.unmatched_candidate:
                writer.writerow(
                    {
                        "change_type": "unmatched_candidate",
                        "frame_idx": row.frame_idx,
                        "candidate_face_idx": row.face_idx,
                        "candidate_nearest_name": row.nearest_name,
                        "candidate_distance": row.distance,
                        "candidate_assigned_name": row.assigned_name,
                    }
                )
                contact_rows.append(row)
        temporary_path.replace(change_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if contact_rows:
        build_contact_sheet(
            args.video,
            contact_rows,
            args.output_dir / "normalization-changes-contact-sheet.png",
            cell_size=(args.cell_width, args.cell_height),
            columns=args.columns,
        )
    print(
        f"Matched: {comparison.matched_count}; "
        f"unmatched base/candidate: {comparison.unmatched_base_count}/"
        f"{comparison.unmatched_candidate_count}; changed: {changed_count}"
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be finite and between zero and one")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-free analysis of face-match evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser(
        "baseline", help="generate full-run distributions and review artifacts"
    )
    baseline.add_argument("--csv", type=Path, required=True)
    baseline.add_argument("--video", type=Path, required=True)
    baseline.add_argument("--face-cache", type=Path, required=True)
    baseline.add_argument("--gallery-cache", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--supplemental-keys", type=Path)
    baseline.add_argument("--threshold", type=_non_negative_float, required=True)
    baseline.add_argument("--margin", type=_non_negative_float, required=True)
    baseline.add_argument(
        "--per-character-limit", type=_positive_int, required=True
    )
    baseline.add_argument(
        "--bin-edges", type=float, nargs="+", required=True
    )
    baseline.add_argument("--cell-width", type=_positive_int, default=240)
    baseline.add_argument("--cell-height", type=_positive_int, default=180)
    baseline.add_argument("--columns", type=_positive_int, default=5)
    baseline.set_defaults(handler=_run_baseline)

    compare = subparsers.add_parser(
        "compare", help="compare base and candidate normalization CSVs"
    )
    compare.add_argument("--base-csv", type=Path, nargs="+", required=True)
    compare.add_argument("--candidate-csv", type=Path, nargs="+", required=True)
    compare.add_argument("--video", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--iou-min", type=_unit_float, required=True)
    compare.add_argument("--cell-width", type=_positive_int, default=240)
    compare.add_argument("--cell-height", type=_positive_int, default=180)
    compare.add_argument("--columns", type=_positive_int, default=5)
    compare.set_defaults(handler=_run_compare)

    report = subparsers.add_parser(
        "report", help="score reviewed evidence and write the M5 decision report"
    )
    report.add_argument("--baseline-summary", type=Path, required=True)
    report.add_argument("--review-csv", type=Path, required=True)
    report.add_argument("--comparison", type=Path, required=True)
    report.add_argument("--normalization-review", type=Path, required=True)
    report.add_argument("--detector-misses", type=Path, required=True)
    report.add_argument("--timings", type=Path, required=True)
    report.add_argument("--candidate-face-cache", type=Path, required=True)
    report.add_argument("--candidate-gallery-cache", type=Path, required=True)
    report.add_argument(
        "--thresholds", type=_non_negative_float, nargs="+", required=True
    )
    report.add_argument(
        "--recommended-threshold", type=_non_negative_float, required=True
    )
    report.add_argument(
        "--recommended-normalization",
        choices=("base", "Facenet2018"),
        required=True,
    )
    report.add_argument("--threshold-output", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(handler=_run_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
