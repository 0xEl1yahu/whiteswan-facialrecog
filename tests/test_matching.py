from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from deepface.modules import verification

from face_labeller import recognition
from face_labeller.config import Config
from face_labeller.contracts import Face, Gallery, Match
from face_labeller.evidence import MatchLogger
from face_labeller.recognition import cosine_distances, match


def unit_vector(axis: int) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[axis] = 1.0
    return vector


def test_cosine_distances_agrees_with_deepface() -> None:
    probe = unit_vector(0)
    tilted = (unit_vector(0) + unit_vector(1)) / np.sqrt(2)
    pins = np.stack([probe, tilted, unit_vector(1)]).astype(np.float32)

    distances = cosine_distances(probe, pins)

    assert distances.shape == (3,)
    for index, pin in enumerate(pins):
        assert distances[index] == pytest.approx(
            verification.find_distance(probe, pin, "cosine"), abs=1e-5
        )


def test_match_nearest_pin_wins_and_gallery_order_breaks_ties() -> None:
    probe = unit_vector(0)
    gallery = Gallery(
        pins=np.stack([unit_vector(1), probe, probe]),
        pin_owner=["Ron Weasley", "Harry Potter", "Hermione Granger"],
        names=["Harry Potter", "Hermione Granger", "Ron Weasley"],
        meta={},
    )
    face = Face((1, 2, 3, 4), {}, probe, 0.95)

    result = match(face, gallery, 0.30)

    assert result.nearest_name == "Harry Potter"
    assert result.name == "Harry Potter"
    assert result.distance == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("threshold", "expected_name", "verified"),
    [(1.01, "Ron Weasley", True), (1.0, None, False), (0.99, None, False)],
)
def test_match_threshold_is_strict_and_unknown_retains_nearest(
    monkeypatch: pytest.MonkeyPatch,
    threshold: float,
    expected_name: str | None,
    verified: bool,
) -> None:
    calls: list[tuple[float, str, bool, str]] = []

    def confidence(distance: float, model: str, is_verified: bool, metric: str) -> float:
        calls.append((distance, model, is_verified, metric))
        return 73.5

    monkeypatch.setattr(recognition.verification, "find_confidence", confidence)
    face = Face((1, 2, 3, 4), {}, unit_vector(0), 0.95)
    gallery = Gallery(np.stack([unit_vector(1)]), ["Ron Weasley"], ["Ron Weasley"], {})

    result = match(face, gallery, threshold)

    assert result.nearest_name == "Ron Weasley"
    assert result.name == expected_name
    assert result.distance == pytest.approx(1.0)
    assert result.confidence == 73.5
    assert calls == [(pytest.approx(1.0), "Facenet512", verified, "cosine")]


def logger_config(tmp_path: Path, *, debug_crops: bool = False) -> Config:
    return Config(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        ref_dir=tmp_path / "references",
        csv_path=tmp_path / "matches.csv",
        debug_crops=debug_crops,
    )


def test_match_logger_writes_one_row_per_face_and_overwrites_each_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = logger_config(tmp_path)
    face = Face((2, 3, 11, 13), {}, unit_vector(0), 0.95)
    known = Match("Harry Potter", "Harry Potter", 0.123456, 87.5)
    unknown = Match("Ron Weasley", None, 0.345678, 23.25)
    monkeypatch.setattr(
        recognition,
        "cosine_distances",
        lambda *args: (_ for _ in ()).throw(AssertionError("logger recomputed distance")),
    )

    with MatchLogger(cfg) as logger:
        logger.log(7, 0, face, known, 0.30)
        logger.log(7, 1, face, unknown, 0.30)
    assert logger.closed

    with cfg.csv_path.open(newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == [
            "frame_idx", "face_idx", "x", "y", "w", "h", "det_conf",
            "nearest_name", "distance", "threshold", "assigned_name", "confidence",
        ]
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0] == {
        "frame_idx": "7", "face_idx": "0", "x": "2", "y": "3", "w": "11", "h": "13",
        "det_conf": "0.95", "nearest_name": "Harry Potter",
        "distance": "0.123456", "threshold": "0.3",
        "assigned_name": "Harry Potter", "confidence": "87.5",
    }
    assert rows[1]["face_idx"] == "1"
    assert rows[1]["nearest_name"] == "Ron Weasley"
    assert rows[1]["assigned_name"] == "Unknown"
    assert float(rows[1]["distance"]) == pytest.approx(0.345678)

    with MatchLogger(cfg) as logger:
        logger.log(8, 0, face, unknown, 0.40)
    with cfg.csv_path.open(newline="") as source:
        reader = csv.DictReader(source)
        assert len(list(reader)) == 1
        assert reader.fieldnames == list(rows[0])


def test_match_logger_clips_debug_crop_without_mutating_frame(tmp_path: Path) -> None:
    cfg = logger_config(tmp_path, debug_crops=True)
    frame = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    before = frame.copy()
    face = Face((-2, 1, 6, 7), {}, unit_vector(0), 0.9)
    result = Match("Prof. McGonagall", None, 0.456789, 10.0)

    with MatchLogger(cfg) as logger:
        logger.log(12, 3, face, result, 0.30, frame=frame)

    crops = list((tmp_path / "debug" / "crops").glob("*.png"))
    assert len(crops) == 1
    assert "12" in crops[0].stem
    assert "03" in crops[0].stem
    assert "Prof._McGonagall" in crops[0].stem
    assert "0.456789" in crops[0].stem
    crop = cv2.imread(str(crops[0]))
    np.testing.assert_array_equal(crop, before[1:5, 0:4])
    np.testing.assert_array_equal(frame, before)


def test_match_logger_publishes_only_after_successful_close(tmp_path: Path) -> None:
    cfg = logger_config(tmp_path)
    cfg.csv_path.write_bytes(b"previous evidence\n")
    face = Face((2, 3, 11, 13), {}, unit_vector(0), 0.95)
    result = Match("Harry Potter", "Harry Potter", 0.1, 90.0)

    with MatchLogger(cfg) as logger:
        logger.log(0, 0, face, result, 0.3)
        assert cfg.csv_path.read_bytes() == b"previous evidence\n"

    with cfg.csv_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["assigned_name"] == "Harry Potter"
