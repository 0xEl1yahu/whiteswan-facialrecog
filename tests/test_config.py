from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from label_video import (
    CHARACTER_NAMES,
    Config,
    Face,
    Gallery,
    Match,
    Track,
    load_config,
    parse_args,
)


REQUIRED_ARGS = [
    "--input",
    "input.mp4",
    "--output",
    "output.mp4",
    "--ref-dir",
    "references",
]


def test_cli_defaults_match_design_spec() -> None:
    cfg = load_config(parse_args(REQUIRED_ARGS))

    assert cfg.input_path == Path("input.mp4")
    assert cfg.output_path == Path("output.mp4")
    assert cfg.ref_dir == Path("references")
    assert cfg.stride == 1
    assert cfg.batch_size == 8
    assert cfg.threshold == pytest.approx(0.30)
    assert cfg.pin_strategy == "mean"
    assert cfg.normalization == "base"
    assert cfg.max_faces is None
    assert cfg.start_frame == 0
    assert cfg.max_frames is None
    assert cfg.cache_dir == Path("cache")
    assert cfg.use_cache is True
    assert cfg.smooth is False
    assert cfg.iou_min == pytest.approx(0.3)
    assert cfg.track_ttl == 15
    assert cfg.csv_path == Path("output/matches.csv")
    assert cfg.debug_crops is False
    assert cfg.debug_landmarks is False
    assert cfg.model_name == "Facenet512"
    assert cfg.detector_backend == "retinaface"
    assert cfg.align is True
    assert cfg.expand_percentage == 0


def test_cli_accepts_approved_overrides() -> None:
    cfg = load_config(
        parse_args(
            REQUIRED_ARGS
            + [
                "--stride",
                "3",
                "--batch-size",
                "4",
                "--threshold",
                "0.25",
                "--pin-strategy",
                "all",
                "--normalization",
                "Facenet2018",
                "--max-faces",
                "7",
                "--start-frame",
                "10",
                "--max-frames",
                "20",
                "--cache-dir",
                "custom-cache",
                "--no-cache",
                "--smooth",
                "--iou-min",
                "0.5",
                "--track-ttl",
                "9",
                "--csv",
                "custom.csv",
                "--debug-crops",
                "--debug-landmarks",
            ]
        )
    )

    assert cfg.stride == 3
    assert cfg.batch_size == 4
    assert cfg.threshold == pytest.approx(0.25)
    assert cfg.pin_strategy == "all"
    assert cfg.normalization == "Facenet2018"
    assert cfg.max_faces == 7
    assert cfg.start_frame == 10
    assert cfg.max_frames == 20
    assert cfg.cache_dir == Path("custom-cache")
    assert cfg.use_cache is False
    assert cfg.smooth is True
    assert cfg.iou_min == pytest.approx(0.5)
    assert cfg.track_ttl == 9
    assert cfg.csv_path == Path("custom.csv")
    assert cfg.debug_crops is True
    assert cfg.debug_landmarks is True


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--stride", "0"),
        ("--batch-size", "0"),
        ("--threshold", "-0.1"),
        ("--max-faces", "0"),
        ("--start-frame", "-1"),
        ("--max-frames", "0"),
        ("--iou-min", "1.1"),
        ("--track-ttl", "0"),
    ],
)
def test_cli_rejects_invalid_numeric_values(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(REQUIRED_ARGS + [option, value])
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("option", "value"),
    [("--pin-strategy", "median"), ("--normalization", "raw")],
)
def test_cli_rejects_unapproved_choices(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(REQUIRED_ARGS + [option, value])
    assert error.value.code == 2


def test_character_names_are_canonical_and_sorted() -> None:
    assert CHARACTER_NAMES == (
        "Harry Potter",
        "Hermione Granger",
        "Prof. McGonagall",
        "Prof. Severus Snape",
        "Ron Weasley",
    )


def test_approved_data_contracts() -> None:
    embedding = np.ones(512, dtype=np.float32)
    face = Face(
        box=(1, 2, 3, 4),
        landmarks={"nose": (2, 3)},
        embedding=embedding,
        det_conf=0.99,
    )
    match = Match(
        nearest_name="Harry Potter",
        name=None,
        distance=0.31,
        confidence=48.0,
    )
    gallery = Gallery(
        pins=embedding.reshape(1, -1),
        pin_owner=["Harry Potter"],
        names=["Harry Potter"],
        meta={},
    )
    track = Track(id=1, box=face.box, last_seen=0)

    assert match.nearest_name == "Harry Potter"
    assert match.name is None
    assert gallery.pins.shape == (1, 512)
    assert track.votes == {}
    with pytest.raises(FrozenInstanceError):
        face.det_conf = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        match.name = "Harry Potter"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        gallery.meta = {"changed": True}  # type: ignore[misc]


def test_config_is_frozen() -> None:
    cfg = load_config(parse_args(REQUIRED_ARGS))
    assert isinstance(cfg, Config)
    with pytest.raises(FrozenInstanceError):
        cfg.stride = 2  # type: ignore[misc]
