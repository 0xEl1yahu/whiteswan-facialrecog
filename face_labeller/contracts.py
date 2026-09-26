"""Stable data contracts and fixed domain constants."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
CHARACTER_NAMES = (
    "Harry Potter",
    "Hermione Granger",
    "Prof. McGonagall",
    "Prof. Severus Snape",
    "Ron Weasley",
)


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]
    landmarks: dict[str, tuple[int, int]]
    embedding: np.ndarray
    det_conf: float


@dataclass(frozen=True)
class Match:
    nearest_name: str
    name: str | None
    distance: float
    confidence: float


@dataclass(frozen=True)
class Gallery:
    pins: np.ndarray
    pin_owner: list[str]
    names: list[str]
    meta: dict


@dataclass(frozen=True)
class GalleryPhoto:
    owner: str
    source_path: str
    file_hash: str
    embedding: np.ndarray


@dataclass
class Track:
    id: int
    box: tuple[int, int, int, int]
    last_seen: int
    votes: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class RunSummary:
    elapsed_seconds: float
    model_seconds: float
    perception_seconds: float
    processed_frames: int
    written_frames: int
    faces: int
    cache_hits: int
    cache_misses: int
    failures: int
    processing_fps: float
    fps: float
    width: int
    height: int
    label_distribution: dict[str, int]


@dataclass(frozen=True)
class VideoMetadata:
    input_path: Path
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class FramePlan:
    start_frame: int
    stop_frame: int
    written_frames: int
    selected_indices: tuple[int, ...]
