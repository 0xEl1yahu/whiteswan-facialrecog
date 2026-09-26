"""CPU-only Harry Potter face labelling pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
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

# Defaults are approved in docs/design/design-plan.md section 7.
DEFAULT_STRIDE = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.30  # DeepFace Facenet512/cosine default.
DEFAULT_PIN_STRATEGY = "mean"
DEFAULT_NORMALIZATION = "base"
DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_CSV_PATH = Path("output/matches.csv")
DEFAULT_IOU_MIN = 0.3
DEFAULT_TRACK_TTL = 15
DEFAULT_EXPAND_PERCENTAGE = 0  # DeepFace default; part of cache identity.
_MODEL_BUILT = False

# Fixed deterministic BGR presentation palette from the approved rendering design.
BOX_COLORS: dict[str | None, tuple[int, int, int]] = {
    "Harry Potter": (0, 0, 255),
    "Hermione Granger": (255, 0, 255),
    "Prof. McGonagall": (0, 180, 0),
    "Prof. Severus Snape": (255, 0, 0),
    "Ron Weasley": (0, 140, 255),
    None: (128, 128, 128),
}
LANDMARK_COLOR = (0, 255, 255)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_LABEL_PADDING = 3
_LANDMARK_RADIUS = 2


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


@dataclass
class Track:
    id: int
    box: tuple[int, int, int, int]
    last_seen: int
    votes: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    ref_dir: Path
    stride: int = DEFAULT_STRIDE
    batch_size: int = DEFAULT_BATCH_SIZE
    threshold: float = DEFAULT_THRESHOLD
    pin_strategy: Literal["mean", "all"] = DEFAULT_PIN_STRATEGY
    normalization: Literal["base", "Facenet2018"] = DEFAULT_NORMALIZATION
    max_faces: int | None = None  # No cap: required by R1.
    start_frame: int = 0
    max_frames: int | None = None
    cache_dir: Path = DEFAULT_CACHE_DIR
    use_cache: bool = True
    smooth: bool = False
    iou_min: float = DEFAULT_IOU_MIN
    track_ttl: int = DEFAULT_TRACK_TTL
    csv_path: Path = DEFAULT_CSV_PATH
    debug_crops: bool = False
    debug_landmarks: bool = False
    model_name: str = MODEL_NAME
    detector_backend: str = DETECTOR_BACKEND
    align: bool = True
    expand_percentage: int = DEFAULT_EXPAND_PERCENTAGE


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--stride", type=_positive_int, default=DEFAULT_STRIDE)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=_non_negative_float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--pin-strategy", choices=("mean", "all"), default=DEFAULT_PIN_STRATEGY
    )
    parser.add_argument(
        "--normalization", choices=("base", "Facenet2018"), default=DEFAULT_NORMALIZATION
    )
    parser.add_argument("--max-faces", type=_positive_int)
    parser.add_argument("--start-frame", type=_non_negative_int, default=0)
    parser.add_argument("--max-frames", type=_positive_int)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_false", dest="use_cache")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--iou-min", type=_unit_float, default=DEFAULT_IOU_MIN)
    parser.add_argument("--track-ttl", type=_positive_int, default=DEFAULT_TRACK_TTL)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--debug-crops", action="store_true")
    parser.add_argument("--debug-landmarks", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_config(args: argparse.Namespace) -> Config:
    return Config(**vars(args))


def _get_deepface() -> Any:
    from deepface import DeepFace

    return DeepFace


def build_models(cfg: Config) -> None:
    """Load Facenet512 once, only when an uncached embedding path requests it."""
    global _MODEL_BUILT
    if _MODEL_BUILT:
        return
    _get_deepface().build_model(cfg.model_name)
    _MODEL_BUILT = True


def _per_frame_results(raw_results: list, frame_count: int) -> list[list[dict]]:
    if frame_count == 1:
        if not raw_results:
            return [[]]
        if isinstance(raw_results[0], dict):
            return [raw_results]
        if len(raw_results) == 1 and isinstance(raw_results[0], list):
            return raw_results
        raise ValueError("unexpected DeepFace result shape for one frame")

    if len(raw_results) != frame_count or any(
        not isinstance(frame_results, list) for frame_results in raw_results
    ):
        raise ValueError("unexpected DeepFace result shape for frame batch")
    return raw_results


def _clip_box(area: dict, frame: np.ndarray) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame.shape[:2]
    x = int(area["x"])
    y = int(area["y"])
    right = x + int(area["w"])
    bottom = y + int(area["h"])
    clipped_x = min(max(x, 0), frame_width)
    clipped_y = min(max(y, 0), frame_height)
    clipped_right = min(max(right, 0), frame_width)
    clipped_bottom = min(max(bottom, 0), frame_height)
    return (
        clipped_x,
        clipped_y,
        max(0, clipped_right - clipped_x),
        max(0, clipped_bottom - clipped_y),
    )


def _face_from_result(result: dict, frame: np.ndarray) -> Face | None:
    confidence = float(result["face_confidence"])
    if confidence == 0:
        return None
    if not 0 < confidence <= 1:
        raise ValueError(f"invalid face confidence: {confidence}")

    embedding = np.asarray(result["embedding"], dtype=np.float32)
    if embedding.shape != (512,):
        raise ValueError(f"embedding must have shape (512,), got {embedding.shape}")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError(f"embedding must be L2-normalised, got norm {norm}")

    area = result["facial_area"]
    landmarks = {
        key: (int(value[0]), int(value[1]))
        for key, value in area.items()
        if key not in {"x", "y", "w", "h"} and value is not None
    }
    return Face(
        box=_clip_box(area, frame),
        landmarks=landmarks,
        embedding=embedding,
        det_conf=confidence,
    )


def embed_faces(frames: list[np.ndarray], cfg: Config) -> list[list[Face]]:
    """Detect, align, and embed every face in each supplied BGR frame."""
    if not frames:
        raise ValueError("embed_faces requires at least one frame")

    build_models(cfg)
    raw_results = _get_deepface().represent(
        img_path=frames,
        model_name=cfg.model_name,
        detector_backend=cfg.detector_backend,
        enforce_detection=False,
        align=cfg.align,
        normalization=cfg.normalization,
        max_faces=cfg.max_faces,
        l2_normalize=True,
        expand_percentage=cfg.expand_percentage,
    )
    per_frame = _per_frame_results(raw_results, len(frames))

    output: list[list[Face]] = []
    for frame, frame_results in zip(frames, per_frame, strict=True):
        faces = [
            face
            for result in frame_results
            if (face := _face_from_result(result, frame)) is not None
        ]
        output.append(faces)
    return output

def _label_position(
    frame: np.ndarray, box: tuple[int, int, int, int], label: str
) -> tuple[int, int, int, int, tuple[int, int]]:
    frame_height, frame_width = frame.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(
        label, _FONT, _FONT_SCALE, _FONT_THICKNESS
    )
    background_width = min(frame_width, text_width + 2 * _LABEL_PADDING)
    background_height = min(
        frame_height, text_height + baseline + 2 * _LABEL_PADDING
    )
    x, y, _, height = box
    left = min(max(x, 0), max(0, frame_width - background_width))
    above = y - background_height
    top = above if above >= 0 else min(y + height, frame_height - background_height)
    top = max(0, top)
    text_origin = (
        left + _LABEL_PADDING,
        min(frame_height - baseline - _LABEL_PADDING, top + _LABEL_PADDING + text_height),
    )
    return (
        left,
        top,
        left + background_width,
        top + background_height,
        text_origin,
    )


def draw(
    frame: np.ndarray,
    faces: Sequence[Face],
    matches: Sequence[Match],
    cfg: Config,
) -> np.ndarray:
    """Return an annotated copy of a BGR frame without performing I/O."""
    if len(faces) != len(matches):
        raise ValueError("faces and matches must have the same length")

    rendered = frame.copy()
    frame_height, frame_width = rendered.shape[:2]
    for face, match in zip(faces, matches, strict=True):
        x, y, width, height = face.box
        if width <= 0 or height <= 0:
            continue
        color = BOX_COLORS[match.name]
        right = min(frame_width - 1, x + width - 1)
        bottom = min(frame_height - 1, y + height - 1)
        cv2.rectangle(rendered, (x, y), (right, bottom), color, _BOX_THICKNESS)

        label = (
            f"{match.name} {round(match.confidence):d}%"
            if match.name is not None
            else "Unknown"
        )
        left, top, label_right, label_bottom, text_origin = _label_position(
            rendered, face.box, label
        )
        cv2.rectangle(
            rendered, (left, top), (label_right, label_bottom), color, cv2.FILLED
        )
        cv2.putText(
            rendered,
            label,
            text_origin,
            _FONT,
            _FONT_SCALE,
            (255, 255, 255),
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

        if cfg.debug_landmarks:
            for landmark_x, landmark_y in face.landmarks.values():
                if 0 <= landmark_x < frame_width and 0 <= landmark_y < frame_height:
                    cv2.circle(
                        rendered,
                        (landmark_x, landmark_y),
                        _LANDMARK_RADIUS,
                        LANDMARK_COLOR,
                        cv2.FILLED,
                    )

    return rendered
