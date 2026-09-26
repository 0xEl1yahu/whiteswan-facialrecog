"""Command-line parsing and immutable pipeline configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from face_labeller.contracts import DETECTOR_BACKEND, MODEL_NAME


# Defaults are approved in docs/design/design-plan.md section 7.
DEFAULT_STRIDE = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.30  # DeepFace Facenet512/cosine default.
DEFAULT_PIN_STRATEGY = "all"  # D1 owner decision, 2026-09-26.
DEFAULT_NORMALIZATION = "base"
DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_CSV_PATH = Path("output/matches.csv")
DEFAULT_IOU_MIN = 0.3
DEFAULT_TRACK_TTL = 15
DEFAULT_EXPAND_PERCENTAGE = 0  # DeepFace default; part of cache identity.


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
    parser = argparse.ArgumentParser(
        description="CPU-only Harry Potter face labelling pipeline."
    )
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
