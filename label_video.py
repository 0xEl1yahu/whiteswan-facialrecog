"""CPU-only Harry Potter face labelling pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Literal, Sequence

import cv2
import numpy as np
from deepface.modules import verification


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
DEFAULT_PIN_STRATEGY = "all"  # D1 owner decision, 2026-09-26.
DEFAULT_NORMALIZATION = "base"
DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_CSV_PATH = Path("output/matches.csv")
DEFAULT_IOU_MIN = 0.3
DEFAULT_TRACK_TTL = 15
DEFAULT_EXPAND_PERCENTAGE = 0  # DeepFace default; part of cache identity.
_MODEL_BUILT = False
_MODEL_LOAD_SECONDS_TOTAL = 0.0
PROGRESS_EVERY_FRAMES = 100  # M3 reporting interval; does not affect output.

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
    global _MODEL_BUILT, _MODEL_LOAD_SECONDS_TOTAL
    if _MODEL_BUILT:
        return
    started = time.perf_counter()
    _get_deepface().build_model(cfg.model_name)
    _MODEL_LOAD_SECONDS_TOTAL += time.perf_counter() - started
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


_GALLERY_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
_GALLERY_CACHE_SCHEMA_VERSION = 1
_FACE_CACHE_SCHEMA_VERSION = 1
_VERSION_DISTRIBUTIONS = {
    "deepface": "deepface",
    "retinaface": "retina-face",
    "tensorflow": "tensorflow",
    "opencv": "opencv-python",
}


def _gallery_paths(ref_dir: Path) -> dict[str, list[Path]]:
    if not ref_dir.is_dir():
        raise ValueError(f"reference directory does not exist: {ref_dir}")

    missing = [name for name in CHARACTER_NAMES if not (ref_dir / name).is_dir()]
    if missing:
        raise ValueError(
            "missing required character folder(s): " + ", ".join(missing)
        )

    return {
        owner: sorted(
            (
                path
                for path in (ref_dir / owner).iterdir()
                if path.is_file() and path.suffix.lower() in _GALLERY_EXTENSIONS
            ),
            key=lambda path: path.name,
        )
        for owner in CHARACTER_NAMES
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_versions() -> dict[str, str]:
    return {
        name: importlib_metadata.version(distribution)
        for name, distribution in _VERSION_DISTRIBUTIONS.items()
    }


def _gallery_cache_metadata(cfg: Config, versions: dict[str, str]) -> dict:
    return {
        "schema_version": _GALLERY_CACHE_SCHEMA_VERSION,
        "model_name": cfg.model_name,
        "detector_backend": cfg.detector_backend,
        "normalization": cfg.normalization,
        "align": cfg.align,
        "max_faces": 1,
        "expand_percentage": cfg.expand_percentage,
        "l2_normalize": True,
        "versions": dict(sorted(versions.items())),
    }


def gallery_cache_key(cfg: Config, versions: dict[str, str]) -> str:
    """Hash only inputs that change gallery embeddings, never photo state."""
    encoded = json.dumps(
        _gallery_cache_metadata(cfg, versions),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _face_cache_metadata(
    video_path: Path, cfg: Config, versions: dict[str, str]
) -> dict:
    return {
        "schema_version": _FACE_CACHE_SCHEMA_VERSION,
        "video_sha256": _file_sha256(video_path),
        "model_name": cfg.model_name,
        "detector_backend": cfg.detector_backend,
        "normalization": cfg.normalization,
        "align": cfg.align,
        "max_faces": cfg.max_faces,
        "expand_percentage": cfg.expand_percentage,
        "l2_normalize": True,
        "versions": dict(sorted(versions.items())),
    }


def face_cache_key(
    video_path: Path, cfg: Config, versions: dict[str, str]
) -> str:
    """Hash every upstream perception input and no downstream choice."""
    encoded = json.dumps(
        _face_cache_metadata(video_path, cfg, versions),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_embedding(value: object) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32)
    if embedding.shape != (512,):
        raise ValueError(f"embedding must have shape (512,), got {embedding.shape}")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError(f"embedding must be L2-normalised, got norm {norm}")
    return embedding


def _embed_gallery_photo(path: Path, cfg: Config) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("image is unreadable")

    build_models(cfg)
    results = _get_deepface().represent(
        img_path=image,
        model_name=cfg.model_name,
        detector_backend=cfg.detector_backend,
        enforce_detection=True,
        align=cfg.align,
        normalization=cfg.normalization,
        max_faces=1,
        l2_normalize=True,
        expand_percentage=cfg.expand_percentage,
    )
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise ValueError("DeepFace returned no usable face")
    return _validated_embedding(results[0].get("embedding"))


def build_pins(
    records: Sequence[GalleryPhoto], strategy: Literal["mean", "all"]
) -> Gallery:
    """Build deterministic character pins from immutable per-photo embeddings."""
    if strategy not in {"mean", "all"}:
        raise ValueError(f"unsupported pin strategy: {strategy}")
    if not records:
        raise ValueError("cannot build gallery pins without photos")

    ordered = sorted(records, key=lambda record: (record.owner, record.source_path))
    for record in ordered:
        _validated_embedding(record.embedding)
    names = sorted({record.owner for record in ordered})

    if strategy == "all":
        pins = np.stack([record.embedding for record in ordered]).astype(
            np.float32, copy=True
        )
        owners = [record.owner for record in ordered]
    else:
        mean_pins: list[np.ndarray] = []
        for owner in names:
            owner_embeddings = np.stack(
                [record.embedding for record in ordered if record.owner == owner]
            ).astype(np.float32)
            mean = owner_embeddings.mean(axis=0, dtype=np.float32)
            norm = float(np.linalg.norm(mean))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(f"cannot normalise mean pin for {owner}")
            mean_pins.append((mean / norm).astype(np.float32))
        pins = np.stack(mean_pins).astype(np.float32, copy=False)
        owners = names.copy()

    return Gallery(
        pins=pins,
        pin_owner=owners,
        names=names,
        meta={
            "strategy": strategy,
            "photo_count": len(ordered),
            "source_paths": [record.source_path for record in ordered],
            "file_hashes": [record.file_hash for record in ordered],
        },
    )


def _load_gallery_cache(
    cache_path: Path, expected_metadata: dict
) -> dict[str, GalleryPhoto]:
    if not cache_path.is_file():
        return {}
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            required = {
                "meta_json",
                "owners",
                "source_paths",
                "file_hashes",
                "embeddings",
            }
            if set(cached.files) != required:
                return {}
            metadata = json.loads(str(cached["meta_json"].item()))
            if metadata != expected_metadata:
                return {}
            owners = cached["owners"].astype(str).tolist()
            source_paths = cached["source_paths"].astype(str).tolist()
            file_hashes = cached["file_hashes"].astype(str).tolist()
            embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {}

    count = len(source_paths)
    if (
        len(owners) != count
        or len(file_hashes) != count
        or embeddings.shape != (count, 512)
        or len(set(source_paths)) != count
    ):
        return {}

    records: dict[str, GalleryPhoto] = {}
    try:
        for index, source_path in enumerate(source_paths):
            records[source_path] = GalleryPhoto(
                owner=owners[index],
                source_path=source_path,
                file_hash=file_hashes[index],
                embedding=_validated_embedding(embeddings[index]),
            )
    except (ValueError, IndexError):
        return {}
    return records


def _write_gallery_cache(
    cache_path: Path, metadata: dict, records: Sequence[GalleryPhoto]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: (record.owner, record.source_path))
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                meta_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                owners=np.asarray([record.owner for record in ordered], dtype=np.str_),
                source_paths=np.asarray(
                    [record.source_path for record in ordered], dtype=np.str_
                ),
                file_hashes=np.asarray(
                    [record.file_hash for record in ordered], dtype=np.str_
                ),
                embeddings=np.stack(
                    [record.embedding for record in ordered]
                ).astype(np.float32),
            )
        temporary_path.replace(cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class FaceCache:
    """Persistent per-frame perception results with explicit retryable states."""

    def __init__(
        self,
        video_path: Path,
        cfg: Config,
        versions: dict[str, str] | None = None,
    ) -> None:
        self._enabled = cfg.use_cache
        self._metadata = _face_cache_metadata(
            video_path, cfg, versions if versions is not None else _installed_versions()
        )
        key = hashlib.sha256(
            json.dumps(
                self._metadata, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        safe_stem = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in video_path.stem
        ) or "video"
        self.path = cfg.cache_dir / f"faces_{safe_stem}_{key}.npz"
        self._frames: dict[int, tuple[Literal["ok", "failed"], list[Face]]] = {}
        if self._enabled and self.path.is_file():
            self._load()

    def status(self, frame_idx: int) -> Literal["absent", "ok", "failed"]:
        entry = self._frames.get(frame_idx)
        return "absent" if entry is None else entry[0]

    def get(self, frame_idx: int) -> list[Face] | None:
        entry = self._frames.get(frame_idx)
        if entry is None or entry[0] != "ok":
            return None
        return list(entry[1])

    def put_ok(self, frame_idx: int, faces: Sequence[Face]) -> None:
        if frame_idx < 0:
            raise ValueError("frame index must be non-negative")
        validated: list[Face] = []
        for face in faces:
            validated.append(
                Face(
                    box=tuple(int(value) for value in face.box),
                    landmarks={
                        str(name): (int(point[0]), int(point[1]))
                        for name, point in face.landmarks.items()
                    },
                    embedding=_validated_embedding(face.embedding).copy(),
                    det_conf=float(face.det_conf),
                )
            )
        self._frames[frame_idx] = ("ok", validated)

    def put_failed(self, frame_idx: int) -> None:
        if frame_idx < 0:
            raise ValueError("frame index must be non-negative")
        self._frames[frame_idx] = ("failed", [])

    def missing(self, frame_indices: Sequence[int]) -> list[int]:
        return [index for index in frame_indices if self.status(index) != "ok"]

    def flush(self) -> None:
        if not self._enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame_indices = sorted(self._frames)
        statuses: list[str] = []
        offsets = [0]
        faces: list[Face] = []
        for frame_idx in frame_indices:
            status, frame_faces = self._frames[frame_idx]
            statuses.append(status)
            if status == "ok":
                faces.extend(frame_faces)
            offsets.append(len(faces))

        boxes = np.asarray([face.box for face in faces], dtype=np.int32).reshape(-1, 4)
        confidences = np.asarray(
            [face.det_conf for face in faces], dtype=np.float32
        )
        embeddings = (
            np.stack([face.embedding for face in faces]).astype(np.float32)
            if faces
            else np.empty((0, 512), dtype=np.float32)
        )
        landmarks_json = np.asarray(
            [json.dumps(face.landmarks, sort_keys=True) for face in faces],
            dtype=np.str_,
        )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    meta_json=np.asarray(json.dumps(self._metadata, sort_keys=True)),
                    frame_indices=np.asarray(frame_indices, dtype=np.int64),
                    statuses=np.asarray(statuses, dtype=np.str_),
                    face_offsets=np.asarray(offsets, dtype=np.int64),
                    boxes=boxes,
                    landmarks_json=landmarks_json,
                    det_confidences=confidences,
                    embeddings=embeddings,
                )
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _load(self) -> None:
        try:
            with np.load(self.path, allow_pickle=False) as stored:
                required = {
                    "meta_json",
                    "frame_indices",
                    "statuses",
                    "face_offsets",
                    "boxes",
                    "landmarks_json",
                    "det_confidences",
                    "embeddings",
                }
                if set(stored.files) != required:
                    raise ValueError("cache fields do not match schema")
                metadata = json.loads(str(stored["meta_json"].item()))
                if metadata != self._metadata:
                    raise ValueError("cache metadata does not match configuration")
                frame_indices = np.asarray(stored["frame_indices"], dtype=np.int64)
                statuses = stored["statuses"].astype(str).tolist()
                offsets = np.asarray(stored["face_offsets"], dtype=np.int64)
                boxes = np.asarray(stored["boxes"], dtype=np.int32)
                landmarks_json = stored["landmarks_json"].astype(str).tolist()
                confidences = np.asarray(
                    stored["det_confidences"], dtype=np.float32
                )
                embeddings = np.asarray(stored["embeddings"], dtype=np.float32)
            self._frames = self._decode_frames(
                frame_indices,
                statuses,
                offsets,
                boxes,
                landmarks_json,
                confidences,
                embeddings,
            )
        except Exception as error:
            self._frames = {}
            print(f"WARNING: ignored face cache {self.path}: {error}")

    @staticmethod
    def _decode_frames(
        frame_indices: np.ndarray,
        statuses: list[str],
        offsets: np.ndarray,
        boxes: np.ndarray,
        landmarks_json: list[str],
        confidences: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict[int, tuple[Literal["ok", "failed"], list[Face]]]:
        frame_count = len(frame_indices)
        face_count = len(boxes)
        if (
            len(statuses) != frame_count
            or offsets.shape != (frame_count + 1,)
            or len(set(int(index) for index in frame_indices)) != frame_count
            or any(int(index) < 0 for index in frame_indices)
            or offsets[0] != 0
            or offsets[-1] != face_count
            or np.any(np.diff(offsets) < 0)
            or boxes.shape != (face_count, 4)
            or len(landmarks_json) != face_count
            or confidences.shape != (face_count,)
            or embeddings.shape != (face_count, 512)
            or any(status not in {"ok", "failed"} for status in statuses)
        ):
            raise ValueError("cache array shapes or values are invalid")

        decoded: dict[int, tuple[Literal["ok", "failed"], list[Face]]] = {}
        for position, raw_index in enumerate(frame_indices):
            frame_idx = int(raw_index)
            status = statuses[position]
            start = int(offsets[position])
            stop = int(offsets[position + 1])
            if status == "failed" and start != stop:
                raise ValueError("failed cache frame contains faces")
            frame_faces: list[Face] = []
            for face_index in range(start, stop):
                raw_landmarks = json.loads(landmarks_json[face_index])
                if not isinstance(raw_landmarks, dict):
                    raise ValueError("cached landmarks are invalid")
                landmarks = {
                    str(name): (int(point[0]), int(point[1]))
                    for name, point in raw_landmarks.items()
                }
                frame_faces.append(
                    Face(
                        box=tuple(int(value) for value in boxes[face_index]),
                        landmarks=landmarks,
                        embedding=_validated_embedding(embeddings[face_index]),
                        det_conf=float(confidences[face_index]),
                    )
                )
            decoded[frame_idx] = (status, frame_faces)  # type: ignore[assignment]
        return decoded


def selected_frame_indices(start: int, stop: int, stride: int) -> list[int]:
    """Return absolute selected indices, with stride relative to the window start."""
    if start < 0:
        raise ValueError("start frame must be non-negative")
    if stop < start:
        raise ValueError("stop frame must not precede start frame")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    return list(range(start, stop, stride))


def process_batch_with_fallback(
    indexed_frames: Sequence[tuple[int, np.ndarray]], cfg: Config
) -> dict[int, tuple[Literal["ok", "failed"], list[Face]]]:
    """Embed a batch, retrying frames separately if the batch raises."""
    if not indexed_frames:
        return {}
    frame_indices = [frame_idx for frame_idx, _ in indexed_frames]
    frames = [frame for _, frame in indexed_frames]
    try:
        faces_by_frame = embed_faces(frames, cfg)
        if len(faces_by_frame) != len(indexed_frames):
            raise ValueError("perception result count does not match frame batch")
        return {
            frame_idx: ("ok", faces)
            for frame_idx, faces in zip(
                frame_indices, faces_by_frame, strict=True
            )
        }
    except Exception as batch_error:
        print(f"WARNING: batch perception failed; retrying frames: {batch_error}")

    recovered: dict[int, tuple[Literal["ok", "failed"], list[Face]]] = {}
    for frame_idx, frame in indexed_frames:
        try:
            per_frame = embed_faces([frame], cfg)
            if len(per_frame) != 1:
                raise ValueError("perception result count does not match one frame")
            recovered[frame_idx] = ("ok", per_frame[0])
        except Exception as frame_error:
            print(f"WARNING: frame {frame_idx} perception failed: {frame_error}")
            recovered[frame_idx] = ("failed", [])
    return recovered


def _unknown_matches(faces: Sequence[Face]) -> list[Match]:
    return [
        Match(
            nearest_name="Unknown",
            name=None,
            distance=float("inf"),
            confidence=0.0,
        )
        for _ in faces
    ]


def cosine_distances(v: np.ndarray, pins: np.ndarray) -> np.ndarray:
    """Return cosine distances for L2-normalised embeddings and gallery pins."""
    return 1 - pins @ v


def match(face: Face, gallery: Gallery, threshold: float) -> Match:
    """Assign the nearest gallery owner only below the strict threshold."""
    distances = cosine_distances(face.embedding, gallery.pins)
    nearest_index = int(np.argmin(distances))
    distance = float(distances[nearest_index])
    nearest_name = gallery.pin_owner[nearest_index]
    verified = distance < threshold
    return Match(
        nearest_name=nearest_name,
        name=nearest_name if verified else None,
        distance=distance,
        confidence=float(
            verification.find_confidence(distance, MODEL_NAME, verified, "cosine")
        ),
    )


_MATCH_COLUMNS = (
    "frame_idx", "face_idx", "x", "y", "w", "h", "det_conf",
    "nearest_name", "distance", "threshold", "assigned_name", "confidence",
)


def _validate_csv_path(cfg: Config) -> None:
    """Keep evidence output away from input media and embedding stores."""
    csv_path = cfg.csv_path.resolve()
    if csv_path == cfg.input_path.resolve():
        raise ValueError("CSV path must differ from the input video")
    if csv_path == cfg.output_path.resolve():
        raise ValueError("CSV path must differ from the output video")
    if csv_path.is_relative_to(cfg.ref_dir.resolve()):
        raise ValueError("CSV path must be outside the reference images directory")
    if csv_path.is_relative_to(cfg.cache_dir.resolve()):
        raise ValueError("CSV path must be outside the cache directory")


class MatchLogger:
    """Write one evidence row per face and optional original-frame crops."""

    def __init__(self, cfg: Config) -> None:
        _validate_csv_path(cfg)
        self._cfg = cfg
        cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = tempfile.NamedTemporaryFile(
            mode="w",
            dir=cfg.csv_path.parent,
            prefix=f".{cfg.csv_path.name}.",
            suffix=".tmp",
            newline="",
            encoding="utf-8",
            delete=False,
        )
        self._staged_path = Path(self._file.name)
        self._writer = csv.writer(self._file)
        self._writer.writerow(_MATCH_COLUMNS)

    @property
    def closed(self) -> bool:
        return self._file.closed

    def log(
        self,
        frame_idx: int,
        face_idx: int,
        face: Face,
        match: Match,
        threshold: float,
        frame: np.ndarray | None = None,
    ) -> None:
        self._writer.writerow(
            (
                frame_idx, face_idx, *face.box, face.det_conf,
                match.nearest_name, match.distance, threshold,
                match.name if match.name is not None else "Unknown",
                match.confidence,
            )
        )
        if self._cfg.debug_crops:
            if frame is None:
                raise ValueError("debug crops require the original frame")
            x, y, width, height = face.box
            frame_height, frame_width = frame.shape[:2]
            left = min(max(x, 0), frame_width)
            top = min(max(y, 0), frame_height)
            right = min(max(x + width, 0), frame_width)
            bottom = min(max(y + height, 0), frame_height)
            if right > left and bottom > top:
                crop_dir = self._cfg.csv_path.parent / "debug" / "crops"
                crop_dir.mkdir(parents=True, exist_ok=True)
                safe_name = "".join(
                    character if character.isalnum() or character in {".", "_", "-"} else "_"
                    for character in match.nearest_name
                )
                filename = f"{frame_idx:06d}_{face_idx:02d}_{safe_name}_{match.distance}.png"
                if not cv2.imwrite(str(crop_dir / filename), frame[top:bottom, left:right]):
                    raise OSError(f"cannot write debug crop: {crop_dir / filename}")

    def close(self, publish: bool = True) -> None:
        if not self._file.closed:
            self._file.close()
        if self._staged_path is None:
            return
        staged_path = self._staged_path
        self._staged_path = None
        try:
            if publish:
                staged_path.replace(self._cfg.csv_path)
        finally:
            staged_path.unlink(missing_ok=True)

    def __enter__(self) -> MatchLogger:
        return self

    def __exit__(self, exc_type: object, *_exc: object) -> None:
        self.close(publish=exc_type is None)


def process_video(cfg: Config, gallery: Gallery | None = None) -> RunSummary:
    """Write the requested video window with cached face annotations."""
    if not cfg.input_path.is_file():
        raise ValueError(f"cannot open input video: {cfg.input_path}")
    if cfg.input_path.resolve() == cfg.output_path.resolve():
        raise ValueError("input and output video paths must differ")

    run_started = time.perf_counter()
    model_seconds_before = _MODEL_LOAD_SECONDS_TOTAL
    capture = cv2.VideoCapture(str(cfg.input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open input video: {cfg.input_path}")

    writer: Any | None = None
    logger: MatchLogger | None = None
    temporary_output: Path | None = None
    succeeded = False
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"input video has invalid FPS: {fps}")
        if width <= 0 or height <= 0:
            raise ValueError(f"input video has invalid size: {width}x{height}")
        if frame_count <= 0:
            raise ValueError(f"input video has invalid frame count: {frame_count}")
        if cfg.start_frame >= frame_count:
            raise ValueError(
                f"start frame {cfg.start_frame} is outside {frame_count}-frame input"
            )

        stop_frame = frame_count
        if cfg.max_frames is not None:
            stop_frame = min(frame_count, cfg.start_frame + cfg.max_frames)
        selected = selected_frame_indices(cfg.start_frame, stop_frame, cfg.stride)
        selected_set = set(selected)

        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = cfg.output_path.suffix or ".mp4"
        with tempfile.NamedTemporaryFile(
            dir=cfg.output_path.parent,
            prefix=f".{cfg.output_path.stem}.",
            suffix=suffix,
            delete=False,
        ) as temporary:
            temporary_output = Path(temporary.name)
        writer = cv2.VideoWriter(
            str(temporary_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"cannot open output video: {cfg.output_path}")

        if gallery is not None:
            logger = MatchLogger(cfg)

        cache = FaceCache(cfg.input_path, cfg)
        initial_status = {frame_idx: cache.status(frame_idx) for frame_idx in selected}
        cache_hits = sum(status == "ok" for status in initial_status.values())
        cache_misses = len(selected) - cache_hits
        perception_seconds = 0.0
        written_frames = 0
        last_faces: list[Face] = []
        last_matches: list[Match] = []
        label_counts: Counter[str] = Counter()
        pending_frames: list[tuple[int, np.ndarray]] = []
        pending_missing: list[tuple[int, np.ndarray]] = []
        max_pending_frames = max(1, cfg.batch_size * cfg.stride)

        def flush_pending() -> None:
            nonlocal perception_seconds, written_frames, last_faces, last_matches
            if not pending_frames:
                return
            if pending_missing:
                perception_started = time.perf_counter()
                batch_results = process_batch_with_fallback(pending_missing, cfg)
                perception_seconds += time.perf_counter() - perception_started
                for frame_idx, (status, faces) in batch_results.items():
                    if status == "ok":
                        cache.put_ok(frame_idx, faces)
                    else:
                        cache.put_failed(frame_idx)
                cache.flush()

            for frame_idx, frame in pending_frames:
                if frame_idx in selected_set:
                    cached_faces = cache.get(frame_idx)
                    last_faces = [] if cached_faces is None else cached_faces
                    last_matches = (
                        [match(face, gallery, cfg.threshold) for face in last_faces]
                        if gallery is not None
                        else _unknown_matches(last_faces)
                    )
                rendered = draw(frame, last_faces, last_matches, cfg)
                for face_idx, (face, assignment) in enumerate(
                    zip(last_faces, last_matches, strict=True)
                ):
                    label_counts[assignment.name or "Unknown"] += 1
                    if logger is not None:
                        logger.log(
                            frame_idx, face_idx, face, assignment, cfg.threshold,
                            frame=frame,
                        )
                writer.write(rendered)
                written_frames += 1
                if written_frames % PROGRESS_EVERY_FRAMES == 0:
                    elapsed = max(time.perf_counter() - run_started, 1e-9)
                    print(
                        f"frames_written={written_frames} "
                        f"faces={sum(len(cache.get(index) or []) for index in selected)} "
                        f"fps={written_frames / elapsed:.2f}"
                    )
            pending_frames.clear()
            pending_missing.clear()

        if cfg.start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)
        for frame_idx in range(cfg.start_frame, stop_frame):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read input frame {frame_idx}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"input frame {frame_idx} has unexpected size "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )
            pending_frames.append((frame_idx, frame))
            if frame_idx in selected_set and cache.status(frame_idx) != "ok":
                pending_missing.append((frame_idx, frame))
            if (
                len(pending_missing) >= cfg.batch_size
                or len(pending_frames) >= max_pending_frames
            ):
                flush_pending()
        flush_pending()

        expected_written = stop_frame - cfg.start_frame
        if written_frames != expected_written:
            raise RuntimeError(
                f"wrote {written_frames} frames; expected {expected_written}"
            )
        writer.release()
        writer = None
        capture.release()
        faces = sum(label_counts.values())
        failures = sum(cache.status(index) == "failed" for index in selected)
        temporary_output.replace(cfg.output_path)
        temporary_output = None
        if logger is not None:
            logger.close()
            logger = None
        succeeded = True
        elapsed_seconds = time.perf_counter() - run_started
        model_seconds = _MODEL_LOAD_SECONDS_TOTAL - model_seconds_before
        return RunSummary(
            elapsed_seconds=elapsed_seconds,
            model_seconds=model_seconds,
            perception_seconds=perception_seconds,
            processed_frames=len(selected),
            written_frames=written_frames,
            faces=faces,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            failures=failures,
            processing_fps=(
                cache_misses / perception_seconds if perception_seconds > 0 else 0.0
            ),
            fps=fps,
            width=width,
            height=height,
            label_distribution=dict(sorted(label_counts.items())),
        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if logger is not None:
            logger.close(publish=False)
        if not succeeded and temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


def leave_one_out_report(
    records: Sequence[GalleryPhoto], threshold: float
) -> dict[str, dict]:
    """Evaluate each photo against pins built from every other photo."""
    if threshold < 0:
        raise ValueError("threshold must be zero or greater")
    ordered = sorted(records, key=lambda record: (record.owner, record.source_path))
    owner_counts = Counter(record.owner for record in ordered)
    insufficient = [owner for owner, count in owner_counts.items() if count < 2]
    if insufficient:
        raise ValueError(
            "leave-one-out requires at least two photos for: "
            + ", ".join(sorted(insufficient))
        )

    report: dict[str, dict] = {}
    for strategy in ("mean", "all"):
        photos: list[dict] = []
        aggregate = {"correct": 0, "unknown": 0, "wrong": 0}
        for held_out_index, held_out in enumerate(ordered):
            candidates = [
                record
                for index, record in enumerate(ordered)
                if index != held_out_index
            ]
            gallery = build_pins(candidates, strategy)
            distances = 1.0 - gallery.pins @ held_out.embedding
            nearest_index = int(np.argmin(distances))
            distance = max(0.0, float(distances[nearest_index]))
            nearest_name = gallery.pin_owner[nearest_index]
            assigned_name = nearest_name if distance < threshold else None
            if assigned_name is None:
                outcome = "unknown"
            elif assigned_name == held_out.owner:
                outcome = "correct"
            else:
                outcome = "wrong"
            aggregate[outcome] += 1
            photos.append(
                {
                    "owner": held_out.owner,
                    "source_path": held_out.source_path,
                    "nearest_name": nearest_name,
                    "distance": distance,
                    "assigned_name": assigned_name,
                    "outcome": outcome,
                    "candidate_source_paths": [
                        candidate.source_path for candidate in candidates
                    ],
                }
            )
        report[strategy] = {
            "photos": photos,
            "per_character": {
                owner: [photo for photo in photos if photo["owner"] == owner]
                for owner in sorted(owner_counts)
            },
            "aggregate": aggregate,
        }
    return report


def load_gallery(ref_dir: Path, cfg: Config) -> Gallery:
    """Validate and embed the owner-curated reference gallery."""
    paths_by_owner = _gallery_paths(ref_dir)
    cache_key: str | None = None
    cache_metadata: dict | None = None
    cached_records: dict[str, GalleryPhoto] = {}
    cache_path: Path | None = None
    if cfg.use_cache:
        cache_metadata = _gallery_cache_metadata(cfg, _installed_versions())
        cache_key = gallery_cache_key(cfg, cache_metadata["versions"])
        cache_path = cfg.cache_dir / f"gallery_{cache_key}.npz"
        cached_records = _load_gallery_cache(cache_path, cache_metadata)

    records: list[GalleryPhoto] = []

    for owner in CHARACTER_NAMES:
        paths = paths_by_owner[owner]
        owner_records: list[GalleryPhoto] = []
        for path in paths:
            relative_path = path.relative_to(ref_dir).as_posix()
            file_hash = _file_sha256(path)
            cached = cached_records.get(relative_path)
            if (
                cached is not None
                and cached.owner == owner
                and cached.file_hash == file_hash
            ):
                owner_records.append(cached)
                continue
            try:
                embedding = _embed_gallery_photo(path, cfg)
            except Exception as error:  # DeepFace exposes several detector exceptions.
                print(f"WARNING: skipped {relative_path}: {error}")
                continue
            owner_records.append(
                GalleryPhoto(
                    owner=owner,
                    source_path=relative_path,
                    file_hash=file_hash,
                    embedding=embedding,
                )
            )

        skipped = len(paths) - len(owner_records)
        print(
            f"{owner}: found={len(paths)} used={len(owner_records)} skipped={skipped}"
        )
        if len(owner_records) < 2:
            raise ValueError(
                f"{owner} has {len(owner_records)} valid reference image(s); minimum is 2"
            )
        records.extend(owner_records)

    if cache_path is not None and cache_metadata is not None:
        _write_gallery_cache(cache_path, cache_metadata, records)

    gallery = build_pins(records, cfg.pin_strategy)
    return Gallery(
        pins=gallery.pins,
        pin_owner=gallery.pin_owner,
        names=gallery.names,
        meta={
            **gallery.meta,
            "cache_key": cache_key,
            "embedding_config": cache_metadata,
        },
    )


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


def main(argv: Sequence[str] | None = None) -> int:
    cfg = load_config(parse_args(argv))
    run_started = time.perf_counter()
    model_seconds_before = _MODEL_LOAD_SECONDS_TOTAL
    try:
        if not cfg.input_path.is_file():
            raise ValueError(f"cannot open input video: {cfg.input_path}")
        if cfg.input_path.resolve() == cfg.output_path.resolve():
            raise ValueError("input and output video paths must differ")
        if not cfg.ref_dir.is_dir():
            raise ValueError(f"reference directory does not exist: {cfg.ref_dir}")
        _validate_csv_path(cfg)
        gallery = load_gallery(cfg.ref_dir, cfg)
        gallery_seconds = time.perf_counter() - run_started
        gallery_model_seconds = _MODEL_LOAD_SECONDS_TOTAL - model_seconds_before
        summary = process_video(cfg, gallery)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    elapsed_seconds = time.perf_counter() - run_started
    print(
        f"complete elapsed={elapsed_seconds:.3f}s "
        f"gallery={gallery_seconds:.3f}s "
        f"model={summary.model_seconds + gallery_model_seconds:.3f}s "
        f"gallery_model={gallery_model_seconds:.3f}s "
        f"perception={summary.perception_seconds:.3f}s "
        f"processing_fps={summary.processing_fps:.2f} "
        f"processed={summary.processed_frames} written={summary.written_frames} "
        f"faces={summary.faces} failures={summary.failures} "
        f"cache_hits={summary.cache_hits} cache_misses={summary.cache_misses} "
        f"labels={summary.label_distribution}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
