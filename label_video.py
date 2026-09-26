"""CPU-only Harry Potter face labelling pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import tempfile
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
DEFAULT_PIN_STRATEGY = "all"  # D1 owner decision, 2026-09-26.
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


_GALLERY_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
_GALLERY_CACHE_SCHEMA_VERSION = 1
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
