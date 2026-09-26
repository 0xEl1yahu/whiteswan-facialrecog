"""Owner-curated reference gallery discovery, caching, and pin construction."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
from typing import Literal, Sequence

import cv2
import numpy as np

from face_labeller import perception
from face_labeller.cache import (
    file_sha256,
    installed_versions,
    stable_json_hash,
    validated_embedding,
)
from face_labeller.config import Config
from face_labeller.contracts import CHARACTER_NAMES, Gallery, GalleryPhoto


GALLERY_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
GALLERY_CACHE_SCHEMA_VERSION = 1


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
                if path.is_file() and path.suffix.lower() in GALLERY_EXTENSIONS
            ),
            key=lambda path: path.name,
        )
        for owner in CHARACTER_NAMES
    }


def _gallery_cache_metadata(cfg: Config, versions: dict[str, str]) -> dict:
    return {
        "schema_version": GALLERY_CACHE_SCHEMA_VERSION,
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
    return stable_json_hash(_gallery_cache_metadata(cfg, versions))


def _embed_gallery_photo(path: Path, cfg: Config) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("image is unreadable")

    perception.build_models(cfg)
    results = perception._get_deepface().represent(
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
    return validated_embedding(results[0].get("embedding"))


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
        validated_embedding(record.embedding)
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
                embedding=validated_embedding(embeddings[index]),
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
        cache_metadata = _gallery_cache_metadata(cfg, installed_versions())
        cache_key = gallery_cache_key(cfg, cache_metadata["versions"])
        cache_path = cfg.cache_dir / f"gallery_{cache_key}.npz"
        cached_records = _load_gallery_cache(cache_path, cache_metadata)

    records: list[GalleryPhoto] = []

    for owner in CHARACTER_NAMES:
        paths = paths_by_owner[owner]
        owner_records: list[GalleryPhoto] = []
        for path in paths:
            relative_path = path.relative_to(ref_dir).as_posix()
            file_hash = file_sha256(path)
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
