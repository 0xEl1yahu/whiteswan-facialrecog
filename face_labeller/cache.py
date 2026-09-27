"""Stable cache identity helpers and persistent per-frame face results."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import tempfile
from typing import Literal, Sequence

import numpy as np

from face_labeller.config import Config
from face_labeller.contracts import Face


FACE_CACHE_SCHEMA_VERSION = 2
VERSION_DISTRIBUTIONS = {
    "deepface": "deepface",
    "retinaface": "retina-face",
    "tensorflow": "tensorflow",
    "opencv": "opencv-python",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_versions() -> dict[str, str]:
    return {
        name: importlib_metadata.version(distribution)
        for name, distribution in VERSION_DISTRIBUTIONS.items()
    }


def stable_json_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def face_cache_metadata(
    video_path: Path, cfg: Config, versions: dict[str, str]
) -> dict:
    return {
        "schema_version": FACE_CACHE_SCHEMA_VERSION,
        "video_sha256": file_sha256(video_path),
        "model_name": cfg.model_name,
        "detector_backend": cfg.detector_backend,
        "normalization": cfg.normalization,
        "align": cfg.align,
        "max_faces": cfg.max_faces,
        "expand_percentage": cfg.expand_percentage,
        "perception_pipeline": cfg.perception_pipeline,
        "detector_black_halo": cfg.detector_black_halo,
        "l2_normalize": True,
        "versions": dict(sorted(versions.items())),
    }


def face_cache_key(
    video_path: Path, cfg: Config, versions: dict[str, str]
) -> str:
    """Hash every upstream perception input and no downstream choice."""
    return stable_json_hash(face_cache_metadata(video_path, cfg, versions))


def validated_embedding(value: object) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32)
    if embedding.shape != (512,):
        raise ValueError(f"embedding must have shape (512,), got {embedding.shape}")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError(f"embedding must be L2-normalised, got norm {norm}")
    return embedding


class FaceCache:
    """Persistent per-frame perception results with explicit retryable states."""

    def __init__(
        self,
        video_path: Path,
        cfg: Config,
        versions: dict[str, str] | None = None,
    ) -> None:
        self._enabled = cfg.use_cache
        self._metadata = face_cache_metadata(
            video_path, cfg, versions if versions is not None else installed_versions()
        )
        key = stable_json_hash(self._metadata)
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
                    embedding=validated_embedding(face.embedding).copy(),
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
                        embedding=validated_embedding(embeddings[face_index]),
                        det_conf=float(confidences[face_index]),
                    )
                )
            decoded[frame_idx] = (status, frame_faces)  # type: ignore[assignment]
        return decoded
