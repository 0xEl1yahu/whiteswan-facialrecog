"""Lazy DeepFace model lifecycle and frame perception."""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

import numpy as np

from face_labeller.cache import validated_embedding
from face_labeller.config import Config
from face_labeller.contracts import Face


_MODEL_BUILT = False
_MODEL_LOAD_SECONDS_TOTAL = 0.0


def _get_deepface() -> Any:
    from deepface import DeepFace

    return DeepFace


def model_load_seconds() -> float:
    return _MODEL_LOAD_SECONDS_TOTAL


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

    embedding = validated_embedding(result["embedding"])
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
            for frame_idx, faces in zip(frame_indices, faces_by_frame, strict=True)
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
