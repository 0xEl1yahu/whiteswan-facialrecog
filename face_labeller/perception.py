"""Lazy DeepFace model lifecycle and frame perception."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest
import math
import time
from typing import Any, Literal, Sequence

import cv2
import numpy as np

from face_labeller.cache import validated_embedding
from face_labeller.config import Config
from face_labeller.contracts import Face


_MODEL_BUILT = False
_MODEL_LOAD_SECONDS_TOTAL = 0.0
_DETECTOR_MODEL: Any | None = None
_DETECTOR_MODEL_LOAD_SECONDS_TOTAL = 0.0

# RetinaFace 0.0.18 preprocessing constants and coarsest FPN stride.
_RETINAFACE_RESIZE_SCALES = [1024, 1980]
_RETINAFACE_COARSEST_STRIDE = 32
_RETINAFACE_DETECTION_THRESHOLD = 0.9  # DeepFace 0.0.101 RetinaFace adapter.


@dataclass(frozen=True)
class _DetectorInput:
    image: np.ndarray
    scale: float
    crop_x: int
    crop_y: int
    width_border: int
    height_border: int


@dataclass(frozen=True)
class _AlignedFace:
    box: tuple[int, int, int, int]
    landmarks: dict[str, tuple[int, int]]
    det_conf: float
    image: np.ndarray


def _get_retinaface_preprocess() -> Any:
    from retinaface.commons import preprocess

    return preprocess


def _get_deepface_detection() -> Any:
    from deepface.modules import detection

    return detection


def _prepare_detector_input(frame: np.ndarray, black_halo: int) -> _DetectorInput:
    if frame.ndim != 3:
        raise ValueError("detector frame must be three-dimensional")
    if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("detector frame must be non-empty")
    if black_halo < 0:
        raise ValueError("detector black halo must be non-negative")
    if black_halo % _RETINAFACE_COARSEST_STRIDE != 0:
        raise ValueError("detector black halo must be a multiple of 32")

    frame_height, frame_width = frame.shape[:2]
    width_border = frame_width // 2
    height_border = frame_height // 2
    padded = cv2.copyMakeBorder(
        frame,
        height_border,
        height_border,
        width_border,
        width_border,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    resized, scale = _get_retinaface_preprocess().resize_image(
        padded, _RETINAFACE_RESIZE_SCALES, True
    )

    def crop_for(border: int) -> int:
        removable = border * scale - black_halo
        return max(
            0,
            math.floor(removable / _RETINAFACE_COARSEST_STRIDE)
            * _RETINAFACE_COARSEST_STRIDE,
        )

    crop_x = crop_for(width_border)
    crop_y = crop_for(height_border)
    resized_height, resized_width = resized.shape[:2]
    if crop_x * 2 >= resized_width or crop_y * 2 >= resized_height:
        raise ValueError("detector crop would be empty")
    image = resized[
        crop_y : resized_height - crop_y,
        crop_x : resized_width - crop_x,
    ]
    if image.size == 0:
        raise ValueError("detector crop would be empty")
    return _DetectorInput(
        image=image,
        scale=float(scale),
        crop_x=crop_x,
        crop_y=crop_y,
        width_border=width_border,
        height_border=height_border,
    )


def _restore_point(
    point: Sequence[float | int], prepared: _DetectorInput
) -> tuple[int, int]:
    return (
        int((point[0] + prepared.crop_x) / prepared.scale - prepared.width_border),
        int((point[1] + prepared.crop_y) / prepared.scale - prepared.height_border),
    )


def _detect_and_align_faces(frame: np.ndarray, cfg: Config) -> list[_AlignedFace]:
    prepared = _prepare_detector_input(frame, cfg.detector_black_halo)
    detector_model = build_detector_model()
    detections = _get_retinaface().detect_faces(
        prepared.image,
        model=detector_model,
        threshold=_RETINAFACE_DETECTION_THRESHOLD,
        allow_upscaling=False,
    )
    if not isinstance(detections, dict):
        return []

    detection_module = _get_deepface_detection()
    regions: list[Any] = []
    for detection in detections.values():
        x1, y1 = _restore_point(detection["facial_area"][:2], prepared)
        x2, y2 = _restore_point(detection["facial_area"][2:], prepared)
        mapped_landmarks = {
            name: _restore_point(point, prepared)
            for name, point in detection["landmarks"].items()
            if point is not None
        }
        regions.append(
            detection_module.FacialAreaRegion(
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                left_eye=mapped_landmarks.get("left_eye"),
                right_eye=mapped_landmarks.get("right_eye"),
                confidence=float(detection["score"]),
                nose=mapped_landmarks.get("nose"),
                mouth_left=mapped_landmarks.get("mouth_left"),
                mouth_right=mapped_landmarks.get("mouth_right"),
            )
        )

    if cfg.max_faces is not None and cfg.max_faces < len(regions):
        regions = nlargest(
            cfg.max_faces, regions, key=lambda region: region.w * region.h
        )

    frame_height, frame_width = frame.shape[:2]
    aligned_faces: list[_AlignedFace] = []
    for region in regions:
        detected_face = detection_module.extract_face(
            facial_area=region,
            img=frame,
            align=True,
            expand_percentage=cfg.expand_percentage,
            width_border=0,
            height_border=0,
            detector_backend="retinaface",
        )
        published_region = detected_face.facial_area
        x = max(0, int(published_region.x))
        y = max(0, int(published_region.y))
        width = max(0, min(frame_width - x - 1, int(published_region.w)))
        height = max(0, min(frame_height - y - 1, int(published_region.h)))
        landmarks = {}
        for name in (
            "left_eye",
            "right_eye",
            "nose",
            "mouth_left",
            "mouth_right",
        ):
            point = getattr(published_region, name, None)
            if (
                point is not None
                and 0 <= point[0] < frame_width
                and 0 <= point[1] < frame_height
            ):
                landmarks[name] = (int(point[0]), int(point[1]))
        aligned_faces.append(
            _AlignedFace(
                box=(x, y, width, height),
                landmarks=landmarks,
                det_conf=round(float(published_region.confidence or 0), 2),
                image=detected_face.img,
            )
        )
    return aligned_faces


def _get_deepface() -> Any:
    from deepface import DeepFace

    return DeepFace


def _get_retinaface() -> Any:
    from retinaface import RetinaFace

    return RetinaFace


def facenet_model_load_seconds() -> float:
    return _MODEL_LOAD_SECONDS_TOTAL


def detector_model_load_seconds() -> float:
    return _DETECTOR_MODEL_LOAD_SECONDS_TOTAL


def model_load_seconds() -> float:
    return facenet_model_load_seconds() + detector_model_load_seconds()


def build_detector_model() -> Any:
    """Load RetinaFace once, only when an uncached detection path requests it."""
    global _DETECTOR_MODEL, _DETECTOR_MODEL_LOAD_SECONDS_TOTAL
    if _DETECTOR_MODEL is not None:
        return _DETECTOR_MODEL
    started = time.perf_counter()
    _DETECTOR_MODEL = _get_retinaface().build_model()
    _DETECTOR_MODEL_LOAD_SECONDS_TOTAL += time.perf_counter() - started
    return _DETECTOR_MODEL


def build_models(cfg: Config) -> None:
    """Load Facenet512 once, only when an uncached embedding path requests it."""
    global _MODEL_BUILT, _MODEL_LOAD_SECONDS_TOTAL
    if _MODEL_BUILT:
        return
    started = time.perf_counter()
    _get_deepface().build_model(cfg.model_name)
    _MODEL_LOAD_SECONDS_TOTAL += time.perf_counter() - started
    _MODEL_BUILT = True


def _per_crop_results(raw_results: list, crop_count: int) -> list[list[dict]]:
    if crop_count == 1:
        if not raw_results:
            raise ValueError("DeepFace embedding result count does not match aligned crops")
        if isinstance(raw_results[0], dict):
            if len(raw_results) != 1:
                raise ValueError(
                    "DeepFace embedding result count does not match aligned crops"
                )
            return [raw_results]
        if len(raw_results) == 1 and isinstance(raw_results[0], list):
            return raw_results
        raise ValueError("unexpected DeepFace embedding result shape for one crop")

    if len(raw_results) != crop_count:
        raise ValueError("DeepFace embedding result count does not match aligned crops")
    if any(not isinstance(crop_results, list) for crop_results in raw_results):
        raise ValueError("unexpected DeepFace embedding result shape for crop batch")
    return raw_results


def embed_faces(frames: list[np.ndarray], cfg: Config) -> list[list[Face]]:
    """Detect, align, and embed every face in each supplied BGR frame."""
    if not frames:
        raise ValueError("embed_faces requires at least one frame")

    aligned_by_frame = [_detect_and_align_faces(frame, cfg) for frame in frames]
    flat_aligned = [face for frame_faces in aligned_by_frame for face in frame_faces]
    if not flat_aligned:
        return [[] for _ in frames]

    build_models(cfg)
    raw_results = _get_deepface().represent(
        img_path=[face.image for face in flat_aligned],
        model_name=cfg.model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=True,
        normalization=cfg.normalization,
        max_faces=None,
        l2_normalize=True,
        expand_percentage=0,
    )
    per_crop = _per_crop_results(raw_results, len(flat_aligned))
    if any(len(results) != 1 for results in per_crop):
        raise ValueError("DeepFace must return exactly one embedding per aligned crop")

    embedded = [
        Face(
            box=aligned.box,
            landmarks=aligned.landmarks,
            embedding=validated_embedding(results[0]["embedding"]),
            det_conf=aligned.det_conf,
        )
        for aligned, results in zip(flat_aligned, per_crop, strict=True)
    ]
    output: list[list[Face]] = []
    cursor = 0
    for frame_faces in aligned_by_frame:
        next_cursor = cursor + len(frame_faces)
        output.append(embedded[cursor:next_cursor])
        cursor = next_cursor
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
