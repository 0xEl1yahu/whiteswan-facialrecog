"""Video metadata inspection and deterministic frame planning."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import time
from typing import Any

import cv2
import numpy as np

from face_labeller.cache import FaceCache
from face_labeller.config import Config
from face_labeller.contracts import (
    Face,
    FramePlan,
    Gallery,
    Match,
    RunSummary,
    VideoMetadata,
)
from face_labeller.evidence import MatchLogger
from face_labeller.perception import model_load_seconds, process_batch_with_fallback
from face_labeller.recognition import match, unknown_matches
from face_labeller.rendering import draw
from face_labeller.tracking import Tracker


PROGRESS_EVERY_FRAMES = 100  # M3 reporting interval; does not affect output.


def inspect_video(input_path: Path) -> VideoMetadata:
    """Read and validate input metadata without loading gallery or ML models."""
    capture = cv2.VideoCapture(str(input_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open input video: {input_path}")
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
        return VideoMetadata(input_path, fps, width, height, frame_count)
    finally:
        capture.release()


def selected_frame_indices(start: int, stop: int, stride: int) -> list[int]:
    """Return absolute selected indices, with stride relative to the window start."""
    if start < 0:
        raise ValueError("start frame must be non-negative")
    if stop < start:
        raise ValueError("stop frame must not precede start frame")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    return list(range(start, stop, stride))


def build_frame_plan(
    metadata: VideoMetadata,
    *,
    start_frame: int,
    max_frames: int | None,
    stride: int,
) -> FramePlan:
    """Calculate the exact output window and selected absolute frame indices."""
    if start_frame < 0:
        raise ValueError("start frame must be non-negative")
    if start_frame >= metadata.frame_count:
        raise ValueError(
            f"start frame {start_frame} is outside {metadata.frame_count}-frame input"
        )
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max frames must be greater than zero")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")

    stop_frame = metadata.frame_count
    if max_frames is not None:
        stop_frame = min(metadata.frame_count, start_frame + max_frames)
    selected = selected_frame_indices(start_frame, stop_frame, stride)
    return FramePlan(
        start_frame=start_frame,
        stop_frame=stop_frame,
        written_frames=stop_frame - start_frame,
        selected_indices=tuple(selected),
    )


def process_video(
    cfg: Config,
    gallery: Gallery | None = None,
    *,
    metadata: VideoMetadata | None = None,
    plan: FramePlan | None = None,
    face_cache: FaceCache | None = None,
) -> RunSummary:
    """Write the requested video window with cached face annotations."""
    if not cfg.input_path.is_file():
        raise ValueError(f"cannot open input video: {cfg.input_path}")
    if cfg.input_path.resolve() == cfg.output_path.resolve():
        raise ValueError("input and output video paths must differ")

    supplied_preflight = (metadata is not None, plan is not None, face_cache is not None)
    if any(supplied_preflight) and not all(supplied_preflight):
        raise ValueError("metadata, plan, and face_cache must be supplied together")
    if metadata is None:
        metadata = inspect_video(cfg.input_path)
        plan = build_frame_plan(
            metadata,
            start_frame=cfg.start_frame,
            max_frames=cfg.max_frames,
            stride=cfg.stride,
        )
        face_cache = FaceCache(cfg.input_path, cfg)
    assert plan is not None
    assert face_cache is not None
    if metadata.input_path != cfg.input_path:
        raise ValueError("preflight metadata input path does not match configuration")

    run_started = time.perf_counter()
    model_seconds_before = model_load_seconds()
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
        observed_metadata = (fps, width, height, frame_count)
        expected_metadata = (
            metadata.fps,
            metadata.width,
            metadata.height,
            metadata.frame_count,
        )
        if observed_metadata != expected_metadata:
            raise ValueError(
                "input video metadata changed after preflight: "
                f"expected {expected_metadata}, got {observed_metadata}"
            )

        stop_frame = plan.stop_frame
        selected = list(plan.selected_indices)
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

        cache = face_cache
        initial_status = {frame_idx: cache.status(frame_idx) for frame_idx in selected}
        cache_hits = sum(status == "ok" for status in initial_status.values())
        cache_misses = len(selected) - cache_hits
        perception_seconds = 0.0
        written_frames = 0
        last_faces: list[Face] = []
        last_matches: list[Match] = []
        last_display_matches: list[Match] = []
        label_counts: Counter[str] = Counter()
        tracker = (
            Tracker(iou_min=cfg.iou_min, track_ttl=cfg.track_ttl)
            if cfg.smooth
            else None
        )
        pending_frames: list[tuple[int, np.ndarray]] = []
        pending_missing: list[tuple[int, np.ndarray]] = []
        max_pending_frames = max(1, cfg.batch_size * cfg.stride)

        def flush_pending() -> None:
            nonlocal perception_seconds, written_frames, last_faces, last_matches
            nonlocal last_display_matches
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
                        else unknown_matches(last_faces)
                    )
                    last_display_matches = (
                        [
                            displayed_match
                            for _face, displayed_match, _track_id in tracker.update(
                                frame_idx, last_faces, last_matches
                            )
                        ]
                        if tracker is not None
                        else last_matches
                    )
                rendered = draw(frame, last_faces, last_display_matches, cfg)
                for face_idx, (face, raw_assignment, displayed_assignment) in enumerate(
                    zip(
                        last_faces,
                        last_matches,
                        last_display_matches,
                        strict=True,
                    )
                ):
                    label_counts[displayed_assignment.name or "Unknown"] += 1
                    if logger is not None:
                        logger.log(
                            frame_idx,
                            face_idx,
                            face,
                            raw_assignment,
                            cfg.threshold,
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

        if plan.start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, plan.start_frame)
        for frame_idx in range(plan.start_frame, stop_frame):
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

        expected_written = plan.written_frames
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
        model_seconds = model_load_seconds() - model_seconds_before
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
