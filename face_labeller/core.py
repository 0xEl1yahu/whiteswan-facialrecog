"""Application orchestration for one complete face-labelling run."""

from __future__ import annotations

import time

from face_labeller import perception
from face_labeller.cache import FaceCache
from face_labeller.config import Config
from face_labeller.contracts import RunSummary
from face_labeller.evidence import validate_csv_path
from face_labeller.gallery import load_gallery
from face_labeller.video import build_frame_plan, inspect_video, process_video


def validate_paths(cfg: Config) -> None:
    """Validate cross-path constraints before opening models or output files."""
    if not cfg.input_path.is_file():
        raise ValueError(f"cannot open input video: {cfg.input_path}")
    if cfg.input_path.resolve() == cfg.output_path.resolve():
        raise ValueError("input and output video paths must differ")
    if not cfg.ref_dir.is_dir():
        raise ValueError(f"reference directory does not exist: {cfg.ref_dir}")
    validate_csv_path(cfg)


def run(cfg: Config) -> RunSummary:
    """Preflight, load the gallery, process the video, and report the run."""
    run_started = time.perf_counter()
    model_seconds_before = perception.model_load_seconds()
    validate_paths(cfg)
    metadata = inspect_video(cfg.input_path)
    plan = build_frame_plan(
        metadata,
        start_frame=cfg.start_frame,
        max_frames=cfg.max_frames,
        stride=cfg.stride,
    )
    face_cache = FaceCache(cfg.input_path, cfg)
    pending_indices = tuple(face_cache.missing(plan.selected_indices))
    print(
        f"preflight input={cfg.input_path} total={metadata.frame_count} "
        f"window={plan.start_frame}:{plan.stop_frame} "
        f"written={plan.written_frames} stride={cfg.stride} "
        f"selected={len(plan.selected_indices)} "
        f"cached={len(plan.selected_indices) - len(pending_indices)} "
        f"to_infer={len(pending_indices)}"
    )
    gallery = load_gallery(cfg.ref_dir, cfg)
    gallery_seconds = time.perf_counter() - run_started
    gallery_model_seconds = perception.model_load_seconds() - model_seconds_before
    summary = process_video(
        cfg,
        gallery,
        metadata=metadata,
        plan=plan,
        face_cache=face_cache,
    )
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
    return summary
