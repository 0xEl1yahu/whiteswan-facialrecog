from __future__ import annotations

import label_video
from face_labeller import (
    cache,
    core,
    evidence,
    gallery,
    perception,
    recognition,
    rendering,
    video,
)
from label_video import Face, Gallery, Match, RunSummary


def test_label_video_exports_supported_api() -> None:
    supported_names = (
        "CHARACTER_NAMES",
        "MODEL_NAME",
        "DETECTOR_BACKEND",
        "BOX_COLORS",
        "LANDMARK_COLOR",
        "Config",
        "Face",
        "Match",
        "Gallery",
        "GalleryPhoto",
        "Track",
        "RunSummary",
        "FaceCache",
        "MatchLogger",
        "build_parser",
        "parse_args",
        "load_config",
        "build_models",
        "embed_faces",
        "gallery_cache_key",
        "face_cache_key",
        "build_pins",
        "load_gallery",
        "leave_one_out_report",
        "selected_frame_indices",
        "process_batch_with_fallback",
        "cosine_distances",
        "match",
        "draw",
        "process_video",
        "main",
    )

    missing = [name for name in supported_names if not hasattr(label_video, name)]

    assert missing == []


def test_approved_contract_fields_are_unchanged() -> None:
    assert tuple(Face.__dataclass_fields__) == (
        "box",
        "landmarks",
        "embedding",
        "det_conf",
    )
    assert tuple(Match.__dataclass_fields__) == (
        "nearest_name",
        "name",
        "distance",
        "confidence",
    )
    assert tuple(Gallery.__dataclass_fields__) == (
        "pins",
        "pin_owner",
        "names",
        "meta",
    )
    assert tuple(RunSummary.__dataclass_fields__) == (
        "elapsed_seconds",
        "model_seconds",
        "perception_seconds",
        "processed_frames",
        "written_frames",
        "faces",
        "cache_hits",
        "cache_misses",
        "failures",
        "processing_fps",
        "fps",
        "width",
        "height",
        "label_distribution",
    )


def test_extracted_behaviors_are_reexported_from_the_facade() -> None:
    assert label_video.FaceCache is cache.FaceCache
    assert label_video.build_models is perception.build_models
    assert label_video.embed_faces is perception.embed_faces
    assert label_video.process_batch_with_fallback is perception.process_batch_with_fallback
    assert label_video.match is recognition.match
    assert label_video.draw is rendering.draw
    assert label_video.MatchLogger is evidence.MatchLogger
    assert label_video.gallery_cache_key is gallery.gallery_cache_key
    assert label_video.build_pins is gallery.build_pins
    assert label_video.load_gallery is gallery.load_gallery
    assert label_video.leave_one_out_report is gallery.leave_one_out_report
    assert label_video.process_video is video.process_video
    assert label_video.run is core.run
