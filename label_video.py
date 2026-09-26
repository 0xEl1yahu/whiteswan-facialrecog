"""Compatibility facade and CLI entry point for the face labeller."""

from __future__ import annotations

import sys
from typing import Sequence

from face_labeller.cache import FaceCache, face_cache_key
from face_labeller.config import Config, build_parser, load_config, parse_args
from face_labeller.contracts import (
    CHARACTER_NAMES,
    DETECTOR_BACKEND,
    MODEL_NAME,
    Face,
    FramePlan,
    Gallery,
    GalleryPhoto,
    Match,
    RunSummary,
    Track,
    VideoMetadata,
)
from face_labeller.core import run
from face_labeller.evidence import MatchLogger
from face_labeller.gallery import (
    build_pins,
    gallery_cache_key,
    leave_one_out_report,
    load_gallery,
)
from face_labeller.perception import (
    build_models,
    embed_faces,
    model_load_seconds,
    process_batch_with_fallback,
)
from face_labeller.recognition import cosine_distances, match, unknown_matches
from face_labeller.rendering import BOX_COLORS, LANDMARK_COLOR, draw
from face_labeller.video import (
    build_frame_plan,
    inspect_video,
    process_video,
    selected_frame_indices,
)


def main(argv: Sequence[str] | None = None) -> int:
    cfg = load_config(parse_args(argv))
    try:
        run(cfg)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
