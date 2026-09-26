"""Pure deterministic frame annotation."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from face_labeller.config import Config
from face_labeller.contracts import Face, Match


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
