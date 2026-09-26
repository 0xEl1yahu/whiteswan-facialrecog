"""Pure face-to-gallery recognition."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from deepface.modules import verification

from face_labeller.contracts import Face, Gallery, Match, MODEL_NAME


def unknown_matches(faces: Sequence[Face]) -> list[Match]:
    return [
        Match(
            nearest_name="Unknown",
            name=None,
            distance=float("inf"),
            confidence=0.0,
        )
        for _ in faces
    ]


def cosine_distances(v: np.ndarray, pins: np.ndarray) -> np.ndarray:
    """Return cosine distances for L2-normalised embeddings and gallery pins."""
    return 1 - pins @ v


def match(face: Face, gallery: Gallery, threshold: float) -> Match:
    """Assign the nearest gallery owner only below the strict threshold."""
    distances = cosine_distances(face.embedding, gallery.pins)
    nearest_index = int(np.argmin(distances))
    distance = float(distances[nearest_index])
    nearest_name = gallery.pin_owner[nearest_index]
    verified = distance < threshold
    return Match(
        nearest_name=nearest_name,
        name=nearest_name if verified else None,
        distance=distance,
        confidence=float(
            verification.find_confidence(distance, MODEL_NAME, verified, "cosine")
        ),
    )
