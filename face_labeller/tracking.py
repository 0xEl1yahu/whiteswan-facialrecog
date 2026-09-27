"""Pure overlap geometry and optional temporal label smoothing."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from face_labeller.contracts import Face, Match, Track


_UNKNOWN_VOTE = "__unknown__"


def iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    """Return intersection-over-union for two ``(x, y, width, height)`` boxes."""
    a_x, a_y, a_width, a_height = a
    b_x, b_y, b_width, b_height = b
    if a_width <= 0 or a_height <= 0 or b_width <= 0 or b_height <= 0:
        return 0.0

    intersection_width = max(
        0, min(a_x + a_width, b_x + b_width) - max(a_x, b_x)
    )
    intersection_height = max(
        0, min(a_y + a_height, b_y + b_height) - max(a_y, b_y)
    )
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    union = a_width * a_height + b_width * b_height - intersection
    return intersection / union


class Tracker:
    """Associate faces across frames and smooth only their displayed names."""

    def __init__(self, *, iou_min: float, track_ttl: int) -> None:
        if not 0 <= iou_min <= 1:
            raise ValueError("iou_min must be between zero and one")
        if track_ttl <= 0:
            raise ValueError("track_ttl must be greater than zero")
        self._iou_min = iou_min
        self._track_ttl = track_ttl
        self._tracks: dict[int, Track] = {}
        self._next_id = 0

    @staticmethod
    def _displayed_name(votes: Counter[str]) -> str | None:
        highest_count = max(votes.values())
        winners = [name for name, count in votes.items() if count == highest_count]
        if len(winners) != 1 or winners[0] == _UNKNOWN_VOTE:
            return None
        return winners[0]

    def update(
        self, frame_idx: int, faces: list[Face], matches: list[Match]
    ) -> list[tuple[Face, Match, int]]:
        if len(faces) != len(matches):
            raise ValueError("faces and matches must have the same length")

        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_idx - track.last_seen >= self._track_ttl
        ]
        for track_id in expired_ids:
            del self._tracks[track_id]

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for face_index, face in enumerate(faces):
                overlap = iou(track.box, face.box)
                if overlap >= self._iou_min:
                    candidates.append((overlap, track_id, face_index))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        assigned_tracks: set[int] = set()
        assigned_faces: dict[int, int] = {}
        for _overlap, track_id, face_index in candidates:
            if track_id in assigned_tracks or face_index in assigned_faces:
                continue
            assigned_tracks.add(track_id)
            assigned_faces[face_index] = track_id

        results: list[tuple[Face, Match, int]] = []
        for face_index, (face, current_match) in enumerate(
            zip(faces, matches, strict=True)
        ):
            track_id = assigned_faces.get(face_index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(
                    id=track_id,
                    box=face.box,
                    last_seen=frame_idx,
                    votes=Counter(),
                )
            track = self._tracks[track_id]
            track.box = face.box
            track.last_seen = frame_idx
            track.votes[current_match.name or _UNKNOWN_VOTE] += 1
            displayed_match = replace(
                current_match, name=self._displayed_name(track.votes)
            )
            results.append((face, displayed_match, track_id))
        return results
