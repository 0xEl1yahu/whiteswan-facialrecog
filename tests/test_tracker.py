from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from face_labeller.contracts import Face, Match
from face_labeller.tracking import Tracker, iou


def make_face(box: tuple[int, int, int, int]) -> Face:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return Face(box=box, landmarks={}, embedding=embedding, det_conf=0.95)


def make_match(name: str | None, *, nearest: str = "Harry Potter") -> Match:
    return Match(
        nearest_name=nearest,
        name=name,
        distance=0.2 if name is not None else 0.4,
        confidence=80.0 if name is not None else 20.0,
    )


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
        ((0, 0, 10, 10), (20, 20, 5, 5), 0.0),
        ((0, 0, 10, 10), (10, 0, 10, 10), 0.0),
        ((0, 0, 0, 10), (0, 0, 10, 10), 0.0),
        ((0, 0, 10, 10), (5, 5, 10, 10), 1.0 / 7.0),
    ],
)
def test_iou_handles_exact_box_geometry(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    expected: float,
) -> None:
    assert iou(a, b) == pytest.approx(expected)
    assert iou(b, a) == pytest.approx(expected)


def test_tracker_associates_at_boundary_and_starts_new_track_below_it() -> None:
    first = make_face((0, 0, 10, 10))
    at_boundary = make_face((5, 0, 10, 10))
    below_boundary = make_face((11, 0, 10, 10))
    boundary = iou(first.box, at_boundary.box)
    tracker = Tracker(iou_min=boundary, track_ttl=15)

    assert tracker.update(0, [first], [make_match("Harry Potter")])[0][2] == 0
    assert tracker.update(1, [at_boundary], [make_match("Harry Potter")])[0][2] == 0
    assert tracker.update(2, [below_boundary], [make_match("Harry Potter")])[0][2] == 1


def test_tracker_greedily_associates_multiple_faces_and_preserves_face_order() -> None:
    left = make_face((0, 0, 10, 10))
    right = make_face((20, 0, 10, 10))
    tracker = Tracker(iou_min=0.3, track_ttl=15)

    initial = tracker.update(
        0,
        [left, right],
        [make_match("Harry Potter"), make_match("Ron Weasley", nearest="Ron Weasley")],
    )
    reversed_faces = tracker.update(
        1,
        [make_face((21, 0, 10, 10)), make_face((1, 0, 10, 10))],
        [make_match("Ron Weasley", nearest="Ron Weasley"), make_match("Harry Potter")],
    )

    assert [track_id for _, _, track_id in initial] == [0, 1]
    assert [track_id for _, _, track_id in reversed_faces] == [1, 0]


def test_tracker_uses_stable_tie_breaks_for_equally_good_associations() -> None:
    tracker = Tracker(iou_min=0.1, track_ttl=15)
    tracker.update(
        0,
        [make_face((0, 0, 10, 10)), make_face((10, 0, 10, 10))],
        [make_match(None), make_match(None)],
    )

    result = tracker.update(
        1,
        [make_face((5, 0, 10, 10)), make_face((5, 0, 10, 10))],
        [make_match(None), make_match(None)],
    )

    assert [track_id for _, _, track_id in result] == [0, 1]


def test_tracker_expires_after_exactly_ttl_unseen_frames() -> None:
    tracker = Tracker(iou_min=0.3, track_ttl=2)
    face = make_face((0, 0, 10, 10))
    match = make_match("Harry Potter")

    assert tracker.update(0, [face], [match])[0][2] == 0
    assert tracker.update(1, [], []) == []
    assert tracker.update(2, [face], [match])[0][2] == 1


def test_tracker_majority_vote_and_unknown_tie_winner() -> None:
    tracker = Tracker(iou_min=0.3, track_ttl=15)
    face = make_face((0, 0, 10, 10))
    harry = make_match("Harry Potter")
    unknown = make_match(None)

    first = tracker.update(0, [face], [harry])
    tied = tracker.update(1, [face], [unknown])
    majority = tracker.update(2, [face], [harry])

    assert first[0][1].name == "Harry Potter"
    assert tied[0][1].name is None
    assert majority[0][1].name == "Harry Potter"


def test_tracker_displays_unknown_for_a_tie_between_named_identities() -> None:
    tracker = Tracker(iou_min=0.3, track_ttl=15)
    face = make_face((0, 0, 10, 10))

    tracker.update(0, [face], [make_match("Harry Potter")])
    tied = tracker.update(
        1, [face], [make_match("Ron Weasley", nearest="Ron Weasley")]
    )

    assert tied[0][1].name is None


def test_tracker_does_not_mutate_inputs_or_non_name_match_fields() -> None:
    tracker = Tracker(iou_min=0.3, track_ttl=15)
    face = make_face((0, 0, 10, 10))
    first_match = make_match("Harry Potter")
    current_match = make_match(None, nearest="Hermione Granger")
    original_box = face.box
    original_embedding = face.embedding.copy()
    original_match = replace(current_match)

    tracker.update(0, [face], [first_match])
    _, displayed, _ = tracker.update(1, [face], [current_match])[0]

    assert face.box == original_box
    np.testing.assert_array_equal(face.embedding, original_embedding)
    assert current_match == original_match
    assert displayed.name is None
    assert replace(displayed, name=current_match.name) == current_match


def test_tracker_rejects_mismatched_faces_and_matches() -> None:
    tracker = Tracker(iou_min=0.3, track_ttl=15)

    with pytest.raises(ValueError, match="same length"):
        tracker.update(0, [make_face((0, 0, 10, 10))], [])
