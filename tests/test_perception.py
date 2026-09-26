from pathlib import Path
import time

import numpy as np
import pytest

import label_video
from face_labeller import perception
from face_labeller.config import Config
from face_labeller.contracts import Face, Match
from face_labeller.perception import build_models, embed_faces
from face_labeller.rendering import (
    BOX_COLORS,
    LANDMARK_COLOR,
    draw,
)


def make_config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "input_path": Path("input.mp4"),
        "output_path": Path("output.mp4"),
        "ref_dir": Path("references"),
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def unit_embedding(index: int = 0) -> list[float]:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[index] = 1.0
    return embedding.tolist()


def result(
    *,
    embedding: list[float] | None = None,
    confidence: float = 0.99,
    area: dict | None = None,
) -> dict:
    return {
        "embedding": embedding if embedding is not None else unit_embedding(),
        "facial_area": area
        or {
            "x": 1,
            "y": 2,
            "w": 3,
            "h": 4,
            "left_eye": (2, 3),
            "right_eye": None,
            "nose": (3, 4),
        },
        "face_confidence": confidence,
    }


@pytest.fixture(autouse=True)
def reset_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perception, "_MODEL_BUILT", False)


def test_model_load_seconds_exposes_timing_without_mutable_global_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perception, "_MODEL_LOAD_SECONDS_TOTAL", 1.25)

    assert perception.model_load_seconds() == 1.25


def test_build_models_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeDeepFace:
        @staticmethod
        def build_model(model_name: str) -> object:
            calls.append(model_name)
            return object()

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)

    build_models(make_config())
    build_models(make_config())

    assert calls == ["Facenet512"]


def test_embed_faces_passes_explicit_approved_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    represent_calls: list[dict] = []

    class FakeDeepFace:
        @staticmethod
        def build_model(model_name: str) -> object:
            return object()

        @staticmethod
        def represent(**kwargs: object) -> list[dict]:
            represent_calls.append(kwargs)
            return [result()]

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)
    frame = np.zeros((12, 10, 3), dtype=np.uint8)

    embed_faces([frame], make_config(max_faces=5, normalization="Facenet2018"))

    assert len(represent_calls) == 1
    call = represent_calls[0]
    assert call["img_path"] == [frame]
    assert call["model_name"] == "Facenet512"
    assert call["detector_backend"] == "retinaface"
    assert call["enforce_detection"] is False
    assert call["align"] is True
    assert call["normalization"] == "Facenet2018"
    assert call["max_faces"] == 5
    assert call["l2_normalize"] is True
    assert call["expand_percentage"] == 0


def test_embed_faces_normalizes_flat_single_frame_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(lambda **kwargs: [result(), result(embedding=unit_embedding(1))])

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)

    faces_by_frame = embed_faces([np.zeros((12, 10, 3), dtype=np.uint8)], make_config())

    assert len(faces_by_frame) == 1
    assert len(faces_by_frame[0]) == 2


def test_embed_faces_preserves_multi_frame_order(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(
            lambda **kwargs: [[result(embedding=unit_embedding(1))], [result(embedding=unit_embedding(2))]]
        )

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    faces_by_frame = embed_faces(frames, make_config())

    assert int(np.argmax(faces_by_frame[0][0].embedding)) == 1
    assert int(np.argmax(faces_by_frame[1][0].embedding)) == 2


def test_embed_faces_filters_placeholder_and_drops_none_landmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(lambda **kwargs: [result(confidence=0), result()])

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)

    faces = embed_faces([np.zeros((12, 10, 3), dtype=np.uint8)], make_config())[0]

    assert len(faces) == 1
    assert faces[0].landmarks == {"left_eye": (2, 3), "nose": (3, 4)}


def test_embed_faces_clips_boxes_and_returns_float32_unit_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    area = {"x": -5, "y": -4, "w": 30, "h": 40, "nose": (2, 3)}

    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(lambda **kwargs: [result(area=area)])

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)

    face = embed_faces([np.zeros((12, 10, 3), dtype=np.uint8)], make_config())[0][0]

    assert face.box == (0, 0, 10, 12)
    assert face.embedding.shape == (512,)
    assert face.embedding.dtype == np.float32
    assert np.linalg.norm(face.embedding) == pytest.approx(1.0, abs=1e-5)


def test_embed_faces_can_return_no_real_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(lambda **kwargs: [result(confidence=0)])

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)

    assert embed_faces([np.zeros((8, 8, 3), dtype=np.uint8)], make_config()) == [[]]


@pytest.mark.parametrize(
    "embedding",
    [unit_embedding()[:-1], (np.asarray(unit_embedding()) * 2).tolist()],
)
def test_embed_faces_rejects_invalid_embeddings(
    monkeypatch: pytest.MonkeyPatch, embedding: list[float]
) -> None:
    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(lambda **kwargs: [result(embedding=embedding)])

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)

    with pytest.raises(ValueError, match="embedding"):
        embed_faces([np.zeros((8, 8, 3), dtype=np.uint8)], make_config())


def test_embed_faces_rejects_empty_frame_batch() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        embed_faces([], make_config())


def make_face(
    *,
    box: tuple[int, int, int, int] = (20, 20, 30, 30),
    landmarks: dict[str, tuple[int, int]] | None = None,
) -> Face:
    return Face(
        box=box,
        landmarks=landmarks or {},
        embedding=np.asarray(unit_embedding(), dtype=np.float32),
        det_conf=0.99,
    )


def test_draw_returns_same_shape_dtype_without_mutating_input() -> None:
    frame = np.full((100, 120, 3), 17, dtype=np.uint8)
    before = frame.copy()
    face = make_face()
    match = Match("Harry Potter", "Harry Potter", 0.1, 84.2)

    rendered = draw(frame, [face], [match], make_config())

    assert rendered.shape == frame.shape
    assert rendered.dtype == frame.dtype
    assert rendered is not frame
    np.testing.assert_array_equal(frame, before)
    assert tuple(rendered[20, 20]) == BOX_COLORS["Harry Potter"]


def test_draw_uses_grey_for_unknown() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    face = make_face()
    match = Match("Harry Potter", None, 0.31, 48.0)

    rendered = draw(frame, [face], [match], make_config())

    assert tuple(rendered[20, 20]) == BOX_COLORS[None]


@pytest.mark.parametrize(
    "box",
    [(0, 0, 10, 10), (110, 0, 10, 10), (0, 90, 10, 10), (110, 90, 10, 10)],
)
def test_draw_keeps_edge_labels_inside_frame(box: tuple[int, int, int, int]) -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    rendered = draw(
        frame,
        [make_face(box=box)],
        [Match("Prof. Severus Snape", "Prof. Severus Snape", 0.1, 99.0)],
        make_config(),
    )

    assert np.count_nonzero(rendered) > 0
    assert rendered.shape == (100, 120, 3)


def test_draw_zero_faces_returns_equal_copy() -> None:
    frame = np.full((20, 30, 3), 9, dtype=np.uint8)

    rendered = draw(frame, [], [], make_config())

    assert rendered is not frame
    np.testing.assert_array_equal(rendered, frame)


def test_draw_skips_zero_area_box() -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    rendered = draw(
        frame,
        [make_face(box=(3, 4, 0, 10))],
        [Match("Harry Potter", None, 0.5, 20.0)],
        make_config(),
    )

    np.testing.assert_array_equal(rendered, frame)


def test_draw_landmark_dots_only_when_enabled() -> None:
    landmarks = {
        "left_eye": (30, 30),
        "right_eye": (40, 30),
        "nose": (35, 35),
        "mouth_left": (31, 40),
        "mouth_right": (39, 40),
    }
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    face = make_face(box=(20, 20, 30, 30), landmarks=landmarks)
    match = Match("Harry Potter", None, 0.5, 20.0)

    without_landmarks = draw(frame, [face], [match], make_config())
    with_landmarks = draw(
        frame, [face], [match], make_config(debug_landmarks=True)
    )

    for x, y in landmarks.values():
        assert tuple(without_landmarks[y, x]) != LANDMARK_COLOR
        assert tuple(with_landmarks[y, x]) == LANDMARK_COLOR


def test_draw_rejects_mismatched_faces_and_matches() -> None:
    with pytest.raises(ValueError, match="same length"):
        draw(np.zeros((20, 20, 3), dtype=np.uint8), [make_face()], [], make_config())


def read_video_frame(frame_index: int) -> np.ndarray:
    capture = label_video.cv2.VideoCapture("data/video-source/nimbus.mp4")
    assert capture.isOpened()
    capture.set(label_video.cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None
    return frame


@pytest.mark.slow
def test_real_frame_artifact_and_face_less_frame() -> None:
    face_frame = read_video_frame(150)
    face_less_frame = read_video_frame(0)
    cfg = make_config(debug_landmarks=True)

    model_started = time.perf_counter()
    build_models(cfg)
    model_seconds = time.perf_counter() - model_started

    perception_started = time.perf_counter()
    faces_by_frame = embed_faces([face_frame, face_less_frame], cfg)
    perception_seconds = time.perf_counter() - perception_started

    assert len(faces_by_frame[0]) == 2
    assert faces_by_frame[1] == []

    matches = [
        Match("Unknown", None, float("inf"), 0.0) for _ in faces_by_frame[0]
    ]
    rendered = draw(face_frame, faces_by_frame[0], matches, cfg)
    output_path = Path("output/m1_single_frame.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert label_video.cv2.imwrite(str(output_path), rendered)
    assert output_path.is_file()
    print(
        f"model_load_seconds={model_seconds:.3f} "
        f"perception_seconds={perception_seconds:.3f} "
        f"faces={len(faces_by_frame[0])} output={output_path}"
    )
