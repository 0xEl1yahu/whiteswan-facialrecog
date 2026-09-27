from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from retinaface.commons import preprocess as retinaface_preprocess

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
    monkeypatch.setattr(perception, "_MODEL_LOAD_SECONDS_TOTAL", 0.0)
    monkeypatch.setattr(perception, "_DETECTOR_MODEL", None, raising=False)
    monkeypatch.setattr(
        perception, "_DETECTOR_MODEL_LOAD_SECONDS_TOTAL", 0.0, raising=False
    )


def test_prepare_detector_input_preserves_exact_resized_pixels() -> None:
    y, x = np.indices((1080, 1920))
    frame = np.stack(
        ((x % 251), (y % 253), ((x + y) % 255)), axis=-1
    ).astype(np.uint8)

    prepared = perception._prepare_detector_input(frame, 32)

    padded = cv2.copyMakeBorder(
        frame, 540, 540, 960, 960, cv2.BORDER_CONSTANT, value=0
    )
    resized, expected_scale = retinaface_preprocess.resize_image(
        padded, [1024, 1980], True
    )
    assert prepared.scale == pytest.approx(1024 / 2160)
    assert prepared.scale == pytest.approx(expected_scale)
    assert prepared.width_border == 960
    assert prepared.height_border == 540
    assert prepared.crop_x == 416
    assert prepared.crop_y == 224
    assert prepared.image.shape == (576, 988, 3)
    np.testing.assert_array_equal(prepared.image, resized[224:800, 416:1404])


@pytest.mark.parametrize("shape", [(479, 641, 3), (641, 479, 3), (10, 2000, 3)])
def test_prepare_detector_input_keeps_symmetric_grid_aligned_crops(
    shape: tuple[int, int, int],
) -> None:
    frame = np.full(shape, 127, dtype=np.uint8)

    prepared = perception._prepare_detector_input(frame, 32)

    assert prepared.crop_x >= 0 and prepared.crop_x % 32 == 0
    assert prepared.crop_y >= 0 and prepared.crop_y % 32 == 0
    assert prepared.image.shape[0] > 0 and prepared.image.shape[1] > 0
    for border, crop in (
        (prepared.width_border, prepared.crop_x),
        (prepared.height_border, prepared.crop_y),
    ):
        retained_margin = border * prepared.scale - crop
        if crop:
            assert 32 <= retained_margin < 64


@pytest.mark.parametrize(
    ("frame", "halo", "message"),
    [
        (np.empty((0, 2, 3), dtype=np.uint8), 32, "non-empty"),
        (np.empty((2, 2), dtype=np.uint8), 32, "three-dimensional"),
        (np.empty((2, 2, 3), dtype=np.uint8), -32, "non-negative"),
        (np.empty((2, 2, 3), dtype=np.uint8), 16, "multiple of 32"),
    ],
)
def test_prepare_detector_input_rejects_invalid_geometry(
    frame: np.ndarray, halo: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        perception._prepare_detector_input(frame, halo)


def test_model_load_seconds_exposes_timing_without_mutable_global_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perception, "_MODEL_LOAD_SECONDS_TOTAL", 1.25)

    assert perception.model_load_seconds() == 1.25


def test_model_load_seconds_combines_detector_and_facenet_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perception, "_MODEL_LOAD_SECONDS_TOTAL", 1.25)
    monkeypatch.setattr(perception, "_DETECTOR_MODEL_LOAD_SECONDS_TOTAL", 0.75)

    assert perception.facenet_model_load_seconds() == 1.25
    assert perception.detector_model_load_seconds() == 0.75
    assert perception.model_load_seconds() == 2.0


def test_build_detector_model_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    models = [object()]
    calls: list[str] = []

    class FakeRetinaFace:
        @staticmethod
        def build_model() -> object:
            calls.append("build")
            return models[0]

    monkeypatch.setattr(perception, "_get_retinaface", lambda: FakeRetinaFace)

    first = perception.build_detector_model()
    second = perception.build_detector_model()

    assert first is models[0]
    assert second is models[0]
    assert calls == ["build"]


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


def _install_direct_detection_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepared: object,
    response: object,
    detector_calls: list[dict],
    alignment_calls: list[dict],
) -> object:
    resident_model = object()

    class FakeRegion:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class FakeDetection:
        FacialAreaRegion = FakeRegion

        @staticmethod
        def extract_face(**kwargs: object) -> object:
            alignment_calls.append(kwargs)
            region = kwargs["facial_area"]
            marker = max(1, int(getattr(region, "w")))
            return SimpleNamespace(
                img=np.full((4, 5, 3), marker % 255, dtype=np.uint8),
                facial_area=region,
            )

    class FakeRetinaFace:
        @staticmethod
        def detect_faces(image: np.ndarray, **kwargs: object) -> object:
            detector_calls.append({"image": image, **kwargs})
            return response

    monkeypatch.setattr(perception, "_prepare_detector_input", lambda frame, halo: prepared)
    monkeypatch.setattr(perception, "build_detector_model", lambda: resident_model)
    monkeypatch.setattr(perception, "_get_retinaface", lambda: FakeRetinaFace)
    monkeypatch.setattr(perception, "_get_deepface_detection", lambda: FakeDetection)
    return resident_model


def test_detect_and_align_maps_coordinates_and_uses_original_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_image = np.full((40, 60, 3), 9, dtype=np.uint8)
    prepared = perception._DetectorInput(
        image=prepared_image,
        scale=0.5,
        crop_x=32,
        crop_y=64,
        width_border=100,
        height_border=50,
    )
    response = {
        "face_1": {
            "score": 0.947,
            "facial_area": [10, 20, 50, 80],
            "landmarks": {
                "left_eye": [40, 40],
                "right_eye": [20, 40],
                "nose": [30, 50],
                "mouth_left": [40, 65],
                "mouth_right": [20, 65],
            },
        }
    }
    detector_calls: list[dict] = []
    alignment_calls: list[dict] = []
    resident_model = _install_direct_detection_fakes(
        monkeypatch,
        prepared=prepared,
        response=response,
        detector_calls=detector_calls,
        alignment_calls=alignment_calls,
    )
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    aligned = perception._detect_and_align_faces(frame, make_config())

    assert len(detector_calls) == 1
    assert detector_calls[0]["image"] is prepared_image
    assert detector_calls[0]["model"] is resident_model
    assert detector_calls[0]["threshold"] == 0.9
    assert detector_calls[0]["allow_upscaling"] is False
    call = alignment_calls[0]
    assert call["img"] is frame
    assert call["align"] is True
    assert call["expand_percentage"] == 0
    assert call["width_border"] == 0
    assert call["height_border"] == 0
    assert call["detector_backend"] == "retinaface"
    region = call["facial_area"]
    assert (region.x, region.y, region.w, region.h) == (-16, 118, 80, 120)
    assert region.left_eye == (44, 158)
    assert region.right_eye == (4, 158)
    assert region.nose == (24, 178)
    assert region.mouth_left == (44, 208)
    assert region.mouth_right == (4, 208)
    assert aligned[0].box == (0, 118, 80, 120)
    assert aligned[0].det_conf == 0.95


def test_detect_and_align_clips_edge_box_and_drops_invalid_landmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = perception._DetectorInput(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        scale=1.0,
        crop_x=0,
        crop_y=0,
        width_border=0,
        height_border=0,
    )
    response = {
        "face_1": {
            "score": 0.991,
            "facial_area": [-5, -4, 120, 110],
            "landmarks": {
                "left_eye": [10, 10],
                "right_eye": [105, 10],
                "nose": [50, 50],
                "mouth_left": [-1, 80],
                "mouth_right": [70, 101],
            },
        }
    }
    detector_calls: list[dict] = []
    alignment_calls: list[dict] = []
    _install_direct_detection_fakes(
        monkeypatch,
        prepared=prepared,
        response=response,
        detector_calls=detector_calls,
        alignment_calls=alignment_calls,
    )

    aligned = perception._detect_and_align_faces(
        np.zeros((100, 100, 3), dtype=np.uint8), make_config()
    )

    region = alignment_calls[0]["facial_area"]
    assert (region.x, region.y, region.w, region.h) == (-5, -4, 125, 114)
    assert aligned[0].box == (0, 0, 99, 99)
    assert aligned[0].landmarks == {"left_eye": (10, 10), "nose": (50, 50)}


def test_detect_and_align_publishes_region_returned_by_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = perception._DetectorInput(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        scale=1.0,
        crop_x=0,
        crop_y=0,
        width_border=0,
        height_border=0,
    )
    response = {
        "face_1": {
            "score": 0.95,
            "facial_area": [10, 10, 20, 20],
            "landmarks": {
                "left_eye": [17, 14],
                "right_eye": [13, 14],
                "nose": [15, 16],
                "mouth_left": [17, 18],
                "mouth_right": [13, 18],
            },
        }
    }
    detector_calls: list[dict] = []
    alignment_calls: list[dict] = []
    _install_direct_detection_fakes(
        monkeypatch,
        prepared=prepared,
        response=response,
        detector_calls=detector_calls,
        alignment_calls=alignment_calls,
    )

    class ExpandedDetection:
        class FacialAreaRegion:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        @staticmethod
        def extract_face(**kwargs: object) -> object:
            original = kwargs["facial_area"]
            expanded = ExpandedDetection.FacialAreaRegion(
                x=8,
                y=8,
                w=14,
                h=14,
                confidence=original.confidence,
                left_eye=original.left_eye,
                right_eye=original.right_eye,
                nose=original.nose,
                mouth_left=original.mouth_left,
                mouth_right=original.mouth_right,
            )
            return SimpleNamespace(
                img=np.ones((6, 6, 3), dtype=np.uint8), facial_area=expanded
            )

    monkeypatch.setattr(perception, "_get_deepface_detection", lambda: ExpandedDetection)

    aligned = perception._detect_and_align_faces(
        np.zeros((100, 100, 3), dtype=np.uint8),
        make_config(expand_percentage=40),
    )

    assert aligned[0].box == (8, 8, 14, 14)


def test_detect_and_align_applies_largest_face_cap_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = perception._DetectorInput(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        scale=1.0,
        crop_x=0,
        crop_y=0,
        width_border=0,
        height_border=0,
    )

    def detection(x: int, size: int) -> dict:
        return {
            "score": 0.95,
            "facial_area": [x, 10, x + size, 10 + size],
            "landmarks": {
                "left_eye": [x + 7, 17],
                "right_eye": [x + 3, 17],
                "nose": [x + 5, 20],
                "mouth_left": [x + 7, 24],
                "mouth_right": [x + 3, 24],
            },
        }

    response = {
        "small": detection(1, 10),
        "large_first": detection(20, 20),
        "large_second": detection(50, 20),
    }
    detector_calls: list[dict] = []
    alignment_calls: list[dict] = []
    _install_direct_detection_fakes(
        monkeypatch,
        prepared=prepared,
        response=response,
        detector_calls=detector_calls,
        alignment_calls=alignment_calls,
    )

    aligned = perception._detect_and_align_faces(
        np.zeros((100, 100, 3), dtype=np.uint8), make_config(max_faces=2)
    )

    assert [face.box for face in aligned] == [(20, 10, 20, 20), (50, 10, 20, 20)]
    assert [call["facial_area"].x for call in alignment_calls] == [20, 50]


def test_detect_and_align_empty_detector_result_skips_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = perception._DetectorInput(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        scale=1.0,
        crop_x=0,
        crop_y=0,
        width_border=0,
        height_border=0,
    )
    detector_calls: list[dict] = []
    alignment_calls: list[dict] = []
    _install_direct_detection_fakes(
        monkeypatch,
        prepared=prepared,
        response={},
        detector_calls=detector_calls,
        alignment_calls=alignment_calls,
    )

    assert perception._detect_and_align_faces(
        np.zeros((10, 10, 3), dtype=np.uint8), make_config()
    ) == []
    assert alignment_calls == []


def test_embed_faces_batches_aligned_crops_and_reassembles_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    represent_calls: list[dict] = []
    crop_a = np.full((4, 5, 3), 1, dtype=np.uint8)
    crop_b = np.full((4, 5, 3), 2, dtype=np.uint8)
    crop_c = np.full((4, 5, 3), 3, dtype=np.uint8)
    frame_a = np.zeros((12, 10, 3), dtype=np.uint8)
    frame_b = np.ones((12, 10, 3), dtype=np.uint8)
    aligned_by_frame = {
        id(frame_a): [
            perception._AlignedFace((1, 2, 3, 4), {"nose": (2, 3)}, 0.91, crop_a),
            perception._AlignedFace((5, 6, 7, 8), {}, 0.92, crop_b),
        ],
        id(frame_b): [
            perception._AlignedFace((9, 10, 11, 1), {"left_eye": (10, 10)}, 0.93, crop_c)
        ],
    }

    class FakeDeepFace:
        @staticmethod
        def build_model(model_name: str) -> object:
            return object()

        @staticmethod
        def represent(**kwargs: object) -> list[list[dict]]:
            represent_calls.append(kwargs)
            return [
                [result(embedding=unit_embedding(1))],
                [result(embedding=unit_embedding(2))],
                [result(embedding=unit_embedding(3))],
            ]

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)
    monkeypatch.setattr(
        perception,
        "_detect_and_align_faces",
        lambda frame, cfg: aligned_by_frame[id(frame)],
    )

    faces_by_frame = embed_faces(
        [frame_a, frame_b], make_config(max_faces=5, normalization="Facenet2018")
    )

    assert len(represent_calls) == 1
    call = represent_calls[0]
    assert call["img_path"] == [crop_a, crop_b, crop_c]
    assert call["model_name"] == "Facenet512"
    assert call["detector_backend"] == "skip"
    assert call["enforce_detection"] is False
    assert call["align"] is True
    assert call["normalization"] == "Facenet2018"
    assert call["max_faces"] is None
    assert call["l2_normalize"] is True
    assert call["expand_percentage"] == 0
    assert [[face.box for face in faces] for faces in faces_by_frame] == [
        [(1, 2, 3, 4), (5, 6, 7, 8)],
        [(9, 10, 11, 1)],
    ]
    assert [
        int(np.argmax(face.embedding)) for faces in faces_by_frame for face in faces
    ] == [1, 2, 3]
    assert faces_by_frame[0][0].landmarks == {"nose": (2, 3)}
    assert faces_by_frame[1][0].det_conf == 0.93


def _aligned_face(index: int) -> perception._AlignedFace:
    return perception._AlignedFace(
        box=(index, index + 1, 3, 4),
        landmarks={"nose": (index + 1, index + 2)},
        det_conf=0.9 + index / 100,
        image=np.full((4, 5, 3), index + 1, dtype=np.uint8),
    )


def test_embed_faces_normalizes_flat_single_crop_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(lambda **kwargs: [result(embedding=unit_embedding(4))])

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)
    monkeypatch.setattr(
        perception, "_detect_and_align_faces", lambda frame, cfg: [_aligned_face(0)]
    )

    faces_by_frame = embed_faces([np.zeros((12, 10, 3), dtype=np.uint8)], make_config())

    assert len(faces_by_frame) == 1
    assert len(faces_by_frame[0]) == 1
    assert int(np.argmax(faces_by_frame[0][0].embedding)) == 4


def test_embed_faces_preserves_mixed_zero_one_many_frame_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(3)]
    aligned_by_frame = {
        id(frames[0]): [],
        id(frames[1]): [_aligned_face(1)],
        id(frames[2]): [_aligned_face(2), _aligned_face(3)],
    }

    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(
            lambda **kwargs: [
                [result(embedding=unit_embedding(1))],
                [result(embedding=unit_embedding(2))],
                [result(embedding=unit_embedding(3))],
            ]
        )

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)
    monkeypatch.setattr(
        perception,
        "_detect_and_align_faces",
        lambda frame, cfg: aligned_by_frame[id(frame)],
    )

    faces_by_frame = embed_faces(frames, make_config())

    assert [len(faces) for faces in faces_by_frame] == [0, 1, 2]
    assert [
        int(np.argmax(face.embedding)) for faces in faces_by_frame for face in faces
    ] == [1, 2, 3]


def test_embed_faces_all_empty_skips_facenet_build_and_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        perception, "_detect_and_align_faces", lambda frame, cfg: []
    )
    monkeypatch.setattr(
        perception,
        "build_models",
        lambda cfg: pytest.fail("Facenet512 must not build for an all-empty batch"),
    )

    assert embed_faces(
        [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)], make_config()
    ) == [[], []]


def test_embed_faces_rejects_embedding_result_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(lambda **kwargs: [[result()]])

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)
    monkeypatch.setattr(
        perception,
        "_detect_and_align_faces",
        lambda frame, cfg: [_aligned_face(0), _aligned_face(1)],
    )

    with pytest.raises(ValueError, match="result count"):
        embed_faces([np.zeros((8, 8, 3), dtype=np.uint8)], make_config())


def test_embed_faces_rejects_empty_per_crop_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDeepFace:
        build_model = staticmethod(lambda model_name: object())
        represent = staticmethod(lambda **kwargs: [[]])

    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)
    monkeypatch.setattr(
        perception, "_detect_and_align_faces", lambda frame, cfg: [_aligned_face(0)]
    )

    with pytest.raises(ValueError, match="exactly one embedding"):
        embed_faces([np.zeros((8, 8, 3), dtype=np.uint8)], make_config())


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
    monkeypatch.setattr(
        perception, "_detect_and_align_faces", lambda frame, cfg: [_aligned_face(0)]
    )

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
    capture = cv2.VideoCapture("data/video-source/nimbus.mp4")
    assert capture.isOpened()
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
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
    assert cv2.imwrite(str(output_path), rendered)
    assert output_path.is_file()
    print(
        f"model_load_seconds={model_seconds:.3f} "
        f"perception_seconds={perception_seconds:.3f} "
        f"faces={len(faces_by_frame[0])} output={output_path}"
    )
