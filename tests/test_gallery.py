from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from face_labeller import gallery, perception
from face_labeller.config import Config
from face_labeller.contracts import (
    CHARACTER_NAMES,
    GalleryPhoto,
)
from face_labeller.gallery import (
    build_pins,
    gallery_cache_key,
    leave_one_out_report,
    load_gallery,
)


def make_config(tmp_path: Path, **overrides: object) -> Config:
    values: dict[str, object] = {
        "input_path": Path("input.mp4"),
        "output_path": Path("output.mp4"),
        "ref_dir": tmp_path / "references",
        "cache_dir": tmp_path / "cache",
        "use_cache": False,
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def unit_embedding(index: int) -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


def create_gallery_files(
    root: Path,
    *,
    names: tuple[str, ...] = CHARACTER_NAMES,
    filenames: tuple[str, ...] = ("02.JPG", "01.png"),
) -> None:
    for owner_index, owner in enumerate(names):
        owner_dir = root / owner
        owner_dir.mkdir(parents=True)
        for photo_index, filename in enumerate(filenames):
            (owner_dir / filename).write_bytes(bytes([owner_index, photo_index]))


def install_fake_deepface(
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_markers: set[int] | None = None,
    calls: list[dict] | None = None,
    model_calls: list[str] | None = None,
) -> None:
    invalid_markers = invalid_markers or set()

    def fake_imread(path: str) -> np.ndarray:
        payload = Path(path).read_bytes()
        marker = payload[0] * 10 + payload[1]
        return np.full((8, 8, 3), marker, dtype=np.uint8)

    class FakeDeepFace:
        @staticmethod
        def build_model(model_name: str) -> object:
            if model_calls is not None:
                model_calls.append(model_name)
            return object()

        @staticmethod
        def represent(**kwargs: object) -> list[dict]:
            if calls is not None:
                calls.append(kwargs)
            image = np.asarray(kwargs["img_path"])
            marker = int(image[0, 0, 0])
            if marker in invalid_markers:
                raise ValueError("no face detected")
            return [{"embedding": unit_embedding(marker).tolist()}]

    monkeypatch.setattr(gallery.cv2, "imread", fake_imread)
    monkeypatch.setattr(perception, "_get_deepface", lambda: FakeDeepFace)


@pytest.fixture(autouse=True)
def reset_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perception, "_MODEL_BUILT", False)
    monkeypatch.setattr(perception, "_DETECTOR_MODEL", None, raising=False)

    class FakeRetinaFace:
        build_model = staticmethod(lambda: object())

    monkeypatch.setattr(perception, "_get_retinaface", lambda: FakeRetinaFace, raising=False)


def test_gallery_validation_requires_all_five_character_folders(tmp_path: Path) -> None:
    create_gallery_files(tmp_path / "references", names=CHARACTER_NAMES[:-1])

    with pytest.raises(ValueError, match="missing required character folder.*Ron Weasley"):
        load_gallery(tmp_path / "references", make_config(tmp_path))


def test_gallery_validation_accepts_extensions_case_insensitively_in_sorted_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root, filenames=("z.JPEG", "A.PNG"))
    for owner in CHARACTER_NAMES:
        (root / owner / "ignored.gif").write_bytes(b"ignored")
    install_fake_deepface(monkeypatch)

    gallery = load_gallery(root, make_config(tmp_path))

    expected = [
        f"{owner}/{filename}"
        for owner in CHARACTER_NAMES
        for filename in ("A.PNG", "z.JPEG")
    ]
    assert gallery.meta["source_paths"] == expected
    assert gallery.names == sorted(CHARACTER_NAMES)


def test_gallery_validation_skips_no_face_image_and_reports_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root, filenames=("01.jpg", "02.jpg", "03.jpg"))
    install_fake_deepface(monkeypatch, invalid_markers={2, 12, 22, 32, 42})

    gallery = load_gallery(root, make_config(tmp_path))

    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "no face detected" in output
    assert "Harry Potter: found=3 used=2 skipped=1" in output
    assert gallery.meta["photo_count"] == 10


def test_gallery_validation_fails_when_skips_leave_fewer_than_two_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root)
    install_fake_deepface(monkeypatch, invalid_markers={1})

    with pytest.raises(ValueError, match="Harry Potter.*1 valid.*minimum is 2"):
        load_gallery(root, make_config(tmp_path))


def test_gallery_validation_embeds_with_explicit_gallery_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root)
    calls: list[dict] = []
    install_fake_deepface(monkeypatch, calls=calls)

    load_gallery(root, make_config(tmp_path, normalization="Facenet2018"))

    assert len(calls) == 10
    assert all(call["model_name"] == "Facenet512" for call in calls)
    assert all(call["detector_backend"] == "retinaface" for call in calls)
    assert all(call["enforce_detection"] is True for call in calls)
    assert all(call["align"] is True for call in calls)
    assert all(call["normalization"] == "Facenet2018" for call in calls)
    assert all(call["max_faces"] == 1 for call in calls)
    assert all(call["l2_normalize"] is True for call in calls)
    assert all(call["expand_percentage"] == 0 for call in calls)


def test_gallery_pins_mean_are_renormalized_and_sorted() -> None:
    records = [
        GalleryPhoto("Ron Weasley", "Ron Weasley/b.jpg", "b", unit_embedding(2)),
        GalleryPhoto("Harry Potter", "Harry Potter/a.jpg", "a", unit_embedding(0)),
        GalleryPhoto("Harry Potter", "Harry Potter/b.jpg", "b", unit_embedding(1)),
        GalleryPhoto("Ron Weasley", "Ron Weasley/a.jpg", "a", unit_embedding(2)),
    ]

    gallery = build_pins(records, "mean")

    assert gallery.pin_owner == ["Harry Potter", "Ron Weasley"]
    assert gallery.names == ["Harry Potter", "Ron Weasley"]
    assert gallery.pins.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(gallery.pins, axis=1), [1.0, 1.0])
    np.testing.assert_allclose(
        gallery.pins[0, :3], [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-6
    )


def test_gallery_pins_all_preserve_photo_ownership_and_deterministic_order() -> None:
    records = [
        GalleryPhoto("Ron Weasley", "Ron Weasley/z.jpg", "z", unit_embedding(3)),
        GalleryPhoto("Harry Potter", "Harry Potter/b.jpg", "b", unit_embedding(1)),
        GalleryPhoto("Harry Potter", "Harry Potter/a.jpg", "a", unit_embedding(0)),
    ]

    gallery = build_pins(records, "all")

    assert gallery.pin_owner == ["Harry Potter", "Harry Potter", "Ron Weasley"]
    assert gallery.names == ["Harry Potter", "Ron Weasley"]
    assert np.argmax(gallery.pins, axis=1).tolist() == [0, 1, 3]


def test_gallery_pins_strategies_use_identical_per_photo_inputs() -> None:
    records = [
        GalleryPhoto("Harry Potter", "Harry Potter/a.jpg", "ha", unit_embedding(0)),
        GalleryPhoto("Harry Potter", "Harry Potter/b.jpg", "hb", unit_embedding(1)),
        GalleryPhoto("Ron Weasley", "Ron Weasley/a.jpg", "ra", unit_embedding(2)),
        GalleryPhoto("Ron Weasley", "Ron Weasley/b.jpg", "rb", unit_embedding(3)),
    ]
    original = [replace(record, embedding=record.embedding.copy()) for record in records]

    mean_gallery = build_pins(records, "mean")
    all_gallery = build_pins(records, "all")

    assert mean_gallery.meta["source_paths"] == all_gallery.meta["source_paths"]
    assert mean_gallery.meta["file_hashes"] == all_gallery.meta["file_hashes"]
    for before, after in zip(original, records, strict=True):
        np.testing.assert_array_equal(after.embedding, before.embedding)


VERSIONS = {
    "deepface": "0.0.101",
    "retinaface": "0.0.18",
    "tensorflow": "2.21.0",
    "opencv": "5.0.0.93",
}


def test_gallery_cache_key_contains_every_upstream_configuration_input(
    tmp_path: Path,
) -> None:
    cfg = make_config(tmp_path, use_cache=True)
    expected_payload = {
        "schema_version": 1,
        "model_name": "Facenet512",
        "detector_backend": "retinaface",
        "normalization": "base",
        "align": True,
        "max_faces": 1,
        "expand_percentage": 0,
        "l2_normalize": True,
        "versions": VERSIONS,
    }
    encoded = json.dumps(
        expected_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert gallery_cache_key(cfg, VERSIONS) == hashlib.sha256(encoded).hexdigest()


def test_gallery_cache_key_excludes_all_downstream_matching_and_tracking_choices(
    tmp_path: Path,
) -> None:
    base = make_config(tmp_path, use_cache=True)
    expected = gallery_cache_key(base, VERSIONS)

    assert gallery_cache_key(replace(base, pin_strategy="all"), VERSIONS) == expected
    assert gallery_cache_key(replace(base, max_faces=7), VERSIONS) == expected
    assert gallery_cache_key(replace(base, threshold=0.12), VERSIONS) == expected
    assert gallery_cache_key(replace(base, smooth=True), VERSIONS) == expected
    assert gallery_cache_key(replace(base, iou_min=0.9), VERSIONS) == expected
    assert gallery_cache_key(replace(base, track_ttl=99), VERSIONS) == expected


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("model_name", "DifferentModel"),
        ("detector_backend", "different-detector"),
        ("normalization", "Facenet2018"),
        ("align", False),
        ("expand_percentage", 5),
    ],
)
def test_gallery_cache_key_changes_for_upstream_config(
    tmp_path: Path, change: str, value: object
) -> None:
    base = make_config(tmp_path, use_cache=True)

    assert gallery_cache_key(replace(base, **{change: value}), VERSIONS) != gallery_cache_key(
        base, VERSIONS
    )


@pytest.mark.parametrize("version_name", sorted(VERSIONS))
def test_gallery_cache_key_changes_for_each_library_version(
    tmp_path: Path, version_name: str
) -> None:
    changed_versions = {**VERSIONS, version_name: "changed"}
    cfg = make_config(tmp_path, use_cache=True)

    assert gallery_cache_key(cfg, changed_versions) != gallery_cache_key(cfg, VERSIONS)


def test_gallery_cache_cold_then_warm_avoids_all_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root)
    represent_calls: list[dict] = []
    model_calls: list[str] = []
    detector_calls: list[str] = []

    class FakeRetinaFace:
        @staticmethod
        def build_model() -> object:
            detector_calls.append("retinaface")
            return object()

    monkeypatch.setattr(perception, "_get_retinaface", lambda: FakeRetinaFace)
    install_fake_deepface(
        monkeypatch, calls=represent_calls, model_calls=model_calls
    )
    cfg = make_config(tmp_path, use_cache=True)

    cold = load_gallery(root, cfg)
    assert len(represent_calls) == 10
    assert model_calls == ["Facenet512"]
    assert detector_calls == ["retinaface"]
    assert len(list(cfg.cache_dir.glob("gallery_*.npz"))) == 1

    represent_calls.clear()
    model_calls.clear()
    monkeypatch.setattr(perception, "_MODEL_BUILT", False)
    monkeypatch.setattr(perception, "_DETECTOR_MODEL", None)
    warm = load_gallery(root, replace(cfg, pin_strategy="all"))

    assert represent_calls == []
    assert model_calls == []
    assert detector_calls == ["retinaface"]
    assert warm.meta["source_paths"] == cold.meta["source_paths"]
    assert warm.pins.shape == (10, 512)


def test_gallery_cache_adds_and_modifies_only_one_photo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root)
    calls: list[dict] = []
    install_fake_deepface(monkeypatch, calls=calls)
    cfg = make_config(tmp_path, use_cache=True)
    load_gallery(root, cfg)

    calls.clear()
    (root / "Harry Potter" / "03.jpg").write_bytes(bytes([0, 3]))
    load_gallery(root, cfg)
    assert len(calls) == 1

    calls.clear()
    (root / "Harry Potter" / "03.jpg").write_bytes(bytes([0, 3, 99]))
    load_gallery(root, cfg)
    assert len(calls) == 1


def test_gallery_cache_deletion_removes_only_that_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root, filenames=("01.jpg", "02.jpg", "03.jpg"))
    calls: list[dict] = []
    install_fake_deepface(monkeypatch, calls=calls)
    cfg = make_config(tmp_path, use_cache=True)
    load_gallery(root, cfg)

    calls.clear()
    (root / "Harry Potter" / "03.jpg").unlink()
    gallery = load_gallery(root, cfg)

    assert calls == []
    assert gallery.meta["photo_count"] == 14
    assert "Harry Potter/03.jpg" not in gallery.meta["source_paths"]


def test_gallery_cache_normalization_variants_coexist_and_are_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root)
    calls: list[dict] = []
    install_fake_deepface(monkeypatch, calls=calls)
    cfg = make_config(tmp_path, use_cache=True)

    load_gallery(root, cfg)
    assert len(calls) == 10
    calls.clear()
    load_gallery(root, replace(cfg, normalization="Facenet2018"))
    assert len(calls) == 10
    assert len(list(cfg.cache_dir.glob("gallery_*.npz"))) == 2
    calls.clear()
    load_gallery(root, cfg)
    assert calls == []


@pytest.mark.parametrize("damage", ["corrupt", "metadata"])
def test_gallery_cache_corrupt_or_mismatched_metadata_is_a_clean_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    root = tmp_path / "references"
    create_gallery_files(root)
    calls: list[dict] = []
    install_fake_deepface(monkeypatch, calls=calls)
    cfg = make_config(tmp_path, use_cache=True)
    load_gallery(root, cfg)
    cache_path = next(cfg.cache_dir.glob("gallery_*.npz"))

    if damage == "corrupt":
        cache_path.write_bytes(b"not an npz")
    else:
        with np.load(cache_path, allow_pickle=False) as data:
            payload = {name: data[name].copy() for name in data.files}
        metadata = json.loads(str(payload["meta_json"].item()))
        metadata["normalization"] = "wrong"
        payload["meta_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        with cache_path.open("wb") as target:
            np.savez_compressed(target, **payload)

    calls.clear()
    gallery = load_gallery(root, cfg)

    assert len(calls) == 10
    assert gallery.meta["photo_count"] == 10
    with np.load(cache_path, allow_pickle=False) as repaired:
        assert set(repaired.files) == {
            "meta_json",
            "owners",
            "source_paths",
            "file_hashes",
            "embeddings",
        }


def similar_embedding(primary: int, secondary: int) -> np.ndarray:
    embedding = unit_embedding(primary)
    embedding[secondary] = 0.1
    return (embedding / np.linalg.norm(embedding)).astype(np.float32)


def test_leave_one_out_report_tests_every_photo_under_both_strategies() -> None:
    records = [
        GalleryPhoto("Harry Potter", "Harry Potter/a.jpg", "ha", unit_embedding(0)),
        GalleryPhoto(
            "Harry Potter",
            "Harry Potter/b.jpg",
            "hb",
            similar_embedding(0, 1),
        ),
        GalleryPhoto("Ron Weasley", "Ron Weasley/a.jpg", "ra", unit_embedding(2)),
        GalleryPhoto(
            "Ron Weasley",
            "Ron Weasley/b.jpg",
            "rb",
            similar_embedding(2, 3),
        ),
    ]

    report = leave_one_out_report(records, threshold=0.30)

    assert set(report) == {"mean", "all"}
    for strategy in ("mean", "all"):
        strategy_report = report[strategy]
        assert strategy_report["aggregate"] == {
            "correct": 4,
            "unknown": 0,
            "wrong": 0,
        }
        assert len(strategy_report["photos"]) == 4
        for photo in strategy_report["photos"]:
            assert photo["source_path"] not in photo["candidate_source_paths"]
            assert photo["nearest_name"] == photo["owner"]
            assert photo["assigned_name"] == photo["owner"]
            assert 0 <= photo["distance"] < 0.30


def test_leave_one_out_report_records_unknown_assignments() -> None:
    records = [
        GalleryPhoto("Harry Potter", "Harry Potter/a.jpg", "ha", unit_embedding(0)),
        GalleryPhoto(
            "Harry Potter", "Harry Potter/b.jpg", "hb", similar_embedding(0, 1)
        ),
        GalleryPhoto("Ron Weasley", "Ron Weasley/a.jpg", "ra", unit_embedding(2)),
        GalleryPhoto(
            "Ron Weasley", "Ron Weasley/b.jpg", "rb", similar_embedding(2, 3)
        ),
    ]

    report = leave_one_out_report(records, threshold=0.0)

    for strategy in ("mean", "all"):
        assert report[strategy]["aggregate"] == {
            "correct": 0,
            "unknown": 4,
            "wrong": 0,
        }
        assert all(
            photo["assigned_name"] is None
            for photo in report[strategy]["photos"]
        )
