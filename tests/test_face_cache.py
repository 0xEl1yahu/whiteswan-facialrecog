from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import label_video
from face_labeller import perception
from face_labeller.cache import (
    FaceCache,
    face_cache_key,
)
from face_labeller.config import Config
from face_labeller.contracts import Face
from face_labeller.perception import process_batch_with_fallback
from face_labeller.video import selected_frame_indices

gallery_cache_key = label_video.gallery_cache_key


VERSIONS = {
    "deepface": "0.0.101",
    "retinaface": "0.0.18",
    "tensorflow": "2.21.0",
    "opencv": "5.0.0.93",
}


def make_config(tmp_path: Path, **overrides: object) -> Config:
    values: dict[str, object] = {
        "input_path": tmp_path / "input.mp4",
        "output_path": tmp_path / "output.mp4",
        "ref_dir": tmp_path / "references",
        "cache_dir": tmp_path / "cache",
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def unit_embedding(index: int = 0) -> np.ndarray:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


def make_face(index: int = 0) -> Face:
    return Face(
        box=(index + 1, index + 2, 20, 30),
        landmarks={"left_eye": (index + 3, index + 4), "nose": (5, 6)},
        embedding=unit_embedding(index),
        det_conf=0.91 + index / 100,
    )


def write_video_bytes(path: Path, payload: bytes = b"video-one") -> None:
    path.write_bytes(payload)


def test_face_cache_key_contains_video_hash_and_every_upstream_input(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cfg = make_config(tmp_path)
    expected_payload = {
        "schema_version": 1,
        "video_sha256": hashlib.sha256(b"video-one").hexdigest(),
        "model_name": "Facenet512",
        "detector_backend": "retinaface",
        "normalization": "base",
        "align": True,
        "max_faces": None,
        "expand_percentage": 0,
        "l2_normalize": True,
        "versions": VERSIONS,
    }
    encoded = json.dumps(
        expected_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert face_cache_key(video_path, cfg, VERSIONS) == hashlib.sha256(
        encoded
    ).hexdigest()


def test_cache_keys_match_pre_modularization_golden_values(tmp_path: Path) -> None:
    video_path = tmp_path / "fixture.mp4"
    write_video_bytes(video_path, b"video-fixture")
    cfg = make_config(tmp_path)

    assert gallery_cache_key(cfg, VERSIONS) == (
        "799c07df4ebcc29fefc98c3feb18383d84f269e9f09ff4afbfe270233161f0fc"
    )
    assert face_cache_key(video_path, cfg, VERSIONS) == (
        "13fdefcc341fd7d2482b1c2b542ea0d0e4b24eee705f77ff2657f53a22fa9051"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "DifferentModel"),
        ("detector_backend", "different-detector"),
        ("normalization", "Facenet2018"),
        ("align", False),
        ("max_faces", 3),
        ("expand_percentage", 5),
    ],
)
def test_face_cache_key_changes_for_upstream_config(
    tmp_path: Path, field: str, value: object
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cfg = make_config(tmp_path)

    assert face_cache_key(video_path, replace(cfg, **{field: value}), VERSIONS) != (
        face_cache_key(video_path, cfg, VERSIONS)
    )


@pytest.mark.parametrize("version_name", sorted(VERSIONS))
def test_face_cache_key_changes_for_library_version(
    tmp_path: Path, version_name: str
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cfg = make_config(tmp_path)
    changed = {**VERSIONS, version_name: "changed"}

    assert face_cache_key(video_path, cfg, changed) != face_cache_key(
        video_path, cfg, VERSIONS
    )


def test_face_cache_key_changes_when_video_content_changes(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    cfg = make_config(tmp_path)
    write_video_bytes(video_path, b"first")
    first = face_cache_key(video_path, cfg, VERSIONS)
    write_video_bytes(video_path, b"second")

    assert face_cache_key(video_path, cfg, VERSIONS) != first


def test_face_cache_key_excludes_every_downstream_or_window_choice(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cfg = make_config(tmp_path)
    changed = replace(
        cfg,
        output_path=tmp_path / "different.mp4",
        ref_dir=tmp_path / "other-references",
        stride=3,
        batch_size=2,
        threshold=0.17,
        pin_strategy="mean",
        start_frame=7,
        max_frames=11,
        cache_dir=tmp_path / "different-cache",
        smooth=True,
        iou_min=0.9,
        track_ttl=99,
        csv_path=tmp_path / "different.csv",
        debug_crops=True,
        debug_landmarks=True,
    )

    assert face_cache_key(video_path, changed, VERSIONS) == face_cache_key(
        video_path, cfg, VERSIONS
    )


def test_face_cache_status_distinguishes_absent_ok_empty_and_failed(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cache = FaceCache(video_path, make_config(tmp_path, use_cache=False), VERSIONS)

    assert cache.status(10) == "absent"
    assert cache.get(10) is None
    cache.put_ok(10, [])
    cache.put_failed(11)

    assert cache.status(10) == "ok"
    assert cache.get(10) == []
    assert cache.status(11) == "failed"
    assert cache.get(11) is None
    assert cache.missing([9, 10, 11, 12]) == [9, 11, 12]


def test_face_cache_status_retries_failed_but_not_ok_empty_frames(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cache = FaceCache(video_path, make_config(tmp_path, use_cache=False), VERSIONS)
    cache.put_ok(0, [])
    cache.put_failed(1)

    assert cache.missing([0, 1]) == [1]
    cache.put_ok(1, [make_face()])
    assert cache.missing([0, 1]) == []


def test_face_cache_persists_complete_faces_and_statuses(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cfg = make_config(tmp_path)
    cache = FaceCache(video_path, cfg, VERSIONS)
    cache.put_ok(2, [make_face(0), make_face(1)])
    cache.put_ok(3, [])
    cache.put_failed(4)
    cache.flush()

    loaded = FaceCache(video_path, cfg, VERSIONS)

    assert loaded.status(2) == "ok"
    assert loaded.status(3) == "ok"
    assert loaded.status(4) == "failed"
    assert loaded.missing([2, 3, 4, 5]) == [4, 5]
    faces = loaded.get(2)
    assert faces is not None and len(faces) == 2
    assert faces[0].box == (1, 2, 20, 30)
    assert faces[0].landmarks == {"left_eye": (3, 4), "nose": (5, 6)}
    assert faces[0].det_conf == pytest.approx(0.91)
    np.testing.assert_array_equal(faces[1].embedding, unit_embedding(1))


@pytest.mark.parametrize("damage", ["corrupt", "metadata"])
def test_face_cache_corrupt_or_mismatched_metadata_is_clean_miss(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    damage: str,
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cfg = make_config(tmp_path)
    cache = FaceCache(video_path, cfg, VERSIONS)
    cache.put_ok(2, [make_face()])
    cache.flush()

    if damage == "corrupt":
        cache.path.write_bytes(b"broken")
    else:
        with np.load(cache.path, allow_pickle=False) as stored:
            payload = {name: stored[name].copy() for name in stored.files}
        metadata = json.loads(str(payload["meta_json"].item()))
        metadata["normalization"] = "wrong"
        payload["meta_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        with cache.path.open("wb") as target:
            np.savez_compressed(target, **payload)

    loaded = FaceCache(video_path, cfg, VERSIONS)

    assert loaded.status(2) == "absent"
    assert "WARNING" in capsys.readouterr().out


def test_face_cache_flush_is_atomic_and_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cfg = make_config(tmp_path)
    cache = FaceCache(video_path, cfg, VERSIONS)
    cache.put_ok(0, [make_face()])

    cache.flush()

    assert cache.path.is_file()
    assert list(cfg.cache_dir.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("start", "stop", "stride", "expected"),
    [
        (0, 10, 3, [0, 3, 6, 9]),
        (5, 10, 3, [5, 8]),
        (7, 8, 99, [7]),
        (8, 8, 1, []),
    ],
)
def test_selected_frame_indices_are_relative_to_window_start(
    start: int, stop: int, stride: int, expected: list[int]
) -> None:
    assert selected_frame_indices(start, stop, stride) == expected


def test_face_cache_reuse_stride_three_then_stride_one_requests_only_gaps(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cache = FaceCache(video_path, make_config(tmp_path, use_cache=False), VERSIONS)
    stride_three = selected_frame_indices(2, 11, 3)
    for frame_idx in stride_three:
        cache.put_ok(frame_idx, [])

    assert stride_three == [2, 5, 8]
    assert cache.missing(selected_frame_indices(2, 11, 1)) == [3, 4, 6, 7, 9, 10]
    for frame_idx in cache.missing(selected_frame_indices(2, 11, 1)):
        cache.put_ok(frame_idx, [])
    assert cache.missing(selected_frame_indices(2, 11, 1)) == []


def test_face_cache_cache_dir_redirects_face_cache(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    redirected = tmp_path / "redirected"

    cache = FaceCache(
        video_path, make_config(tmp_path, cache_dir=redirected), VERSIONS
    )

    assert cache.path.parent == redirected


def test_face_cache_no_cache_bypasses_existing_data_and_writes(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    enabled_cfg = make_config(tmp_path)
    stored = FaceCache(video_path, enabled_cfg, VERSIONS)
    stored.put_ok(3, [make_face()])
    stored.flush()
    before = stored.path.read_bytes()

    disabled = FaceCache(
        video_path, replace(enabled_cfg, use_cache=False), VERSIONS
    )
    disabled.put_ok(4, [make_face(1)])
    disabled.flush()

    assert disabled.status(3) == "absent"
    assert stored.path.read_bytes() == before


def test_process_batch_with_fallback_preserves_successful_batch_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = make_face(0)
    second = make_face(1)

    def fake_embed(frames: list[np.ndarray], cfg: Config) -> list[list[Face]]:
        assert len(frames) == 2
        return [[first], [second]]

    monkeypatch.setattr(perception, "embed_faces", fake_embed)
    indexed_frames = [
        (10, np.zeros((4, 4, 3), dtype=np.uint8)),
        (12, np.ones((4, 4, 3), dtype=np.uint8)),
    ]

    results = process_batch_with_fallback(indexed_frames, make_config(tmp_path))

    assert results == {10: ("ok", [first]), 12: ("ok", [second])}


def test_process_batch_with_fallback_retries_individually_and_isolates_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recovered = make_face(2)

    def fake_embed(frames: list[np.ndarray], cfg: Config) -> list[list[Face]]:
        if len(frames) > 1:
            raise RuntimeError("batch failed")
        marker = int(frames[0][0, 0, 0])
        if marker == 2:
            raise RuntimeError("frame failed")
        return [[recovered] if marker == 1 else []]

    monkeypatch.setattr(perception, "embed_faces", fake_embed)
    indexed_frames = [
        (20, np.zeros((4, 4, 3), dtype=np.uint8)),
        (21, np.ones((4, 4, 3), dtype=np.uint8)),
        (22, np.full((4, 4, 3), 2, dtype=np.uint8)),
    ]

    results = process_batch_with_fallback(indexed_frames, make_config(tmp_path))

    assert results == {
        20: ("ok", []),
        21: ("ok", [recovered]),
        22: ("failed", []),
    }


def test_failed_batch_frame_is_requested_on_next_cache_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "input.mp4"
    write_video_bytes(video_path)
    cfg = make_config(tmp_path)
    cache = FaceCache(video_path, cfg, VERSIONS)
    cache.put_ok(30, [])
    cache.put_failed(31)
    cache.flush()

    reloaded = FaceCache(video_path, cfg, VERSIONS)
    assert reloaded.missing([30, 31]) == [31]

    monkeypatch.setattr(
        perception, "embed_faces", lambda frames, config: [[make_face()]]
    )
    recovered = process_batch_with_fallback(
        [(31, np.zeros((4, 4, 3), dtype=np.uint8))], cfg
    )
    reloaded.put_ok(31, recovered[31][1])

    assert reloaded.missing([30, 31]) == []
