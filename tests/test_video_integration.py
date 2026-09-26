from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import label_video
from label_video import Config, Face, FaceCache, RunSummary, main, process_video


def make_config(tmp_path: Path, **overrides: object) -> Config:
    values: dict[str, object] = {
        "input_path": tmp_path / "input.mp4",
        "output_path": tmp_path / "output.mp4",
        "ref_dir": tmp_path / "references",
        "cache_dir": tmp_path / "cache",
        "batch_size": 8,
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def make_face(x: int = 4) -> Face:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return Face(
        box=(x, 5, 12, 14),
        landmarks={},
        embedding=embedding,
        det_conf=0.95,
    )


def write_test_video(
    path: Path,
    *,
    frame_count: int = 10,
    fps: float = 12.0,
    size: tuple[int, int] = (48, 32),
) -> list[np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    assert writer.isOpened()
    frames: list[np.ndarray] = []
    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), 20 + index * 10, dtype=np.uint8)
        cv2.circle(frame, (30, 16), 3, (0, 0, 255), cv2.FILLED)
        writer.write(frame)
        frames.append(frame)
    writer.release()
    return frames


def read_video(path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps, size


def install_fast_perception(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    calls: list[list[int]] = []

    def fake_embed(frames: list[np.ndarray], cfg: Config) -> list[list[Face]]:
        calls.append([int(round(float(frame.mean()))) for frame in frames])
        return [[make_face()] for _ in frames]

    monkeypatch.setattr(label_video, "embed_faces", fake_embed)
    return calls


def test_process_video_rejects_unreadable_input(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)

    with pytest.raises(ValueError, match="cannot open input video"):
        process_video(cfg)


@pytest.mark.parametrize(
    ("fps", "width", "height", "message"),
    [
        (0.0, 48, 32, "FPS"),
        (12.0, 0, 32, "size"),
        (12.0, 48, 0, "size"),
    ],
)
def test_process_video_rejects_invalid_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fps: float,
    width: int,
    height: int,
    message: str,
) -> None:
    cfg = make_config(tmp_path)
    cfg.input_path.write_bytes(b"placeholder")

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return {
                cv2.CAP_PROP_FPS: fps,
                cv2.CAP_PROP_FRAME_WIDTH: width,
                cv2.CAP_PROP_FRAME_HEIGHT: height,
                cv2.CAP_PROP_FRAME_COUNT: 10,
            }.get(prop, 0.0)

        def release(self) -> None:
            pass

    monkeypatch.setattr(label_video.cv2, "VideoCapture", lambda path: FakeCapture())

    with pytest.raises(ValueError, match=message):
        process_video(cfg)


def test_process_video_rejects_writer_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path)
    write_test_video(cfg.input_path)

    class ClosedWriter:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr(
        label_video.cv2, "VideoWriter", lambda *args, **kwargs: ClosedWriter()
    )

    with pytest.raises(ValueError, match="cannot open output video"):
        process_video(cfg)
    assert not cfg.output_path.exists()


def test_process_video_writes_same_count_fps_size_and_unknown_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path, use_cache=False)
    write_test_video(cfg.input_path)
    install_fast_perception(monkeypatch)

    summary = process_video(cfg)
    input_frames, input_fps, input_size = read_video(cfg.input_path)
    output_frames, output_fps, output_size = read_video(cfg.output_path)

    assert isinstance(summary, RunSummary)
    assert summary.processed_frames == 10
    assert summary.written_frames == 10
    assert summary.faces == 10
    assert summary.label_distribution == {"Unknown": 10}
    assert len(output_frames) == len(input_frames) == 10
    assert output_fps == pytest.approx(input_fps, rel=0.01)
    assert output_size == input_size == (48, 32)
    assert all(
        np.mean(cv2.absdiff(source, rendered)) > 1.0
        for source, rendered in zip(input_frames, output_frames, strict=True)
    )


def test_process_video_stride_reuses_boxes_and_cache_fills_only_selected_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path, stride=3)
    write_test_video(cfg.input_path)
    calls = install_fast_perception(monkeypatch)

    summary = process_video(cfg)

    assert summary.processed_frames == 4
    assert summary.written_frames == 10
    assert summary.cache_hits == 0
    assert summary.cache_misses == 4
    assert sum(len(call) for call in calls) == 4
    cache = FaceCache(cfg.input_path, cfg)
    assert [cache.status(index) for index in range(10)] == [
        "ok",
        "absent",
        "absent",
        "ok",
        "absent",
        "absent",
        "ok",
        "absent",
        "absent",
        "ok",
    ]
    output_frames, _, _ = read_video(cfg.output_path)
    assert len(output_frames) == 10


def test_process_video_warm_replay_and_stride_one_compute_only_missing_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stride_three = make_config(tmp_path, stride=3)
    write_test_video(stride_three.input_path)
    first_calls = install_fast_perception(monkeypatch)
    first = process_video(stride_three)
    assert first.cache_misses == 4
    assert sum(len(call) for call in first_calls) == 4

    stride_one = replace(
        stride_three, output_path=tmp_path / "stride-one.mp4", stride=1
    )
    second_calls = install_fast_perception(monkeypatch)
    second = process_video(stride_one)
    assert second.cache_hits == 4
    assert second.cache_misses == 6
    assert sum(len(call) for call in second_calls) == 6

    monkeypatch.setattr(
        label_video,
        "embed_faces",
        lambda frames, cfg: (_ for _ in ()).throw(
            AssertionError("warm replay called perception")
        ),
    )
    warm = process_video(replace(stride_one, output_path=tmp_path / "warm.mp4"))
    assert warm.cache_hits == 10
    assert warm.cache_misses == 0
    assert warm.written_frames == 10


def test_process_video_continues_and_writes_every_frame_after_one_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path, use_cache=False, batch_size=4)
    write_test_video(cfg.input_path)
    single_call_count = 0

    def flaky_embed(frames: list[np.ndarray], config: Config) -> list[list[Face]]:
        nonlocal single_call_count
        if len(frames) > 1:
            raise RuntimeError("batch failure")
        single_call_count += 1
        if single_call_count == 2:
            raise RuntimeError("one bad frame")
        return [[make_face()]]

    monkeypatch.setattr(label_video, "embed_faces", flaky_embed)

    summary = process_video(cfg)

    output_frames, _, _ = read_video(cfg.output_path)
    assert summary.failures == 1
    assert summary.processed_frames == 10
    assert summary.written_frames == 10
    assert len(output_frames) == 10


def test_main_runs_video_pipeline_and_reports_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[Config] = []
    summary = RunSummary(
        elapsed_seconds=1.25,
        model_seconds=0.5,
        perception_seconds=1.0,
        processed_frames=3,
        written_frames=10,
        faces=4,
        cache_hits=0,
        cache_misses=3,
        failures=0,
        processing_fps=3.0,
        fps=12.0,
        width=48,
        height=32,
        label_distribution={"Unknown": 4},
    )

    def fake_process(cfg: Config, gallery: object = None) -> RunSummary:
        captured.append(cfg)
        return summary

    monkeypatch.setattr(label_video, "process_video", fake_process)

    exit_code = main(
        [
            "--input",
            str(tmp_path / "input.mp4"),
            "--output",
            str(tmp_path / "output.mp4"),
            "--ref-dir",
            str(tmp_path / "references"),
        ]
    )

    assert exit_code == 0
    assert captured[0].input_path == tmp_path / "input.mp4"
    output = capsys.readouterr().out
    assert "model=0.500s" in output
    assert "perception=1.000s" in output
    assert "processing_fps=3.00" in output
    assert "written=10" in output
    assert "faces=4" in output
    assert "cache_hits=0" in output


@pytest.mark.slow
def test_process_video_real_model_ten_frame_fixture(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, use_cache=False, batch_size=5)
    write_test_video(cfg.input_path)

    summary = process_video(cfg)

    output_frames, _, _ = read_video(cfg.output_path)
    assert summary.processed_frames == 10
    assert summary.written_frames == 10
    assert len(output_frames) == 10
