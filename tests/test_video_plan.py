from __future__ import annotations

from pathlib import Path

import cv2
import pytest

import label_video
from face_labeller import core, perception
from face_labeller.config import Config
from face_labeller.contracts import FramePlan, RunSummary, VideoMetadata
from face_labeller.video import build_frame_plan, inspect_video


class MetadataCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        frame_count: int = 10,
    ) -> None:
        self.opened = opened
        self.values = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
        }
        self.release_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        return float(self.values.get(prop, 0.0))

    def release(self) -> None:
        self.release_count += 1


def test_build_frame_plan_uses_absolute_indices_relative_to_window_start() -> None:
    metadata = VideoMetadata(Path("clip.mp4"), 30.0, 1920, 1080, 10)

    plan = build_frame_plan(
        metadata,
        start_frame=2,
        max_frames=6,
        stride=3,
    )

    assert plan == FramePlan(
        start_frame=2,
        stop_frame=8,
        written_frames=6,
        selected_indices=(2, 5),
    )


@pytest.mark.parametrize(
    ("start_frame", "max_frames", "stride", "expected"),
    [
        (0, None, 1, FramePlan(0, 10, 10, tuple(range(10)))),
        (7, 20, 2, FramePlan(7, 10, 3, (7, 9))),
        (9, 1, 3, FramePlan(9, 10, 1, (9,))),
    ],
)
def test_build_frame_plan_handles_full_clipped_and_one_frame_windows(
    start_frame: int,
    max_frames: int | None,
    stride: int,
    expected: FramePlan,
) -> None:
    metadata = VideoMetadata(Path("clip.mp4"), 30.0, 1920, 1080, 10)

    assert build_frame_plan(
        metadata,
        start_frame=start_frame,
        max_frames=max_frames,
        stride=stride,
    ) == expected


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start_frame": -1}, "start frame"),
        ({"start_frame": 10}, "outside"),
        ({"max_frames": 0}, "max frames"),
        ({"stride": 0}, "stride"),
    ],
)
def test_build_frame_plan_rejects_invalid_window_values(
    overrides: dict[str, int], message: str
) -> None:
    metadata = VideoMetadata(Path("clip.mp4"), 30.0, 1920, 1080, 10)
    values: dict[str, int | None] = {
        "start_frame": 0,
        "max_frames": None,
        "stride": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_frame_plan(metadata, **values)  # type: ignore[arg-type]


def test_inspect_video_returns_metadata_and_releases_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "clip.mp4"
    capture = MetadataCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)

    metadata = inspect_video(input_path)

    assert metadata == VideoMetadata(input_path, 30.0, 1920, 1080, 10)
    assert capture.release_count == 1


@pytest.mark.parametrize(
    ("capture", "message"),
    [
        (MetadataCapture(opened=False), "cannot open"),
        (MetadataCapture(fps=0.0), "FPS"),
        (MetadataCapture(width=0), "size"),
        (MetadataCapture(height=0), "size"),
        (MetadataCapture(frame_count=0), "frame count"),
    ],
)
def test_inspect_video_rejects_invalid_metadata_and_releases_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture: MetadataCapture,
    message: str,
) -> None:
    input_path = tmp_path / "clip.mp4"
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)

    with pytest.raises(ValueError, match=message):
        inspect_video(input_path)

    assert capture.release_count == 1


def _summary() -> RunSummary:
    return RunSummary(
        elapsed_seconds=0.0,
        model_seconds=0.0,
        perception_seconds=0.0,
        processed_frames=3,
        written_frames=7,
        faces=0,
        cache_hits=1,
        cache_misses=2,
        failures=0,
        processing_fps=0.0,
        fps=30.0,
        width=1920,
        height=1080,
        label_distribution={},
    )


def _main_args(tmp_path: Path, *extra: str) -> list[str]:
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"video")
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir(exist_ok=True)
    return [
        "--input",
        str(input_path),
        "--output",
        str(tmp_path / "out.mp4"),
        "--ref-dir",
        str(ref_dir),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--csv",
        str(tmp_path / "matches.csv"),
        *extra,
    ]


def test_main_invalid_video_stops_before_gallery_or_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = MetadataCapture(fps=0.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(
        core,
        "load_gallery",
        lambda *args: pytest.fail("invalid video reached gallery loading"),
    )
    monkeypatch.setattr(
        perception,
        "build_models",
        lambda *args: pytest.fail("invalid video reached model loading"),
    )

    assert label_video.main(_main_args(tmp_path)) == 1
    assert capture.release_count == 1


def test_main_out_of_range_window_stops_before_gallery_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "clip.mp4"
    monkeypatch.setattr(
        core,
        "inspect_video",
        lambda path: VideoMetadata(input_path, 30.0, 1920, 1080, 10),
    )
    monkeypatch.setattr(
        core,
        "load_gallery",
        lambda *args: pytest.fail("invalid window reached gallery loading"),
    )

    assert label_video.main(_main_args(tmp_path, "--start-frame", "10")) == 1


def test_main_reports_cached_and_pending_selected_frames_before_gallery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "clip.mp4"
    metadata = VideoMetadata(input_path, 30.0, 1920, 1080, 10)
    events: list[str] = []

    class PlannedCache:
        def missing(self, frame_indices: tuple[int, ...]) -> list[int]:
            assert frame_indices == (2, 5, 8)
            return [5, 8]

    cache = PlannedCache()
    monkeypatch.setattr(core, "inspect_video", lambda path: metadata)
    monkeypatch.setattr(
        core,
        "FaceCache",
        lambda path, cfg: (events.append("cache") or cache),
    )
    monkeypatch.setattr(
        core,
        "load_gallery",
        lambda ref_dir, cfg: (events.append("gallery") or object()),
    )

    def fake_process(
        cfg: Config,
        gallery: object,
        *,
        metadata: VideoMetadata,
        plan: FramePlan,
        face_cache: object,
    ) -> RunSummary:
        events.append("process")
        assert metadata == VideoMetadata(input_path, 30.0, 1920, 1080, 10)
        assert plan == FramePlan(2, 9, 7, (2, 5, 8))
        assert face_cache is cache
        return _summary()

    monkeypatch.setattr(core, "process_video", fake_process)

    assert label_video.main(
        _main_args(
            tmp_path,
            "--start-frame",
            "2",
            "--max-frames",
            "7",
            "--stride",
            "3",
        )
    ) == 0

    output = capsys.readouterr().out
    assert events == ["cache", "gallery", "process"]
    assert "window=2:9" in output
    assert "written=7" in output
    assert "selected=3" in output
    assert "cached=1" in output
    assert "to_infer=2" in output
