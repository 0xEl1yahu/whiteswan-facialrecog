from __future__ import annotations

from pathlib import Path

import pytest

import label_video
from face_labeller import core
from face_labeller.config import Config
from face_labeller.contracts import FramePlan, RunSummary, VideoMetadata


def _config(tmp_path: Path) -> Config:
    return Config(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        ref_dir=tmp_path / "references",
        cache_dir=tmp_path / "cache",
    )


def _summary() -> RunSummary:
    return RunSummary(
        elapsed_seconds=1.0,
        model_seconds=0.1,
        perception_seconds=0.5,
        processed_frames=2,
        written_frames=2,
        faces=3,
        cache_hits=1,
        cache_misses=1,
        failures=0,
        processing_fps=2.0,
        fps=24.0,
        width=64,
        height=48,
        label_distribution={"Unknown": 3},
    )


def test_core_runs_preflight_before_gallery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    metadata = VideoMetadata(cfg.input_path, 24.0, 64, 48, 2)
    plan = FramePlan(0, 2, 2, (0, 1))
    summary = _summary()
    events: list[str] = []

    monkeypatch.setattr(core, "validate_paths", lambda config: events.append("validate"))
    monkeypatch.setattr(core, "inspect_video", lambda path: events.append("inspect") or metadata)
    monkeypatch.setattr(core, "build_frame_plan", lambda *args, **kwargs: events.append("plan") or plan)

    class Cache:
        def missing(self, indices: tuple[int, ...]) -> list[int]:
            return [1]

    monkeypatch.setattr(core, "FaceCache", lambda *args: events.append("cache") or Cache())
    monkeypatch.setattr(core, "load_gallery", lambda *args: events.append("gallery") or object())
    monkeypatch.setattr(
        core,
        "process_video",
        lambda *args, **kwargs: events.append("process") or summary,
    )
    monkeypatch.setattr(core.perception, "model_load_seconds", lambda: 0.0)
    monkeypatch.setattr(
        core,
        "print",
        lambda message: events.append(
            "preflight" if str(message).startswith("preflight ") else "complete"
        ),
        raising=False,
    )

    assert core.run(cfg) is summary
    assert events == [
        "validate",
        "inspect",
        "plan",
        "cache",
        "preflight",
        "gallery",
        "process",
        "complete",
    ]


def test_main_delegates_to_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[Config] = []
    monkeypatch.setattr(label_video, "run", lambda cfg: received.append(cfg) or _summary())

    assert label_video.main([
        "--input", str(tmp_path / "input.mp4"),
        "--output", str(tmp_path / "output.mp4"),
        "--ref-dir", str(tmp_path / "references"),
    ]) == 0
    assert len(received) == 1
    assert received[0].input_path == tmp_path / "input.mp4"


def test_main_maps_core_error_to_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_cfg: Config) -> RunSummary:
        raise RuntimeError("boom")

    monkeypatch.setattr(label_video, "run", fail)

    assert label_video.main([
        "--input", str(tmp_path / "input.mp4"),
        "--output", str(tmp_path / "output.mp4"),
        "--ref-dir", str(tmp_path / "references"),
    ]) == 1
    assert capsys.readouterr().err == "ERROR: boom\n"
