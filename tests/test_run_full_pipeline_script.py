from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_full_pipeline.sh"
CHARACTERS = (
    "Harry Potter",
    "Ron Weasley",
    "Hermione Granger",
    "Prof. McGonagall",
    "Prof. Severus Snape",
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_python(path: Path) -> None:
    _write_executable(
        path,
        """#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_PYTHON_LOG"
if [ "${1:-}" = "-c" ]; then
  exit 0
fi
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "venv" ]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python"
  exit 0
fi
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "gdown" ]; then
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "-O" ]; then
      shift
      mkdir -p "$(dirname "$1")"
      : > "$1"
      exit 0
    fi
    shift
  done
fi
if [ "${1:-}" = "label_video.py" ]; then
  shift
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --output|--csv)
        shift
        mkdir -p "$(dirname "$1")"
        : > "$1"
        ;;
    esac
    shift
  done
  exit 0
fi
exit 0
""",
    )


def _fake_media_tools(directory: Path) -> None:
    _write_executable(
        directory / "ffmpeg",
        """#!/bin/sh
set -eu
printf 'ffmpeg %s\n' "$*" >> "$FAKE_MEDIA_LOG"
if [ "${FAKE_FFMPEG_FAIL:-0}" = "1" ]; then
  exit 42
fi
output=""
for argument in "$@"; do
  output="$argument"
done
printf 'audio-preserved' > "$output"
""",
    )
    _write_executable(
        directory / "ffprobe",
        """#!/bin/sh
set -eu
printf 'ffprobe %s\n' "$*" >> "$FAKE_MEDIA_LOG"
case "$*" in
  *a:0*) printf 'audio\n' ;;
  *v:0*) printf 'video\n' ;;
  *) exit 2 ;;
esac
""",
    )


def _fresh_clone(
    tmp_path: Path, *, with_references: bool = True
) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
    (root / "requirements.txt").write_text("example==1.0\n", encoding="utf-8")
    (root / "label_video.py").write_text("# test entrypoint\n", encoding="utf-8")
    if with_references:
        for character in CHARACTERS:
            directory = root / "data" / "reference-images" / character
            directory.mkdir(parents=True)
            for index in range(2):
                (directory / f"reference-{index}.jpg").write_bytes(b"test")

    tools = tmp_path / "tools"
    tools.mkdir()
    python = tools / "python3.11"
    _fake_python(python)
    _fake_media_tools(tools)
    log = tmp_path / "python.log"
    return root, python, log


def test_fresh_clone_sets_up_downloads_video_and_runs_full_pipeline(
    tmp_path: Path,
) -> None:
    root, python, log = _fresh_clone(tmp_path)
    env = {
        **os.environ,
        "PYTHON_BIN": str(python),
        "FAKE_PYTHON_LOG": str(log),
        "FAKE_MEDIA_LOG": str(tmp_path / "media.log"),
        "PATH": f"{python.parent}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(root / "scripts" / SCRIPT.name)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (root / ".venv" / "bin" / "python").is_file()
    assert (root / "data" / "video-source" / "nimbus.mp4").is_file()
    assert (root / "output" / "nimbus_labelled.mp4").is_file()
    assert (root / "output" / "matches.csv").is_file()
    assert (root / "output" / "nimbus_labelled.mp4").read_bytes() == b"audio-preserved"
    calls = log.read_text(encoding="utf-8")
    assert f"-m venv {root / '.venv'}" in calls
    assert (
        f"-m pip install --disable-pip-version-check -r {root / 'requirements.txt'}"
        in calls
    )
    assert "-m gdown 1CM1IWUN59ZWml9MwgrvSHXz_9AirIiuU" in calls
    assert "label_video.py" in calls
    assert "--stride 1" in calls
    assert "--batch-size 8" in calls
    assert "--threshold 0.30" in calls
    assert "--pin-strategy all" in calls
    media_calls = (tmp_path / "media.log").read_text(encoding="utf-8")
    assert "-map 0:v:0 -map 1:a:0" in media_calls
    assert "-c:v copy -c:a copy -shortest" in media_calls
    assert "ffprobe" in media_calls
    assert (
        "Run report: "
        f"{root / 'docs/results/m4-gallery-perception-consistency.md'}"
        in result.stdout
    )


def test_missing_curated_references_stops_before_setup(tmp_path: Path) -> None:
    root, python, log = _fresh_clone(tmp_path, with_references=False)
    env = {
        **os.environ,
        "PYTHON_BIN": str(python),
        "FAKE_PYTHON_LOG": str(log),
        "FAKE_MEDIA_LOG": str(tmp_path / "media.log"),
        "PATH": f"{python.parent}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(root / "scripts" / SCRIPT.name)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "two curated reference images" in result.stderr
    assert not (root / ".venv").exists()
    assert not log.exists()


def test_rerun_reuses_existing_environment_and_source_video(tmp_path: Path) -> None:
    root, python, log = _fresh_clone(tmp_path)
    existing_python = root / ".venv" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    shutil.copy2(python, existing_python)
    source_video = root / "data" / "video-source" / "nimbus.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"existing-video")
    env = {
        **os.environ,
        "PYTHON_BIN": str(python),
        "FAKE_PYTHON_LOG": str(log),
        "FAKE_MEDIA_LOG": str(tmp_path / "media.log"),
        "PATH": f"{python.parent}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(root / "scripts" / SCRIPT.name)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert "-m venv" not in calls
    assert "-m gdown" not in calls
    assert source_video.read_bytes() == b"existing-video"
    assert "label_video.py" in calls


def test_failed_audio_remux_preserves_previous_video_and_csv(tmp_path: Path) -> None:
    root, python, log = _fresh_clone(tmp_path)
    output_video = root / "output" / "nimbus_labelled.mp4"
    output_csv = root / "output" / "matches.csv"
    output_video.parent.mkdir(parents=True)
    output_video.write_bytes(b"previous-video")
    output_csv.write_bytes(b"previous-csv")
    env = {
        **os.environ,
        "PYTHON_BIN": str(python),
        "FAKE_PYTHON_LOG": str(log),
        "FAKE_MEDIA_LOG": str(tmp_path / "media.log"),
        "FAKE_FFMPEG_FAIL": "1",
        "PATH": f"{python.parent}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(root / "scripts" / SCRIPT.name)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert output_video.read_bytes() == b"previous-video"
    assert output_csv.read_bytes() == b"previous-csv"
    assert list((root / "output").glob(".full-run.*")) == []
