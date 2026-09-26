"""Match evidence logging and optional debug crops."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile

import cv2
import numpy as np

from face_labeller.config import Config
from face_labeller.contracts import Face, Match


MATCH_COLUMNS = (
    "frame_idx",
    "face_idx",
    "x",
    "y",
    "w",
    "h",
    "det_conf",
    "nearest_name",
    "distance",
    "threshold",
    "assigned_name",
    "confidence",
)


def validate_csv_path(cfg: Config) -> None:
    """Keep evidence output away from input media and embedding stores."""
    csv_path = cfg.csv_path.resolve()
    if csv_path == cfg.input_path.resolve():
        raise ValueError("CSV path must differ from the input video")
    if csv_path == cfg.output_path.resolve():
        raise ValueError("CSV path must differ from the output video")
    if csv_path.is_relative_to(cfg.ref_dir.resolve()):
        raise ValueError("CSV path must be outside the reference images directory")
    if csv_path.is_relative_to(cfg.cache_dir.resolve()):
        raise ValueError("CSV path must be outside the cache directory")


class MatchLogger:
    """Write one evidence row per face and optional original-frame crops."""

    def __init__(self, cfg: Config) -> None:
        validate_csv_path(cfg)
        self._cfg = cfg
        cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = tempfile.NamedTemporaryFile(
            mode="w",
            dir=cfg.csv_path.parent,
            prefix=f".{cfg.csv_path.name}.",
            suffix=".tmp",
            newline="",
            encoding="utf-8",
            delete=False,
        )
        self._staged_path = Path(self._file.name)
        self._writer = csv.writer(self._file)
        self._writer.writerow(MATCH_COLUMNS)

    @property
    def closed(self) -> bool:
        return self._file.closed

    def log(
        self,
        frame_idx: int,
        face_idx: int,
        face: Face,
        match: Match,
        threshold: float,
        frame: np.ndarray | None = None,
    ) -> None:
        self._writer.writerow(
            (
                frame_idx,
                face_idx,
                *face.box,
                face.det_conf,
                match.nearest_name,
                match.distance,
                threshold,
                match.name if match.name is not None else "Unknown",
                match.confidence,
            )
        )
        if self._cfg.debug_crops:
            if frame is None:
                raise ValueError("debug crops require the original frame")
            x, y, width, height = face.box
            frame_height, frame_width = frame.shape[:2]
            left = min(max(x, 0), frame_width)
            top = min(max(y, 0), frame_height)
            right = min(max(x + width, 0), frame_width)
            bottom = min(max(y + height, 0), frame_height)
            if right > left and bottom > top:
                crop_dir = self._cfg.csv_path.parent / "debug" / "crops"
                crop_dir.mkdir(parents=True, exist_ok=True)
                safe_name = "".join(
                    character
                    if character.isalnum() or character in {".", "_", "-"}
                    else "_"
                    for character in match.nearest_name
                )
                filename = (
                    f"{frame_idx:06d}_{face_idx:02d}_{safe_name}_{match.distance}.png"
                )
                if not cv2.imwrite(
                    str(crop_dir / filename), frame[top:bottom, left:right]
                ):
                    raise OSError(f"cannot write debug crop: {crop_dir / filename}")

    def close(self, publish: bool = True) -> None:
        if not self._file.closed:
            self._file.close()
        if self._staged_path is None:
            return
        staged_path = self._staged_path
        self._staged_path = None
        try:
            if publish:
                staged_path.replace(self._cfg.csv_path)
        finally:
            staged_path.unlink(missing_ok=True)

    def __enter__(self) -> MatchLogger:
        return self

    def __exit__(self, exc_type: object, *_exc: object) -> None:
        self.close(publish=exc_type is None)
