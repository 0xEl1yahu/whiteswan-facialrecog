import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np
import pytest
from deepface import DeepFace


VIDEO_PATH = Path("data/video-source/nimbus.mp4")
SMOKE_FRAME_INDEX = 150  # First stable close shot with real RetinaFace detections.


def _real_faces(results: list[dict]) -> list[dict]:
    return [result for result in results if result["face_confidence"] > 0]


@pytest.mark.slow
def test_cpu_stack_embeds_same_video_frame_identically() -> None:
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    assert capture.isOpened(), f"Could not open test video: {VIDEO_PATH}"
    capture.set(cv2.CAP_PROP_POS_FRAMES, SMOKE_FRAME_INDEX)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None, f"Could not read frame {SMOKE_FRAME_INDEX}"

    load_started = time.perf_counter()
    DeepFace.build_model("Facenet512")
    model_load_seconds = time.perf_counter() - load_started

    kwargs = {
        "model_name": "Facenet512",
        "detector_backend": "retinaface",
        "enforce_detection": False,
        "align": True,
        "normalization": "base",
        "l2_normalize": True,
    }

    first_started = time.perf_counter()
    first = _real_faces(DeepFace.represent(img_path=frame, **kwargs))
    first_seconds = time.perf_counter() - first_started

    second_started = time.perf_counter()
    second = _real_faces(DeepFace.represent(img_path=frame, **kwargs))
    second_seconds = time.perf_counter() - second_started

    print(
        f"model_load_seconds={model_load_seconds:.3f} "
        f"first_embed_seconds={first_seconds:.3f} "
        f"second_embed_seconds={second_seconds:.3f} faces={len(first)}"
    )

    assert first, f"Frame {SMOKE_FRAME_INDEX} did not contain a real RetinaFace detection"
    assert len(first) == len(second)
    for first_face, second_face in zip(first, second, strict=True):
        assert first_face["facial_area"] == second_face["facial_area"]
        np.testing.assert_array_equal(first_face["embedding"], second_face["embedding"])
