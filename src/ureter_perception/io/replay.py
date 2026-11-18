from collections.abc import Iterator
from pathlib import Path
import json

import numpy as np


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_labels(path: Path) -> np.ndarray:
    import cv2

    labels = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise ValueError(f"Cannot decode label image: {path}")
    return labels


def video_frames(path: Path) -> Iterator[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        while True:
            ok, image = capture.read()
            if not ok:
                return
            yield image
    finally:
        capture.release()
