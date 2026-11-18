from typing import Protocol

import numpy as np

from .contracts import Frame, Segmentation2D


class Segmenter(Protocol):
    def segment(self, frame: Frame) -> Segmentation2D: ...


def segment(frame: Frame, backend: Segmenter) -> Segmentation2D:
    result = backend.segment(frame)
    labels = np.asarray(result.labels)
    if (labels.dtype != np.uint8 or labels.shape != frame.shape or
            result.pixel_domain != frame.pixel_domain or np.any(labels > 2)):
        raise ValueError(f"Frame {frame.index}: expected uint8 labels 0, 1, 2 in the frame pixel grid")
    tip = result.tip_uv
    if tip is not None:
        tip = np.asarray(tip, dtype=float)
        height, width = frame.shape
        if tip.shape != (2,) or not np.all((tip >= 0) & (tip <= [width - 1, height - 1])):
            raise ValueError(f"Frame {frame.index}: tip outside the frame pixel grid")
        if not result.tip_source:
            raise ValueError("A tip observation needs a source")
    return Segmentation2D(labels, tip, result.tip_source, result.pixel_domain)
