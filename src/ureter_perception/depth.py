from typing import Protocol

import numpy as np

from .contracts import DepthObservation, Frame


class DepthBackend(Protocol):
    def estimate(self, frame: Frame) -> DepthObservation | None: ...


def estimate_depth(frame: Frame, backend: DepthBackend | None) -> DepthObservation | None:
    result = None if backend is None else backend.estimate(frame)
    if result is None:
        return None
    values = np.asarray(result.values, dtype=np.float32)
    valid = np.asarray(result.valid, dtype=bool)
    if values.shape != frame.shape or valid.shape != frame.shape or result.pixel_domain != frame.pixel_domain:
        raise ValueError(f"Frame {frame.index}: depth and image must share a pixel grid")
    if result.representation not in {"z_mm", "range_mm", "relative_depth", "relative_inverse_depth"}:
        raise ValueError(f"Unknown depth representation: {result.representation}")
    valid = valid & np.isfinite(values)
    if result.representation in {"z_mm", "range_mm"}:
        valid &= values > 0
    return DepthObservation(values, valid, result.representation, result.pixel_domain, result.source)
