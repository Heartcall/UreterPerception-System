from dataclasses import dataclass
from typing import Literal

import numpy as np

PixelDomain = Literal["raw", "rectified"]


@dataclass(frozen=True)
class Frame:
    stream_id: str
    index: int
    capture_ns: int
    clock_id: str
    timestamp_source: str
    image_bgr: np.ndarray
    pixel_domain: PixelDomain = "raw"

    @property
    def shape(self) -> tuple[int, int]:
        return self.image_bgr.shape[:2]


@dataclass(frozen=True)
class Segmentation2D:
    labels: np.ndarray
    tip_uv: np.ndarray | None = None
    tip_source: str | None = None
    pixel_domain: PixelDomain = "raw"


@dataclass(frozen=True)
class DepthObservation:
    values: np.ndarray
    valid: np.ndarray
    representation: Literal["z_mm", "range_mm", "relative_depth", "relative_inverse_depth"]
    pixel_domain: PixelDomain
    source: str


@dataclass(frozen=True)
class RobotSample:
    event_ns: int
    available_ns: int
    clock_id: str
    kind: Literal["command", "feedback"]
    values: dict[str, float]
