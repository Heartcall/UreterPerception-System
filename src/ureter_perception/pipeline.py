from dataclasses import asdict, dataclass

import numpy as np

from .alignment import AlignedRobot
from .calibration import Calibration
from .contracts import DepthObservation, Frame
from .depth import DepthBackend, estimate_depth
from .geometry import GeometryObservation, observe_geometry
from .segmentation import Segmenter, segment
from .targeting import Measurements, measure


@dataclass(frozen=True)
class FrameResult:
    frame: Frame
    pixel_counts: tuple[int, int, int]
    tip_uv: np.ndarray | None
    tip_source: str | None
    geometry: GeometryObservation
    measurements: Measurements
    robot: AlignedRobot
    depth: DepthObservation | None = None

    def record(self) -> dict:
        return {
            "frame_index": self.frame.index, "stream_id": self.frame.stream_id,
            "capture_ns": self.frame.capture_ns, "clock_id": self.frame.clock_id,
            "timestamp_source": self.frame.timestamp_source, "pixel_domain": self.frame.pixel_domain,
            "pixel_counts": dict(zip(("background", "stone", "fiber"), self.pixel_counts)),
            "tip_uv": self.tip_uv, "tip_source": self.tip_source,
            "geometry": asdict(self.geometry), "measurements": asdict(self.measurements),
            "robot": asdict(self.robot),
        }


def process_frame(frame: Frame, segmenter: Segmenter, depth_backend: DepthBackend | None = None,
                  calibration: Calibration | None = None,
                  robot: AlignedRobot = AlignedRobot(None, None)) -> FrameResult:
    labels = segment(frame, segmenter)
    depth = estimate_depth(frame, depth_backend)
    geometry = observe_geometry(labels, depth, calibration)
    counts = tuple(np.bincount(labels.labels.ravel(), minlength=3).tolist())
    return FrameResult(frame, counts, labels.tip_uv, labels.tip_source, geometry, measure(geometry), robot, depth)
