from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..calibration import Calibration
from ..contracts import DepthObservation, Frame, Segmentation2D
from ..geometry import FiberRay, GeometryObservation, StonePatch
from .replay import read_json


@dataclass(frozen=True)
class Scene:
    frame: Frame
    calibration: Calibration
    segmentation: Segmentation2D
    depth: DepthObservation

    def segment(self, frame: Frame) -> Segmentation2D:
        return self.segmentation

    def estimate(self, frame: Frame) -> DepthObservation:
        return self.depth


def read_scene(path: Path) -> Scene:
    value = read_json(path)
    labels = np.asarray(value["labels"])
    if labels.ndim != 2 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Scene needs a 2D segmentation with labels 0, 1 and 2")
    depth = np.asarray(value["z_mm"], dtype=np.float32)
    frame = Frame(path.stem, 0, 0, "fixture", "synthetic",
                  np.zeros((*labels.shape, 3), dtype=np.uint8), "rectified")
    return Scene(frame, Calibration(path.stem, value["intrinsics"], labels.shape),
                 Segmentation2D(labels.astype(np.uint8), value.get("tip_uv"), "synthetic", "rectified"),
                 DepthObservation(depth, depth > 0, "z_mm", "rectified", "synthetic_truth"))


def read_geometry(path: Path) -> GeometryObservation:
    value = read_json(path)
    fiber = value["fiber"]
    stone = value["stone"]
    return GeometryObservation(
        FiberRay(np.asarray(fiber["tip_mm"]), np.asarray(fiber["axis"])),
        StonePatch(np.asarray(stone["center_mm"]), np.asarray(stone["axes"]),
                   np.asarray(stone["normal"]), np.asarray(stone["limits_mm"])),
        None, None, "synthetic_truth",
    )
