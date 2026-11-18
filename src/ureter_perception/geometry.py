from dataclasses import dataclass

import numpy as np

from .calibration import Calibration
from .contracts import DepthObservation, Segmentation2D
from .math.camera import unproject
from .math.fitting import principal_axes


@dataclass(frozen=True)
class FiberRay:
    tip_mm: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class StonePatch:
    center_mm: np.ndarray
    axes: np.ndarray
    normal: np.ndarray
    limits_mm: np.ndarray


@dataclass(frozen=True)
class GeometryObservation:
    fiber: FiberRay | None
    stone: StonePatch | None
    fiber_reason: str | None
    stone_reason: str | None
    depth_source: str | None


def _points(mask: np.ndarray, depth: DepthObservation, calibration: Calibration) -> np.ndarray:
    row, column = np.nonzero(mask & depth.valid)
    return unproject(np.column_stack((column, row)), depth.values[row, column], calibration.intrinsics)


def _fiber(seg: Segmentation2D, depth: DepthObservation, calib: Calibration) -> tuple[FiberRay | None, str | None]:
    if seg.tip_uv is None:
        return None, "missing_fiber_tip"
    column, row = np.rint(seg.tip_uv).astype(int)
    if not (depth.valid[row, column] and seg.labels[row, column] == 2):
        return None, "missing_tip_depth"
    points = _points(seg.labels == 2, depth, calib)
    if len(points) < 3:
        return None, "insufficient_fiber_points"
    center, spread, axes = principal_axes(points)
    tip = unproject(seg.tip_uv, depth.values[row, column], calib.intrinsics)
    direction = float((tip - center) @ axes[0])
    if spread[0] < 1e-8 or abs(direction) < 1e-8:
        return None, "undetermined_fiber_axis"
    return FiberRay(tip, np.copysign(1.0, direction) * axes[0]), None


def _stone(seg: Segmentation2D, depth: DepthObservation, calib: Calibration) -> tuple[StonePatch | None, str | None]:
    points = _points(seg.labels == 1, depth, calib)
    if len(points) < 3:
        return None, "insufficient_stone_points"
    center, spread, axes = principal_axes(points)
    if spread[1] < 1e-8:
        return None, "degenerate_stone_patch"
    plane = (points - center) @ axes[:2].T
    limits = np.stack((plane.min(axis=0), plane.max(axis=0)))
    return StonePatch(center, axes[:2], axes[2], limits), None


def observe_geometry(seg: Segmentation2D, depth: DepthObservation | None,
                     calibration: Calibration | None) -> GeometryObservation:
    source = None if depth is None else depth.source
    if depth is None or depth.representation != "z_mm":
        return GeometryObservation(None, None, "missing_metric_z", "missing_metric_z", source)
    if calibration is None:
        return GeometryObservation(None, None, "missing_calibration", "missing_calibration", source)
    same_grid = calibration.image_shape == seg.labels.shape == depth.values.shape == depth.valid.shape
    rectified = calibration.pixel_domain == seg.pixel_domain == depth.pixel_domain == "rectified"
    if not (same_grid and rectified):
        raise ValueError("3D geometry requires a shared rectified depth, label and calibration grid")
    fiber, fiber_reason = _fiber(seg, depth, calibration)
    stone, stone_reason = _stone(seg, depth, calibration)
    return GeometryObservation(fiber, stone, fiber_reason, stone_reason, source)
