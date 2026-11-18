from dataclasses import dataclass

import numpy as np

from .geometry import GeometryObservation
from .math.intersections import nearest_on_patch, patch_coordinates, ray_plane


@dataclass(frozen=True)
class Measurements:
    patch_extents_mm: np.ndarray | None
    nearest_patch_mm: float | None
    axial_distance_mm: float | None
    intersection_mm: np.ndarray | None
    intersection_reason: str | None


def measure(observation: GeometryObservation) -> Measurements:
    fiber = observation.fiber
    stone = observation.stone
    if stone is None:
        return Measurements(None, None, None, None, observation.stone_reason)
    lower, upper = stone.limits_mm
    extents = upper - lower
    if fiber is None:
        return Measurements(extents, None, None, None, observation.fiber_reason)
    nearest = nearest_on_patch(fiber.tip_mm, stone.center_mm, stone.axes, stone.limits_mm)
    clearance = float(np.linalg.norm(nearest - fiber.tip_mm))
    hit = ray_plane(fiber.tip_mm, fiber.axis, stone.center_mm, stone.normal)
    if hit.point_mm is None:
        return Measurements(extents, clearance, None, None, hit.reason)
    in_plane = patch_coordinates(hit.point_mm, stone.center_mm, stone.axes)
    if np.any(in_plane < lower - 1e-8) or np.any(in_plane > upper + 1e-8):
        return Measurements(extents, clearance, None, None, "outside_surface")
    return Measurements(extents, clearance, hit.distance_mm, hit.point_mm, None)
