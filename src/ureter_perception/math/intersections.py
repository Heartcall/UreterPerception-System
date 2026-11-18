from dataclasses import dataclass

import numpy as np


def unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    length = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(length) or length <= 1e-12:
        raise ValueError("Expected a nonzero finite 3D vector")
    return vector / length


@dataclass(frozen=True)
class RayIntersection:
    point_mm: np.ndarray | None
    distance_mm: float | None
    reason: str | None = None


def ray_plane(
    origin_mm: np.ndarray,
    direction: np.ndarray,
    plane_point_mm: np.ndarray,
    plane_normal: np.ndarray,
    min_cosine: float = 1e-6,
) -> RayIntersection:
    axis = unit_vector(direction)
    normal = unit_vector(plane_normal)
    origin = np.asarray(origin_mm, dtype=float)
    point = np.asarray(plane_point_mm, dtype=float)
    if (origin.shape != (3,) or point.shape != (3,) or
            not np.all(np.isfinite([origin, point])) or not (0 < min_cosine <= 1)):
        raise ValueError("Expected finite points and a valid incidence threshold")
    denominator = float(normal @ axis)
    if abs(denominator) < min_cosine:
        return RayIntersection(None, None, "grazing_ray")
    distance = float((normal @ (point - origin)) / denominator)
    if distance < 0:
        return RayIntersection(None, None, "behind_ray")
    return RayIntersection(origin + distance * axis, distance)


def patch_coordinates(point: np.ndarray, center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return axes @ (point - center)


def nearest_on_patch(point: np.ndarray, center: np.ndarray, axes: np.ndarray,
                     limits: np.ndarray) -> np.ndarray:
    local = patch_coordinates(point, center, axes)
    return center + np.clip(local, limits[0], limits[1]) @ axes
