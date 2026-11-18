import numpy as np


def project(points_mm: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm, dtype=float)
    if not np.all(np.isfinite(points)) or np.any(points[..., 2] <= 0):
        raise ValueError("Projection requires finite points in front of the camera")
    homogeneous = points @ intrinsics.T
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def unproject(pixels_uv: np.ndarray, z_mm: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=float)
    z = np.asarray(z_mm, dtype=float)
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(z)) or np.any(z <= 0):
        raise ValueError("Unprojection requires finite pixels and positive z depth")
    homogeneous = np.concatenate((pixels, np.ones((*pixels.shape[:-1], 1))), axis=-1)
    rays = np.linalg.solve(intrinsics, homogeneous.reshape(-1, 3).T).T.reshape(homogeneous.shape)
    return (rays / rays[..., 2:3]) * z[..., None]
