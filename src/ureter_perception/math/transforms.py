import numpy as np


def rigid_transform(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected a finite 4x4 rigid transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1]) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or
            not np.isclose(np.linalg.det(rotation), 1)):
        raise ValueError("Expected a rigid transform with a proper rotation")
    return matrix


def transform_points(transform: np.ndarray, points_mm: np.ndarray) -> np.ndarray:
    matrix = rigid_transform(transform)
    return np.asarray(points_mm) @ matrix[:3, :3].T + matrix[:3, 3]


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    matrix = rigid_transform(transform)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result
