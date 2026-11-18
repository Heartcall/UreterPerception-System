import numpy as np


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid, singular values and ordered orthonormal axes."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Expected at least three finite 3D points")
    centroid = values.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(values - centroid, full_matrices=False)
    return centroid, singular_values, axes
