from dataclasses import dataclass

import numpy as np

from .contracts import PixelDomain


@dataclass(frozen=True)
class Calibration:
    id: str
    intrinsics: np.ndarray
    image_shape: tuple[int, int]
    pixel_domain: PixelDomain = "rectified"

    def __post_init__(self) -> None:
        k = np.asarray(self.intrinsics, dtype=float)
        if (k.shape != (3, 3) or not np.all(np.isfinite(k)) or
                k[0, 0] <= 0 or k[1, 1] <= 0 or k[1, 0] != 0 or
                not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("Expected a finite pinhole intrinsic matrix")
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("Expected a positive (height, width)")
        object.__setattr__(self, "intrinsics", k)
