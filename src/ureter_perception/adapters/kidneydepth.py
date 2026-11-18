from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from ..contracts import DepthObservation
from ..datasets.kidneydepth import KidneyDepthSample


@dataclass(frozen=True)
class KidneyDepthTruth:
    sample: KidneyDepthSample
    values: np.ndarray

    def as_depth(self, *, metric_factor: float, representation: str, pixel_domain: str) -> DepthObservation:
        if not np.isfinite(metric_factor) or metric_factor <= 0:
            raise ValueError("The dataset sample needs a positive, explicit conversion to millimetres")
        if representation not in {"z_mm", "range_mm"} or pixel_domain not in {"raw", "rectified"}:
            raise ValueError("The dataset sample needs a declared depth convention and pixel domain")
        values = self.values * metric_factor
        return DepthObservation(values, np.isfinite(values) & (values > 0), representation,
                                pixel_domain, "kidneydepth_dataset_truth")


def read_metric_depth(zip_path: Path, sample: KidneyDepthSample) -> KidneyDepthTruth:
    import tifffile

    with ZipFile(zip_path) as archive:
        archive.getinfo(sample.image_member)
        member = archive.getinfo(sample.depth_member)
        values = np.asarray(tifffile.imread(BytesIO(archive.read(member))), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a single-channel depth map: {sample.depth_member}")
    return KidneyDepthTruth(sample, values)
