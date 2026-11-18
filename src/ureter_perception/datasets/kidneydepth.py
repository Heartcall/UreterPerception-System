from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class KidneyDepthSample:
    trajectory: str
    index: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.trajectory)
        if (not self.trajectory or "\\" in self.trajectory or ":" in self.trajectory or
                path.is_absolute() or ".." in path.parts or len(path.parts) != 4 or self.index < 0):
            raise ValueError("Use shape/material/light/trajectory and a nonnegative frame index")

    @property
    def depth_member(self) -> str:
        return f"KidneyDepthMetric/{self.trajectory}/depth/{self.index:04d}.tif"

    @property
    def image_member(self) -> str:
        return f"KidneyDepthMetric/{self.trajectory}/images/{self.index:04d}.png"
