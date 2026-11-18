from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recording:
    stream_id: str
    video: Path
    segmentation_dir: Path
