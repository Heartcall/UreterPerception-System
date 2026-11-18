from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice, zip_longest
from pathlib import Path
import json

import numpy as np

from ..contracts import Frame, Segmentation2D
from ..io.replay import read_json, read_labels, video_frames


@dataclass(frozen=True)
class CachedFrame:
    frame: Frame
    result: Segmentation2D

    def segment(self, frame: Frame) -> Segmentation2D:
        if frame is not self.frame:
            raise ValueError("A cached segmentation must be used with its paired frame")
        return self.result


class UreterSAMReplay:
    def __init__(self, video: Path, cache_dir: Path, stream_id: str):
        self.video = video
        self.cache_dir = cache_dir.resolve()
        self.stream_id = stream_id
        self.manifest = read_json(self.cache_dir / "manifest.json")
        if self.manifest["mask_labels"] != {"0": "background", "1": "stone", "2": "fiber"}:
            raise ValueError("Unexpected UreterSAM label map")
        digest = sha256()
        with self.video.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.manifest["input_sha256"]:
            raise ValueError("Video does not match the UreterSAM manifest")

    def frames(self, limit: int | None = None) -> Iterator[CachedFrame]:
        if limit is not None and limit <= 0:
            raise ValueError("Frame limit must be positive")
        previous_ns = -1
        with (self.cache_dir / "frames.jsonl").open(encoding="utf-8") as records:
            images = video_frames(self.video)
            try:
                expected = self.manifest["frames"]
                count = expected if limit is None else min(expected, limit)
                if count <= 0:
                    raise ValueError("UreterSAM manifest has no frames")
                # A valid UreterSAM export may cover only a prefix of its video.
                lines = islice(records, count) if count < expected else records
                pairs = zip_longest(islice(images, count), lines)
                seen = 0
                for index, (image, line) in enumerate(pairs):
                    if image is None or line is None or index >= count:
                        raise ValueError(f"Video and UreterSAM export have different lengths at frame {index}")
                    cached = self._decode(index, image, json.loads(line))
                    if cached.frame.capture_ns <= previous_ns:
                        raise ValueError(f"UreterSAM frame time is not increasing at frame {index}")
                    previous_ns = cached.frame.capture_ns
                    seen += 1
                    yield cached
                if seen != count:
                    raise ValueError("UreterSAM cache ended before the declared frame count")
            finally:
                images.close()

    def _decode(self, index: int, image: np.ndarray, record: dict) -> CachedFrame:
        if record["frame_index"] != index:
            raise ValueError(f"Unexpected frame index in UreterSAM cache at {index}")
        mask_path = (self.cache_dir / record["mask"]).resolve()
        if not mask_path.is_relative_to(self.cache_dir):
            raise ValueError(f"Mask path escapes UreterSAM cache at frame {index}")
        mask = read_labels(mask_path)
        if sha256(mask.tobytes()).hexdigest() != record["mask_sha256"]:
            raise ValueError(f"UreterSAM mask hash mismatch at frame {index}")
        rgb_hash = record.get("source_rgb_sha256")
        if rgb_hash and sha256(image[..., ::-1].tobytes()).hexdigest() != rgb_hash:
            raise ValueError(f"UreterSAM decoded frame hash mismatch at frame {index}")
        timestamp_ns = round(record["timestamp_seconds"] * 1_000_000_000)
        frame = Frame(self.stream_id, index, timestamp_ns, f"{self.stream_id}:video",
                      "frame_index_fps", image)
        info = record.get("g0") or {}
        tip_data = info.get("tip") or {}
        tip = tip_data.get("vertex") if info.get("applied") else None
        return CachedFrame(frame, Segmentation2D(mask, tip, "g0_vertex_estimate" if tip is not None else None))
