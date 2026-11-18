from contextlib import contextmanager
from pathlib import Path
import json

import numpy as np


def json_array(value: object) -> list:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


@contextmanager
def write_run(output: Path, manifest: dict):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    pending = output / "frames.partial.jsonl"
    count = 0
    with pending.open("x", encoding="utf-8", newline="\n") as stream:
        def write(record: dict) -> None:
            nonlocal count
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=json_array) + "\n")
            count += 1

        yield write
    pending.replace(output / "frames.jsonl")
    temporary_manifest = output / "manifest.partial.json"
    with temporary_manifest.open("x", encoding="utf-8") as stream:
        json.dump({**manifest, "schema_version": 1, "frames": count}, stream, indent=2)
    temporary_manifest.replace(output / "manifest.json")
