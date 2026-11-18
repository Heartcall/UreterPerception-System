from collections.abc import Iterable
from pathlib import Path
from typing import Protocol
import csv
import math

from ..contracts import RobotSample


class RobotSource(Protocol):
    """Read-only boundary for an eventual device connector."""

    def samples(self) -> Iterable[RobotSample]: ...


class RobotCsv:
    def __init__(self, path: Path):
        self.path = path

    def samples(self) -> Iterable[RobotSample]:
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"event_ns", "available_ns", "clock_id", "kind"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Robot log must contain {sorted(required)}")
            for line, row in enumerate(reader, start=2):
                try:
                    event_ns = int(row["event_ns"])
                    available_ns = int(row["available_ns"])
                    kind = row["kind"]
                    clock_id = row["clock_id"].strip()
                    values = {key: float(value) for key, value in row.items()
                              if key not in required and value and value.strip()}
                    if (event_ns < 0 or event_ns > available_ns or not clock_id or
                            kind not in {"command", "feedback"} or not values or
                            not all(math.isfinite(value) for value in values.values())):
                        raise ValueError("Invalid robot sample")
                    yield RobotSample(event_ns, available_ns, clock_id, kind, values)
                except (AttributeError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid robot log at line {line}: {error}") from error
