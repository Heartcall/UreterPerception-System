from bisect import bisect_right
from dataclasses import dataclass
from collections.abc import Iterable

from .contracts import RobotSample


@dataclass(frozen=True)
class AlignedRobot:
    command: RobotSample | None
    feedback: RobotSample | None


class RobotTimeline:
    def __init__(self, samples: Iterable[RobotSample], clock_id: str):
        self.clock_id = clock_id
        self.rows = {"command": [], "feedback": []}
        for sample in samples:
            if sample.clock_id != clock_id:
                raise ValueError("Robot samples and frame must use the same mapped clock")
            self.rows[sample.kind].append(sample)
        self.times = {}
        for kind, rows in self.rows.items():
            rows.sort(key=lambda sample: (sample.event_ns, sample.available_ns))
            self.times[kind] = [sample.event_ns for sample in rows]

    def at(self, capture_ns: int, decision_ns: int, max_age_ns: int) -> AlignedRobot:
        if not 0 <= capture_ns <= decision_ns or max_age_ns < 0:
            raise ValueError("Invalid capture time, decision time, or sample age")

        def latest(kind: str) -> RobotSample | None:
            start = bisect_right(self.times[kind], capture_ns) - 1
            for index in range(start, -1, -1):
                sample = self.rows[kind][index]
                if capture_ns - sample.event_ns > max_age_ns:
                    break
                if sample.available_ns <= decision_ns:
                    return sample
            return None

        return AlignedRobot(latest("command"), latest("feedback"))
