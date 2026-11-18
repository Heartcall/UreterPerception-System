from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ureter_perception.adapters.kidneydepth import KidneyDepthTruth
from ureter_perception.adapters.robot import RobotCsv
from ureter_perception.adapters.uretersam import UreterSAMReplay
from ureter_perception.alignment import RobotTimeline
from ureter_perception.datasets.kidneydepth import KidneyDepthSample

FIXTURES = Path(__file__).parent / "fixtures"


class AdapterTests(unittest.TestCase):
    def test_robot_arrival_time_and_stale_data(self):
        samples = list(RobotCsv(FIXTURES / "robot.csv").samples())
        replay = RobotTimeline(samples, "fixture")
        first = replay.at(100, 100, 50)
        self.assertEqual(first.command.event_ns, 60)
        self.assertEqual(first.feedback.event_ns, 85)
        self.assertEqual(replay.at(100, 105, 20).feedback.event_ns, 95)
        self.assertIsNone(replay.at(100, 100, 10).feedback)
        with self.assertRaises(ValueError):
            RobotTimeline(samples, "other_clock")

    def test_kidneydepth_sample_convention_is_explicit(self):
        sample = KidneyDepthSample("ct/flesh/dynamic/traj_1", 0)
        self.assertEqual(sample.depth_member, "KidneyDepthMetric/ct/flesh/dynamic/traj_1/depth/0000.tif")
        with self.assertRaises(ValueError):
            KidneyDepthSample("ct/../dynamic/traj_1", 0)
        raw = KidneyDepthTruth(sample, np.array([[1.5, np.inf]], dtype=np.float32))
        values = raw.as_depth(metric_factor=10, representation="z_mm", pixel_domain="rectified")
        self.assertEqual(values.values[0, 0], 15)
        self.assertFalse(values.valid[0, 1])

    def test_g0_cache_rejects_wrong_video_and_wrong_mask(self):
        with TemporaryDirectory(dir=FIXTURES.parents[1]) as directory:
            cache = Path(directory)
            video = cache / "video.mp4"
            video.write_bytes(b"stub video")
            manifest = {"input_sha256": "wrong", "frames": 1,
                        "mask_labels": {"0": "background", "1": "stone", "2": "fiber"}}
            manifest_path = cache / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Video does not match"):
                UreterSAMReplay(video, cache, "stub")
            manifest["input_sha256"] = sha256(video.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            labels = np.array([[0, 1, 2]], dtype=np.uint8)
            record = {"frame_index": 0, "timestamp_seconds": 0.0, "mask": "mask.png",
                      "mask_sha256": "wrong"}
            records = cache / "frames.jsonl"
            records.write_text(json.dumps(record) + "\n")
            source = UreterSAMReplay(video, cache, "stub")
            with patch("ureter_perception.adapters.uretersam.video_frames",
                       side_effect=lambda path: (x for x in [np.zeros((1, 3, 3), np.uint8)])), \
                    patch("ureter_perception.adapters.uretersam.read_labels", return_value=labels):
                with self.assertRaisesRegex(ValueError, "mask hash mismatch"):
                    list(source.frames())
                record["mask_sha256"] = sha256(labels.tobytes()).hexdigest()
                records.write_text(json.dumps(record) + "\n")
                [result] = list(source.frames())
                np.testing.assert_array_equal(result.result.labels, labels)
                self.assertEqual(result.frame.timestamp_source, "frame_index_fps")
                records.write_text("")
                with self.assertRaisesRegex(ValueError, "different lengths"):
                    list(source.frames())
