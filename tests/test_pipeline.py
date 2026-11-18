from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ureter_perception.depth import estimate_depth
from ureter_perception.io.artifacts import write_run
from ureter_perception.io.scene import read_geometry, read_scene
from ureter_perception.pipeline import process_frame
from ureter_perception.segmentation import segment
from ureter_perception.targeting import measure

FIXTURES = Path(__file__).parent / "fixtures"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.scene = read_scene(FIXTURES / "known_plane.json")

    def test_known_metric_scene_end_to_end(self):
        scene = self.scene
        output = process_frame(scene.frame, scene, scene, scene.calibration)
        self.assertEqual(output.pixel_counts, (47, 13, 3))
        np.testing.assert_allclose(output.geometry.fiber.tip_mm, [-1.5, 0, 6])
        np.testing.assert_allclose(output.measurements.intersection_mm, [-1.5, 0, 12])
        self.assertAlmostEqual(output.measurements.axial_distance_mm, 6)
        self.assertAlmostEqual(output.measurements.nearest_patch_mm, 6)

    def test_relative_depth_and_missing_depth_stay_nonmetric(self):
        scene = self.scene
        relative = replace(scene, depth=replace(scene.depth, representation="relative_inverse_depth"))
        for backend in (None, relative):
            with self.subTest(backend=type(backend).__name__):
                result = process_frame(scene.frame, scene, backend, scene.calibration)
                self.assertIsNone(result.measurements.axial_distance_mm)
                self.assertEqual(result.measurements.intersection_reason, "missing_metric_z")

    def test_public_boundaries_reject_bad_labels_and_depth_grid(self):
        scene = self.scene
        bad_labels = replace(scene, segmentation=replace(
            scene.segmentation, labels=np.full(scene.frame.shape, 3, np.uint8)))
        with self.assertRaises(ValueError):
            segment(scene.frame, bad_labels)
        wrong_labels_grid = replace(scene, segmentation=replace(scene.segmentation, pixel_domain="raw"))
        with self.assertRaises(ValueError):
            segment(scene.frame, wrong_labels_grid)
        wrong_grid = replace(scene, depth=replace(scene.depth, pixel_domain="raw"))
        with self.assertRaises(ValueError):
            estimate_depth(scene.frame, wrong_grid)

    def test_targeting_independent_of_segmenter_and_depth(self):
        geometry = read_geometry(FIXTURES / "known_ray.json")
        self.assertAlmostEqual(measure(geometry).axial_distance_mm, 10)
        outside = replace(geometry, fiber=replace(geometry.fiber, tip_mm=np.array([8, 0, 2])))
        output = measure(outside)
        self.assertEqual(output.intersection_reason, "outside_surface")
        self.assertGreater(output.nearest_patch_mm, 10)

    def test_results_are_committed_only_for_completed_replay(self):
        scene = self.scene
        record = process_frame(scene.frame, scene, scene, scene.calibration).record()
        with TemporaryDirectory(dir=FIXTURES.parents[1]) as directory:
            output = Path(directory) / "run"
            with write_run(output, {"mode": "test"}) as write:
                write(record)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["frames"], 1)
            self.assertEqual(json.loads((output / "frames.jsonl").read_text())["tip_uv"], [3.0, 3.0])
            with self.assertRaises(FileExistsError):
                with write_run(output, {}):
                    pass
            failed = Path(directory) / "failed"
            with self.assertRaises(RuntimeError):
                with write_run(failed, {}) as write:
                    write(record)
                    raise RuntimeError("simulated input failure")
            self.assertFalse((failed / "manifest.json").exists())
