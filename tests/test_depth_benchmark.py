import unittest
from unittest.mock import Mock

import numpy as np

from ureter_perception.depth_benchmark import ordering_score, partition


class DepthBenchmarkTests(unittest.TestCase):
    def test_order_direction_and_constant_prediction(self):
        truth = np.arange(1, 101, dtype=float).reshape(10, 10)
        for pred, inverse, expected in [(truth, False, 1), (1 / truth, True, 1),
                                        (-truth, False, 0), (np.ones_like(truth), False, .5)]:
            self.assertEqual(ordering_score(truth, pred, inverse), (expected, 1))

    def test_trajectory_variants_stay_together_and_require_pairs(self):
        names = []
        for traj in ["traj_1", "traj_random_1", "traj_3", "traj_5"]:
            base = "KidneyDepthMetric/ct/flesh/static/" + traj
            names.extend([base + "/images/0000.png", base + "/depth/0000.tif"])
        archive = Mock()
        archive.namelist.return_value = names
        groups = partition(archive)
        self.assertEqual(sum(split == "train" for split, _ in groups), 2)
        names.pop()
        with self.assertRaises(ValueError):
            partition(archive)
