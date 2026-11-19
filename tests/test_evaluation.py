import unittest
from unittest.mock import Mock

import numpy as np

from ureter_perception.evaluation import counts, summarize, target_masks


class EvaluationTests(unittest.TestCase):
    def test_polygon_class_one_changes_meaning_between_exports(self):
        label = Mock()
        label.read_text.return_value = "1 0 0 1 0 1 1 0 1\n"
        v1 = target_masks(label, ["fiber", "laser", "stone", "tissue"], (4, 6))
        v2 = target_masks(label, ["fiber", "stone", "tissue"], (4, 6))
        self.assertEqual(int(v1["stone"].sum()), 0)
        self.assertEqual(int(v2["stone"].sum()), 24)

    def test_empty_reference_false_positive_contributes_to_micro_scores(self):
        truth = np.array([1, 1, 0], dtype=np.uint8)
        pred = np.array([True, False, False])
        positive = counts(truth, pred)
        empty = counts(truth * 0, pred)
        result = summarize([dict(stone=positive, fiber=empty), dict(stone=empty, fiber=empty)])
        self.assertEqual(result["stone"]["micro_dice"], 0.5)
        self.assertAlmostEqual(result["stone"]["micro_iou"], 1 / 3)
        self.assertEqual(result["stone"]["false_positive_images"], 1)
