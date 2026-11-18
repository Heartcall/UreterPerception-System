import unittest

import numpy as np

from ureter_perception.math.camera import project, unproject
from ureter_perception.math.intersections import nearest_on_patch, ray_plane
from ureter_perception.math.transforms import inverse_transform, transform_points


class MathTests(unittest.TestCase):
    def test_camera_round_trip_with_skew(self):
        k = np.array([[500, 2, 200], [0, 490, 180], [0, 0, 1]])
        points = np.array([[-3, 4, 20], [10, -5, 40]])
        pixels = project(points, k)
        np.testing.assert_allclose(unproject(pixels, points[:, 2], k), points)
        with self.assertRaises(ValueError):
            project(np.array([1, 1, 0]), k)

    def test_rigid_transform_round_trip(self):
        matrix = np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]])
        point = np.array([2, 3, 4])
        moved = transform_points(matrix, point)
        np.testing.assert_allclose(moved, [1, 7, 10])
        np.testing.assert_allclose(transform_points(inverse_transform(matrix), moved), point)
        reflected = np.diag([-1, 1, 1, 1])
        with self.assertRaises(ValueError):
            inverse_transform(reflected)

    def test_ray_normalization_and_failure_reasons(self):
        hit = ray_plane([0, 0, 2], [0, 0, 5], [0, 0, 12], [0, 0, 2])
        self.assertEqual(hit.distance_mm, 10)
        np.testing.assert_allclose(hit.point_mm, [0, 0, 12])
        self.assertEqual(ray_plane([0, 0, 2], [1, 0, 0], [0, 0, 12], [0, 0, 1]).reason,
                         "grazing_ray")
        self.assertEqual(ray_plane([0, 0, 2], [0, 0, -1], [0, 0, 12], [0, 0, 1]).reason,
                         "behind_ray")

    def test_nearest_point_uses_bounded_patch(self):
        nearest = nearest_on_patch(np.array([5, 1, 2]), np.array([0, 0, 12]),
                                   np.array([[1, 0, 0], [0, 1, 0]]),
                                   np.array([[-3, -2], [3, 2]]))
        np.testing.assert_allclose(nearest, [3, 1, 12])
