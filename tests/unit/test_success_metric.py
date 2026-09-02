import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "navol"))

from navol.evaluation.metrics import is_success


class SuccessMetricTests(unittest.TestCase):
    def test_scalar_distance_below_one_metre_succeeds(self):
        self.assertTrue(is_success(0.999))

    def test_scalar_distance_at_one_metre_fails(self):
        self.assertFalse(is_success(1.0))

    def test_scalar_distance_above_one_metre_fails(self):
        self.assertFalse(is_success(1.001))

    def test_array_distances_use_the_same_boundary(self):
        actual = is_success(np.array([0.999, 1.0, 1.001]))
        np.testing.assert_array_equal(actual, np.array([True, False, False]))


if __name__ == "__main__":
    unittest.main()

