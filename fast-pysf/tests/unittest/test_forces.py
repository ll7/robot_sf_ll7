import unittest

import numpy as np
from pysocialforce.forces import centroid  # Import the centroid function from its module


class TestCentroidFunction(unittest.TestCase):
    """TestCentroidFunction"""

    def test_single_point(self):
        """Test that the centroid of a single point is the point itself."""
        points = np.array([[1, 2]])
        expected_centroid = (1.0, 2.0)
        self.assertEqual(centroid(points), expected_centroid)

    def test_two_points(self):
        """Test that the centroid of two points is the midpoint between them."""
        points = np.array([[1, 2], [3, 4]])
        expected_centroid = (2.0, 3.0)
        self.assertEqual(centroid(points), expected_centroid)

    def test_multiple_points(self):
        """Test the centroid of multiple points."""
        points = np.array([[1, 2], [3, 4], [-2, -1], [-3, -6]])
        expected_centroid = (-0.25, -0.25)  # Calculated manually or by another method
        self.assertEqual(centroid(points), expected_centroid)

    def test_with_zeros(self):
        """Test centroid calculation when zeroes are included in the points."""
        points = np.array([[0, 0], [2, 2], [0, 2]])
        expected_centroid = (0.6666666666666666, 1.3333333333333333)  # 2/3, 4/3
        self.assertAlmostEqual(centroid(points)[0], expected_centroid[0])
        self.assertAlmostEqual(centroid(points)[1], expected_centroid[1])

    def test_empty_array(self):
        """Test that the centroid of an empty array raises an error."""
        points = np.array([], dtype=np.float64).reshape(0, 2)
        with self.assertRaises(ValueError):
            centroid(points)


# This allows the test code to be executed when the script is run directly
if __name__ == "__main__":
    unittest.main()
