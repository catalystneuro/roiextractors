import unittest

import numpy as np
from hdmf.testing import TestCase

from roiextractors.testing import generate_dummy_segmentation_extractor


class TestSegmentationExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.segmentation_extractor = generate_dummy_segmentation_extractor(num_samples=3, num_rows=2, num_columns=4)

    def test_has_time_vector_true(self):
        segmentation_extractor_with_times = generate_dummy_segmentation_extractor(
            num_frames=3, num_rows=3, num_columns=4, sampling_frequency=20.0
        )
        segmentation_extractor_with_times.set_times(times=np.array([1.1, 2.3, 3.7]))
        self.assertTrue(segmentation_extractor_with_times.has_time_vector())

    def test_has_time_vector_false(self):
        self.assertFalse(self.segmentation_extractor.has_time_vector())


if __name__ == "__main__":
    unittest.main()
