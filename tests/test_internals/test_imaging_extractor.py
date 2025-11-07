import unittest

import numpy as np
from hdmf.testing import TestCase

from roiextractors.testing import generate_dummy_imaging_extractor


class TestImagingExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imaging_extractor = generate_dummy_imaging_extractor(
            num_samples=3, num_rows=2, num_columns=4, sampling_frequency=20.0
        )

    def test_has_time_vector_true(self):
        imaging_extractor_with_times = generate_dummy_imaging_extractor(
            num_samples=3, num_rows=3, num_columns=4, sampling_frequency=20.0
        )
        imaging_extractor_with_times.set_times(times=np.array([1.1, 2.3, 3.7]))
        self.assertTrue(imaging_extractor_with_times.has_time_vector())

    def test_has_time_vector_false(self):
        self.assertFalse(self.imaging_extractor.has_time_vector())


if __name__ == "__main__":
    unittest.main()
