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

    def test_time_to_sample_indices_with_sampling_frequency(self):
        """Round-trip: get_timestamps -> time_to_sample_indices should recover original indices."""
        sample_indices = np.array([0, 1, 2])
        times = self.imaging_extractor.get_timestamps()
        recovered = self.imaging_extractor.time_to_sample_indices(times)
        np.testing.assert_array_equal(recovered, sample_indices)

    def test_time_to_sample_indices_with_set_times(self):
        """When an explicit time vector is set, time_to_sample_indices should use it."""
        extractor = generate_dummy_imaging_extractor(num_samples=4, num_rows=2, num_columns=2, sampling_frequency=10.0)
        irregular_times = np.array([0.0, 0.5, 1.5, 3.0])
        extractor.set_times(irregular_times)

        # Query at exact timestamps
        result = extractor.time_to_sample_indices(np.array([0.0, 0.5, 1.5, 3.0]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2, 3]))

        # Query between timestamps should return the sample at or just before
        result = extractor.time_to_sample_indices(np.array([0.3, 1.0, 2.0]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_time_to_sample_indices_with_native_timestamps_uncached(self):
        """The method should work even when native timestamps exist but haven't been cached yet."""
        extractor = generate_dummy_imaging_extractor(num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0)
        native_times = np.array([0.0, 0.1, 0.2])
        extractor.get_native_timestamps = lambda: native_times

        # _times is None but native timestamps exist
        assert extractor._times is None
        result = extractor.time_to_sample_indices(np.array([0.0, 0.1, 0.2]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
