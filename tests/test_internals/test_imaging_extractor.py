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

    def test_has_time_vector_with_native_timestamps(self):
        """An extractor whose format records timestamps has a time vector before and after reading them."""
        extractor = generate_dummy_imaging_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0, native_timestamps="evenly_spaced"
        )
        self.assertTrue(extractor.has_time_vector())
        extractor.get_timestamps()
        self.assertTrue(extractor.has_time_vector())

    def test_native_timestamps_read_only_when_has_native_timestamps(self):
        """Native timestamps are ignored when the extractor does not report that its format has them."""
        extractor = generate_dummy_imaging_extractor(num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0)
        extractor.get_native_timestamps = lambda start_sample=None, end_sample=None: np.array([5.0, 5.1, 5.2])

        self.assertFalse(extractor.has_time_vector())
        np.testing.assert_array_equal(extractor.time_to_sample_indices(np.array([0.1])), np.array([1]))
        np.testing.assert_array_equal(extractor.get_timestamps(), np.array([0.0, 0.1, 0.2]))

    def test_copy_times_with_native_timestamps(self):
        """Native timestamps of the source that have not been read yet are copied."""
        source = generate_dummy_imaging_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0, native_timestamps="unevenly_spaced"
        )
        target = generate_dummy_imaging_extractor(num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0)
        target.copy_times(source)

        np.testing.assert_array_equal(target.get_timestamps(), source.get_native_timestamps())

    def test_timestamps_start_at_t_start(self):
        """Timestamps computed from the sampling frequency start at the start time of the extractor."""
        extractor = generate_dummy_imaging_extractor(num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0)
        extractor._get_t_start = lambda: 5.0

        np.testing.assert_array_almost_equal(extractor.get_timestamps(), np.array([5.0, 5.1, 5.2]))
        np.testing.assert_array_equal(extractor.time_to_sample_indices(np.array([5.0, 5.1, 5.2])), np.array([0, 1, 2]))

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
        extractor._has_native_timestamps = lambda: True

        # _times is None but native timestamps exist
        assert extractor._times is None
        result = extractor.time_to_sample_indices(np.array([0.0, 0.1, 0.2]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_time_to_sample_indices_with_set_times_and_native_timestamps(self):
        """Times set with set_times should take precedence over native timestamps and survive the call."""
        extractor = generate_dummy_imaging_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0, native_timestamps="evenly_spaced"
        )
        set_times = np.array([100.0, 100.1, 100.2])
        extractor.set_times(set_times)

        result = extractor.time_to_sample_indices(np.array([100.0, 100.1, 100.2]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))
        np.testing.assert_array_equal(extractor.get_timestamps(), set_times)


if __name__ == "__main__":
    unittest.main()
