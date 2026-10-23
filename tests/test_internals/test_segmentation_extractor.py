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
            num_samples=3, num_rows=3, num_columns=4, sampling_frequency=20.0
        )
        segmentation_extractor_with_times.set_times(times=np.array([1.1, 2.3, 3.7]))
        self.assertTrue(segmentation_extractor_with_times.has_time_vector())

    def test_has_time_vector_false(self):
        self.assertFalse(self.segmentation_extractor.has_time_vector())

    def test_has_time_vector_with_native_timestamps(self):
        """An extractor whose format records timestamps has a time vector before and after reading them."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0, native_timestamps="evenly_spaced"
        )
        self.assertTrue(extractor.has_time_vector())
        extractor.get_timestamps()
        self.assertTrue(extractor.has_time_vector())

    def test_native_timestamps_read_only_when_has_native_timestamps(self):
        """Native timestamps are ignored when the extractor does not report that its format has them."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0
        )
        extractor.get_native_timestamps = lambda start_sample=None, end_sample=None: np.array([5.0, 5.1, 5.2])

        self.assertFalse(extractor.has_time_vector())
        np.testing.assert_array_equal(extractor.time_to_sample_indices(np.array([0.1])), np.array([1]))
        np.testing.assert_array_equal(extractor.get_timestamps(), np.array([0.0, 0.1, 0.2]))

    def test_timestamps_start_at_t_start(self):
        """Timestamps computed from the sampling frequency start at the start time of the extractor."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0
        )
        extractor._get_native_t_start = lambda: 5.0

        np.testing.assert_array_almost_equal(extractor.get_timestamps(), np.array([5.0, 5.1, 5.2]))
        np.testing.assert_array_equal(extractor.time_to_sample_indices(np.array([5.0, 5.1, 5.2])), np.array([0, 1, 2]))

    def test_shift_times_without_time_vector(self):
        """A shift moves the timestamps computed from the sampling frequency, which stay without a time vector."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0
        )
        extractor.shift_times(2.5)

        self.assertFalse(extractor.has_time_vector())
        self.assertEqual(extractor.get_start_time(), 2.5)
        np.testing.assert_array_almost_equal(extractor.get_timestamps(), np.array([2.5, 2.6, 2.7]))
        np.testing.assert_array_equal(extractor.time_to_sample_indices(np.array([2.5, 2.6, 2.7])), np.array([0, 1, 2]))

    def test_shift_times_with_set_times(self):
        """A shift is added to the times given to set_times, and the stored array is not changed."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0
        )
        times = np.array([10.0, 10.5, 11.0])
        extractor.set_times(times)
        extractor.shift_times(2.0)

        np.testing.assert_array_equal(extractor.get_timestamps(), times + 2.0)
        np.testing.assert_array_equal(extractor._times, times)
        self.assertEqual(extractor.get_start_time(), 12.0)
        np.testing.assert_array_equal(extractor.time_to_sample_indices(np.array([12.2, 12.7])), np.array([0, 1]))

    def test_shift_times_with_native_timestamps(self):
        """A shift is added to the native timestamps when they are read."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0, native_timestamps="unevenly_spaced"
        )
        native_timestamps = extractor.get_native_timestamps()
        extractor.shift_times(2.5)

        np.testing.assert_array_equal(extractor.get_timestamps(), native_timestamps + 2.5)
        np.testing.assert_array_equal(extractor.get_native_timestamps(), native_timestamps)
        self.assertEqual(extractor.get_start_time(), native_timestamps[0] + 2.5)

    def test_shifts_add_up_and_set_times_resets_them(self):
        """Two shifts add, and the times given to set_times afterwards are the times returned."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0
        )
        extractor.shift_times(1.0)
        extractor.shift_times(2.0)
        self.assertEqual(extractor.get_start_time(), 3.0)

        times = np.array([10.0, 10.5, 11.0])
        extractor.set_times(times)
        np.testing.assert_array_equal(extractor.get_timestamps(), times)
        self.assertEqual(extractor.get_start_time(), 10.0)

    def test_time_to_sample_indices_with_sampling_frequency(self):
        """Round-trip: get_timestamps -> time_to_sample_indices should recover original indices."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=4, sampling_frequency=20.0
        )
        sample_indices = np.array([0, 1, 2])
        times = extractor.get_timestamps()
        recovered = extractor.time_to_sample_indices(times)
        np.testing.assert_array_equal(recovered, sample_indices)

    def test_time_to_sample_indices_with_set_times(self):
        """When an explicit time vector is set, time_to_sample_indices should use it."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=4, num_rows=2, num_columns=2, sampling_frequency=10.0
        )
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
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0
        )
        native_times = np.array([0.0, 0.1, 0.2])
        extractor.get_native_timestamps = lambda: native_times
        extractor._has_native_timestamps = lambda: True

        # _times is None but native timestamps exist
        assert extractor._times is None
        result = extractor.time_to_sample_indices(np.array([0.0, 0.1, 0.2]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_time_to_sample_indices_with_set_times_and_native_timestamps(self):
        """Times set with set_times should take precedence over native timestamps and survive the call."""
        extractor = generate_dummy_segmentation_extractor(
            num_samples=3, num_rows=2, num_columns=2, sampling_frequency=10.0, native_timestamps="evenly_spaced"
        )
        set_times = np.array([100.0, 100.1, 100.2])
        extractor.set_times(set_times)

        result = extractor.time_to_sample_indices(np.array([100.0, 100.1, 100.2]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))
        np.testing.assert_array_equal(extractor.get_timestamps(), set_times)


if __name__ == "__main__":
    unittest.main()
