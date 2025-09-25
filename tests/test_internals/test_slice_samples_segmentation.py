import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors.testing import generate_dummy_segmentation_extractor


class TestSampleSlicedTimestampInheritance:
    """Test timestamp inheritance behavior in SampleSlicedSegmentationExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_samples = 10
        self.start_sample = 2
        self.end_sample = 7
        self.parent_extractor = generate_dummy_segmentation_extractor(
            num_samples=self.num_samples, num_rows=5, num_columns=4
        )

    def test_sliced_times_independence_from_parent_modifications(self):
        """Test that sliced extractor's timestamps are independent from parent modifications.

        This test verifies that when a parent extractor's timestamps are modified using set_times(),
        the sliced extractor maintains its original timestamp values and is not affected by the
        parent's changes. This ensures that sliced extractors maintain data integrity even when
        the parent data is modified after the slice is created.

        The test creates a sliced extractor, then modifies the parent's timestamps, and verifies
        that the sliced extractor retains its original timestamp values.
        """
        # Set initial parent times
        original_parent_times = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9])
        self.parent_extractor.set_times(original_parent_times)

        # Create sliced extractor
        sliced_extractor = self.parent_extractor.slice_samples(
            start_sample=self.start_sample, end_sample=self.end_sample
        )

        # Verify initial inheritance
        expected_sliced_times = original_parent_times[self.start_sample : self.end_sample]
        assert_array_equal(sliced_extractor.get_timestamps(), expected_sliced_times)

        # Modify parent times - this should NOT affect the sliced extractor
        new_parent_times = np.array([20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9])
        self.parent_extractor.set_times(new_parent_times)

        # Sliced extractor should still have the original sliced times, not the new ones
        assert_array_equal(sliced_extractor.get_timestamps(), expected_sliced_times)
        # And definitely not the new parent times
        with pytest.raises(AssertionError):
            assert_array_equal(sliced_extractor.get_timestamps(), new_parent_times[self.start_sample : self.end_sample])

    def test_parent_times_independence_from_sliced_modifications(self):
        """Test that parent's timestamps are independent from sliced extractor modifications.

        This test verifies that when a sliced extractor's timestamps are modified using set_times(),
        the parent extractor's timestamps remain unchanged. This ensures that modifications to
        child sliced extractors do not inadvertently affect the parent's data, maintaining
        proper data encapsulation and preventing unexpected side effects.

        The test creates a sliced extractor, modifies its timestamps, and verifies that the
        parent extractor's timestamps remain at their original values.
        """
        # Set initial parent times
        original_parent_times = np.array([30.0, 30.1, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7, 30.8, 30.9])
        self.parent_extractor.set_times(original_parent_times.copy())  # Use copy to be explicit

        # Create sliced extractor
        sliced_extractor = self.parent_extractor.slice_samples(
            start_sample=self.start_sample, end_sample=self.end_sample
        )

        # Modify the sliced extractor's times using proper API
        new_sliced_times = np.array([999.0, 999.1, 999.2, 999.3, 999.4])
        sliced_extractor.set_times(new_sliced_times)

        # Parent times should remain unchanged
        assert_array_equal(self.parent_extractor.get_timestamps(), original_parent_times)
        # The parent should NOT have the modified values
        assert not np.any(self.parent_extractor.get_timestamps() == 999.0)

    def test_multiple_sliced_extractors_independence(self):
        """Test that multiple sliced extractors from the same parent are independent from each other.

        This test verifies that when multiple sliced extractors are created from the same parent,
        modifications to one sliced extractor do not affect the other sliced extractors or the
        parent extractor. This ensures proper isolation of timestamp data across different views
        of the same parent data.

        The test creates two non-overlapping sliced extractors, modifies timestamps in one,
        and verifies that the other slice and the parent remain unaffected.
        """
        # Set parent times
        parent_times = np.array([40.0, 40.1, 40.2, 40.3, 40.4, 40.5, 40.6, 40.7, 40.8, 40.9])
        self.parent_extractor.set_times(parent_times)

        # Create two different sliced extractors
        slice1 = self.parent_extractor.slice_samples(start_sample=1, end_sample=4)
        slice2 = self.parent_extractor.slice_samples(start_sample=5, end_sample=8)

        # Modify slice1's times using proper API
        new_slice1_times = np.array([777.0, 777.1, 777.2])
        slice1.set_times(new_slice1_times)

        # slice2's times should be unaffected
        expected_slice2_times = parent_times[5:8]
        assert_array_equal(slice2.get_timestamps(), expected_slice2_times)

        # Parent should also be unaffected
        assert_array_equal(self.parent_extractor.get_timestamps(), parent_times)
