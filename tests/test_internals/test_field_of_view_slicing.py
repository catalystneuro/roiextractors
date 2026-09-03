import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors.testing import generate_dummy_imaging_extractor


class TestFieldOfViewSlicing2D:
    """Tests for field of view slicing on 2D imaging extractors."""

    def test_basic_fov_slicing(self):
        """Test basic FOV slicing with explicit bounds."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        fov_sliced = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        assert fov_sliced.get_image_shape() == (40, 40)
        assert fov_sliced.get_num_samples() == 10
        assert not fov_sliced.is_volumetric

    def test_fov_slicing_with_defaults(self):
        """Test FOV slicing with default parameters (no slicing)."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=50, num_columns=60)
        fov_sliced = imaging.slice_field_of_view()

        assert fov_sliced.get_image_shape() == (50, 60)
        assert_array_equal(fov_sliced.get_series(), imaging.get_series())

    def test_fov_slicing_partial_defaults(self):
        """Test FOV slicing with some default parameters."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)

        # Only specify row slicing
        row_sliced = imaging.slice_field_of_view(row_start=10, row_end=50)
        assert row_sliced.get_image_shape() == (40, 80)

        # Only specify column slicing
        col_sliced = imaging.slice_field_of_view(column_start=20, column_end=60)
        assert col_sliced.get_image_shape() == (100, 40)

    def test_get_series_shape(self):
        """Test that get_series returns correctly shaped data."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        fov_sliced = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        series = fov_sliced.get_series()
        assert series.shape == (10, 40, 40)

        # Test with sample slicing
        partial_series = fov_sliced.get_series(start_sample=2, end_sample=7)
        assert partial_series.shape == (5, 40, 40)

    def test_get_series_data_correctness(self):
        """Test that get_series returns correctly sliced data."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        fov_sliced = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        # Get data from both extractors
        original_series = imaging.get_series()
        sliced_series = fov_sliced.get_series()

        # Verify the sliced data matches the manual slicing of original
        expected = original_series[:, 20:60, 10:50]
        assert_array_equal(sliced_series, expected)

    def test_get_samples(self):
        """Test get_samples with FOV slicing."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        fov_sliced = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        samples = fov_sliced.get_samples(sample_indices=[2, 5, 8])
        assert samples.shape == (3, 40, 40)

        # Verify correctness
        original_samples = imaging.get_samples(sample_indices=[2, 5, 8])
        expected = original_samples[:, 20:60, 10:50]
        assert_array_equal(samples, expected)

    def test_properties_preservation(self):
        """Test that properties are preserved from parent extractor."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        fov_sliced = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        assert fov_sliced.get_num_samples() == imaging.get_num_samples()
        assert fov_sliced.get_sampling_frequency() == imaging.get_sampling_frequency()
        assert fov_sliced.get_dtype() == imaging.get_dtype()

    def test_set_timestamps_preservation(self):
        """Test that properties including timestamps are preserved correctly."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        times = np.arange(10) + 5.5
        imaging.set_times(times)

        fov_sliced = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        assert fov_sliced.has_time_vector()
        assert_array_equal(fov_sliced.get_timestamps(), times)

    def test_native_timestamps_preservation(self):
        """Test that native timestamps are delegated to parent."""
        # Test with extractor that has native timestamps
        imaging = generate_dummy_imaging_extractor(
            num_samples=10, num_rows=100, num_columns=80, native_timestamps="evenly_spaced"
        )
        fov_sliced = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        # Should delegate to parent and return same timestamps
        parent_timestamps = imaging.get_native_timestamps()
        sliced_timestamps = fov_sliced.get_native_timestamps()
        assert_array_equal(sliced_timestamps, parent_timestamps)

        # Also test with extractor that has no native timestamps
        imaging_no_ts = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        fov_sliced_no_ts = imaging_no_ts.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)
        assert imaging_no_ts.get_native_timestamps() is None
        assert fov_sliced_no_ts.get_native_timestamps() is None

    def test_native_timestamps_with_timestamps(self):
        """Test that native timestamps work correctly when present."""
        # Create extractor with native timestamps
        imaging = generate_dummy_imaging_extractor(
            num_samples=10, num_rows=100, num_columns=80, native_timestamps="evenly_spaced"
        )
        fov_sliced = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        # Get timestamps from both
        parent_timestamps = imaging.get_native_timestamps()
        sliced_timestamps = fov_sliced.get_native_timestamps()

        # Should be the same (FOV slicing doesn't affect temporal data)
        assert_array_equal(sliced_timestamps, parent_timestamps)

        # Test with range
        parent_subset = imaging.get_native_timestamps(start_sample=2, end_sample=7)
        sliced_subset = fov_sliced.get_native_timestamps(start_sample=2, end_sample=7)
        assert_array_equal(sliced_subset, parent_subset)

    def test_composition_with_sample_slicing(self):
        """Test composing FOV slicing with temporal slicing."""
        imaging = generate_dummy_imaging_extractor(num_samples=100, num_rows=100, num_columns=80)

        # FOV then temporal
        subset1 = imaging.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)
        subset1 = subset1.slice_samples(start_sample=10, end_sample=30)

        assert subset1.get_image_shape() == (40, 40)
        assert subset1.get_num_samples() == 20

        # Temporal then FOV
        subset2 = imaging.slice_samples(start_sample=10, end_sample=30)
        subset2 = subset2.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        assert subset2.get_image_shape() == (40, 40)
        assert subset2.get_num_samples() == 20

        # Both should give same result
        assert_array_equal(subset1.get_series(), subset2.get_series())

    def test_double_fov_slicing(self):
        """Test chaining multiple FOV slicing operations."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)

        # First slice
        fov1 = imaging.slice_field_of_view(row_start=10, row_end=90, column_start=10, column_end=70)
        assert fov1.get_image_shape() == (80, 60)

        # Second slice (relative to first)
        fov2 = fov1.slice_field_of_view(row_start=10, row_end=50, column_start=10, column_end=40)
        assert fov2.get_image_shape() == (40, 30)

        # Verify data correctness
        original = imaging.get_series()
        double_sliced = fov2.get_series()
        expected = original[:, 20:60, 20:50]  # Combined offsets
        assert_array_equal(double_sliced, expected)


class TestFieldOfViewSlicingValidation:
    """Tests for validation and edge cases."""

    def test_invalid_row_bounds(self):
        """Test validation of row bounds."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)

        # row_start >= parent height
        with pytest.raises(AssertionError, match="row_start.*must be.*< parent height"):
            imaging.slice_field_of_view(row_start=100, row_end=110)

        # row_end > parent height
        with pytest.raises(AssertionError, match="row_end.*must be.*<= parent height"):
            imaging.slice_field_of_view(row_start=10, row_end=101)

        # row_end <= row_start
        with pytest.raises(AssertionError, match="row_end.*must be greater than.*row_start"):
            imaging.slice_field_of_view(row_start=50, row_end=50)

        with pytest.raises(AssertionError, match="row_end.*must be greater than.*row_start"):
            imaging.slice_field_of_view(row_start=60, row_end=50)

    def test_invalid_column_bounds(self):
        """Test validation of column bounds."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)

        # column_start >= parent width
        with pytest.raises(AssertionError, match="column_start.*must be.*< parent width"):
            imaging.slice_field_of_view(column_start=80, column_end=90)

        # column_end > parent width
        with pytest.raises(AssertionError, match="column_end.*must be.*<= parent width"):
            imaging.slice_field_of_view(column_start=10, column_end=81)

        # column_end <= column_start
        with pytest.raises(AssertionError, match="column_end.*must be greater than.*column_start"):
            imaging.slice_field_of_view(column_start=50, column_end=50)

        with pytest.raises(AssertionError, match="column_end.*must be greater than.*column_start"):
            imaging.slice_field_of_view(column_start=60, column_end=50)

    def test_edge_case_single_pixel_row(self):
        """Test slicing to single row."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        fov_sliced = imaging.slice_field_of_view(row_start=50, row_end=51, column_start=0, column_end=80)

        assert fov_sliced.get_image_shape() == (1, 80)

    def test_edge_case_single_pixel_column(self):
        """Test slicing to single column."""
        imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80)
        fov_sliced = imaging.slice_field_of_view(row_start=0, row_end=100, column_start=40, column_end=41)

        assert fov_sliced.get_image_shape() == (100, 1)


class TestFieldOfViewSlicingVolumetric:
    """Tests for field of view slicing on volumetric imaging extractors."""

    def test_volumetric_basic_fov_slicing(self):
        """Test basic FOV slicing on volumetric extractor."""
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80, num_planes=5)

        fov_sliced = volumetric.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        assert fov_sliced.get_image_shape() == (40, 40)
        assert fov_sliced.get_num_planes() == 5
        assert fov_sliced.get_volume_shape() == (40, 40, 5)
        assert fov_sliced.is_volumetric

    def test_volumetric_get_series_shape(self):
        """Test that get_series returns correct shape for volumetric data."""
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80, num_planes=5)

        fov_sliced = volumetric.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        series = fov_sliced.get_series()
        # Shape should be (samples, rows, columns, planes)
        assert series.shape == (10, 40, 40, 5)

    def test_volumetric_get_series_data_correctness(self):
        """Test that volumetric FOV slicing returns correct data."""
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80, num_planes=3)

        fov_sliced = volumetric.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        # Get data from both
        original_series = volumetric.get_series()
        sliced_series = fov_sliced.get_series()

        # Verify slicing is correct
        expected = original_series[:, 20:60, 10:50, :]
        assert_array_equal(sliced_series, expected)

    def test_volumetric_get_samples(self):
        """Test get_samples on volumetric FOV sliced extractor."""
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80, num_planes=3)

        fov_sliced = volumetric.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        samples = fov_sliced.get_samples(sample_indices=[1, 3, 5])
        assert samples.shape == (3, 40, 40, 3)

        # Verify correctness
        original_samples = volumetric.get_samples(sample_indices=[1, 3, 5])
        expected = original_samples[:, 20:60, 10:50, :]
        assert_array_equal(samples, expected)

    def test_volumetric_temporal_subsetting(self):
        """Test composing FOV slicing with temporal subsetting on volumetric data."""
        volumetric = generate_dummy_imaging_extractor(num_samples=100, num_rows=100, num_columns=80, num_planes=5)

        # Apply FOV slicing
        subset = volumetric.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        # Use get_series() for temporal subsetting
        series = subset.get_series(start_sample=10, end_sample=30)
        assert series.shape == (20, 40, 40, 5)

    def test_volumetric_properties_are_passed_through(self):
        """Test that volumetric properties are preserved."""
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=100, num_columns=80, num_planes=5)

        fov_sliced = volumetric.slice_field_of_view(row_start=20, row_end=60, column_start=10, column_end=50)

        assert fov_sliced.get_num_samples() == volumetric.get_num_samples()
        assert fov_sliced.get_sampling_frequency() == volumetric.get_sampling_frequency()
        assert fov_sliced.get_dtype() == volumetric.get_dtype()
        assert fov_sliced.is_volumetric == volumetric.is_volumetric
