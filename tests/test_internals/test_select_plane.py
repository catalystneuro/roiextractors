"""Tests for ImagingExtractor.select_plane()."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors.testing import generate_dummy_imaging_extractor


class TestSelectPlaneBasic:
    """Basic behavior of select_plane on a volumetric imaging extractor."""

    def test_returns_planar_extractor(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        planar = volumetric.select_plane(2)

        assert planar.is_volumetric is False
        assert planar.get_image_shape() == (20, 15)
        assert planar.get_sample_shape() == (20, 15)

    def test_get_series_shape_is_3d(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        planar = volumetric.select_plane(2)

        series = planar.get_series()
        assert series.shape == (10, 20, 15)

    def test_get_series_data_correctness(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        planar = volumetric.select_plane(2)

        expected = volumetric.get_series()[..., 2]
        assert_array_equal(planar.get_series(), expected)

    def test_get_series_with_range(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        planar = volumetric.select_plane(1)

        partial = planar.get_series(start_sample=2, end_sample=7)
        assert partial.shape == (5, 20, 15)
        expected = volumetric.get_series(start_sample=2, end_sample=7)[..., 1]
        assert_array_equal(partial, expected)

    def test_get_samples(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        planar = volumetric.select_plane(2)

        sample_indices = [0, 3, 7]
        samples = planar.get_samples(sample_indices=sample_indices)
        assert samples.shape == (3, 20, 15)

        expected = volumetric.get_samples(sample_indices=sample_indices)[..., 2]
        assert_array_equal(samples, expected)

    def test_properties_delegate_to_parent(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        planar = volumetric.select_plane(2)

        assert planar.get_num_samples() == volumetric.get_num_samples()
        assert planar.get_sampling_frequency() == volumetric.get_sampling_frequency()
        assert planar.get_dtype() == volumetric.get_dtype()

    def test_select_each_plane(self):
        """Selecting each plane should yield distinct, correct planar data."""
        volumetric = generate_dummy_imaging_extractor(num_samples=5, num_rows=10, num_columns=8, num_planes=3)
        full_series = volumetric.get_series()

        for plane_index in range(3):
            planar = volumetric.select_plane(plane_index)
            assert_array_equal(planar.get_series(), full_series[..., plane_index])


class TestSelectPlaneValidation:
    """Validation and error cases for select_plane."""

    def test_raises_on_non_volumetric(self):
        planar = generate_dummy_imaging_extractor(num_samples=5, num_rows=10, num_columns=8)
        assert planar.is_volumetric is False

        with pytest.raises(ValueError, match="select_plane is only valid for volumetric extractors"):
            planar.select_plane(0)

    def test_raises_on_negative_index(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=5, num_rows=10, num_columns=8, num_planes=3)
        with pytest.raises(ValueError, match="plane_index .* must satisfy"):
            volumetric.select_plane(-1)

    def test_raises_on_out_of_range_index(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=5, num_rows=10, num_columns=8, num_planes=3)
        with pytest.raises(ValueError, match="plane_index .* must satisfy"):
            volumetric.select_plane(3)

    def test_single_plane_volumetric(self):
        """A volumetric extractor with num_planes=1 still allows select_plane(0)."""
        volumetric = generate_dummy_imaging_extractor(num_samples=4, num_rows=8, num_columns=8, num_planes=1)
        planar = volumetric.select_plane(0)

        assert planar.is_volumetric is False
        assert planar.get_series().shape == (4, 8, 8)


class TestSelectPlaneComposition:
    """Composability with the other slicers."""

    def test_compose_with_slice_samples_after(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        composed = volumetric.select_plane(2).slice_samples(2, 6)

        # Use explicit start/end to work around a pre-existing SampleSlicedImagingExtractor.get_series quirk
        # where end_sample=None does not clamp to the slice's _end_sample.
        got = composed.get_series(start_sample=0, end_sample=4)
        expected = volumetric.get_series()[2:6, ..., 2]
        assert_array_equal(got, expected)

    def test_compose_with_slice_samples_before(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        composed = volumetric.slice_samples(2, 6).select_plane(2)

        got = composed.get_series(start_sample=0, end_sample=4)
        expected = volumetric.get_series()[2:6, ..., 2]
        assert_array_equal(got, expected)

    def test_compose_with_slice_field_of_view(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        composed = volumetric.select_plane(2).slice_field_of_view(
            row_start=5, row_end=15, column_start=2, column_end=12
        )

        got = composed.get_series()
        expected = volumetric.get_series()[:, 5:15, 2:12, 2]
        assert_array_equal(got, expected)
        assert composed.get_image_shape() == (10, 10)

    def test_compose_field_of_view_then_select_plane(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        composed = volumetric.slice_field_of_view(row_start=5, row_end=15, column_start=2, column_end=12).select_plane(
            2
        )

        got = composed.get_series()
        expected = volumetric.get_series()[:, 5:15, 2:12, 2]
        assert_array_equal(got, expected)
        assert composed.get_image_shape() == (10, 10)


class TestSelectPlaneTimestamps:
    """Timestamp delegation: plane selection does not affect the time axis."""

    def test_set_times_inherited_from_parent(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        times = np.arange(10) + 5.5
        volumetric.set_times(times)

        planar = volumetric.select_plane(2)
        assert planar.has_time_vector()
        assert_array_equal(planar.get_timestamps(), times)

    def test_native_timestamps_delegate_to_parent(self):
        volumetric = generate_dummy_imaging_extractor(
            num_samples=10, num_rows=20, num_columns=15, num_planes=4, has_native_timestamps=True
        )
        planar = volumetric.select_plane(2)

        assert_array_equal(planar.get_native_timestamps(), volumetric.get_native_timestamps())

    def test_native_timestamps_without_native_support(self):
        volumetric = generate_dummy_imaging_extractor(num_samples=10, num_rows=20, num_columns=15, num_planes=4)
        planar = volumetric.select_plane(2)

        assert volumetric.get_native_timestamps() is None
        assert planar.get_native_timestamps() is None

    def test_native_timestamps_with_range(self):
        volumetric = generate_dummy_imaging_extractor(
            num_samples=10, num_rows=20, num_columns=15, num_planes=4, has_native_timestamps=True
        )
        planar = volumetric.select_plane(2)

        parent_subset = volumetric.get_native_timestamps(start_sample=2, end_sample=7)
        planar_subset = planar.get_native_timestamps(start_sample=2, end_sample=7)
        assert_array_equal(planar_subset, parent_subset)
