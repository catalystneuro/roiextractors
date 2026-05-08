import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors.testing import generate_dummy_imaging_extractor


def test_sample_slicing_imaging_times():
    num_samples = 10
    timestamp_shift = 7.1
    times = np.array(range(num_samples)) + timestamp_shift
    start_sample, end_sample = 2, 7

    imaging_extractor = generate_dummy_imaging_extractor(num_samples=num_samples, num_rows=5, num_columns=4)
    imaging_extractor.set_times(times=times)

    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=start_sample, end_sample=end_sample)
    assert_array_equal(
        sample_sliced_imaging.get_timestamps(),
        times[start_sample:end_sample],
    )


def test_get_image_shape():
    imaging_extractor = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_image_shape() == (5, 4)


def test_get_num_samples():
    imaging_extractor = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_num_samples() == 5


def test_get_sampling_frequency():
    imaging_extractor = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_sampling_frequency() == 30.0


def test_get_samples_assertion():
    imaging_extractor = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    with pytest.raises(AssertionError, match="'sample_indices' range beyond number of available samples!"):
        sample_sliced_imaging.get_samples(sample_indices=[6])


def test_get_dtype():
    imaging_extractor = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_dtype() == imaging_extractor.get_dtype()


def test_has_time_vector_inherits_from_parent():
    """Test that sliced extractor's has_time_vector delegates to parent."""
    imaging_extractor = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)

    # Parent without time vector
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.has_time_vector() == imaging_extractor.has_time_vector()
    assert not sample_sliced_imaging.has_time_vector()

    # Parent with time vector
    times = np.array(range(10)) + 5.5
    imaging_extractor.set_times(times=times)
    sample_sliced_imaging_with_times = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging_with_times.has_time_vector() == imaging_extractor.has_time_vector()
    assert sample_sliced_imaging_with_times.has_time_vector()


def test_get_series_default_args_clamps_to_slice_bounds():
    """Regression test for #585.

    `SampleSlicedImagingExtractor.get_series()` with no arguments must return only
    the samples within the slice's [start_sample, end_sample) range. Previously,
    end_sample=None passed straight through to the parent, which returned all samples
    from the shifted start to the parent's full end, silently producing too many samples.
    """
    imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)
    sliced = imaging.slice_samples(start_sample=2, end_sample=6)

    # No-arg call should respect the slice bounds.
    series = sliced.get_series()
    assert series.shape == (4, 5, 4)
    assert_array_equal(series, imaging.get_series()[2:6])

    # Explicit end_sample beyond the slice's own length is allowed; bounds are caller's responsibility.
    # But default (None) must clamp to the slice.
    full_via_default = sliced.get_series()
    full_via_explicit = sliced.get_series(start_sample=0, end_sample=4)
    assert_array_equal(full_via_default, full_via_explicit)


def test_get_series_partial_default_clamps_to_slice_bounds():
    """Default end_sample with explicit start_sample should still clamp to the slice."""
    imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)
    sliced = imaging.slice_samples(start_sample=2, end_sample=6)

    series = sliced.get_series(start_sample=1)
    assert series.shape == (3, 5, 4)
    assert_array_equal(series, imaging.get_series()[3:6])


def test_get_series_default_start_clamps_correctly():
    """Default start_sample with explicit end_sample should still respect the slice's start."""
    imaging = generate_dummy_imaging_extractor(num_samples=10, num_rows=5, num_columns=4)
    sliced = imaging.slice_samples(start_sample=2, end_sample=6)

    series = sliced.get_series(end_sample=3)
    assert series.shape == (3, 5, 4)
    assert_array_equal(series, imaging.get_series()[2:5])
