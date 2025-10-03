import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors.testing import generate_dummy_imaging_extractor


def test_sample_slicing_imaging_times():
    num_samples = 10
    timestamp_shift = 7.1
    times = np.array(range(num_samples)) + timestamp_shift
    start_sample, end_sample = 2, 7

    imaging_extractor = generate_dummy_imaging_extractor(
        num_frames=num_samples, num_rows=5, num_columns=4, num_channels=1
    )
    imaging_extractor.set_times(times=times)

    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=start_sample, end_sample=end_sample)
    assert_array_equal(
        sample_sliced_imaging.sample_indices_to_time(
            sample_indices=np.array([idx for idx in range(sample_sliced_imaging.get_num_samples())])
        ),
        times[start_sample:end_sample],
    )


def test_get_image_shape():
    imaging_extractor = generate_dummy_imaging_extractor(num_frames=10, num_rows=5, num_columns=4, num_channels=1)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_image_shape() == (5, 4)


def test_get_num_samples():
    imaging_extractor = generate_dummy_imaging_extractor(num_frames=10, num_rows=5, num_columns=4, num_channels=1)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_num_samples() == 5


def test_get_sampling_frequency():
    imaging_extractor = generate_dummy_imaging_extractor(num_frames=10, num_rows=5, num_columns=4, num_channels=1)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_sampling_frequency() == 30.0


def test_get_channel_names():
    imaging_extractor = generate_dummy_imaging_extractor(num_frames=10, num_rows=5, num_columns=4, num_channels=1)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_channel_names() == ["channel_num_0"]


def test_get_samples_assertion():
    imaging_extractor = generate_dummy_imaging_extractor(num_frames=10, num_rows=5, num_columns=4, num_channels=1)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    with pytest.raises(AssertionError, match="'sample_indices' range beyond number of available samples!"):
        sample_sliced_imaging.get_samples(sample_indices=[6])


def test_get_frames():
    imaging_extractor = generate_dummy_imaging_extractor(num_frames=10, num_rows=5, num_columns=4, num_channels=1)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert_array_equal(
        sample_sliced_imaging.get_frames(frame_idxs=[2, 4]),
        imaging_extractor.get_frames(frame_idxs=[4, 6]),
    )


def test_get_dtype():
    imaging_extractor = generate_dummy_imaging_extractor(num_frames=10, num_rows=5, num_columns=4, num_channels=1)
    sample_sliced_imaging = imaging_extractor.slice_samples(start_sample=2, end_sample=7)
    assert sample_sliced_imaging.get_dtype() == imaging_extractor.get_dtype()


def test_has_time_vector_inherits_from_parent():
    """Test that sliced extractor's has_time_vector delegates to parent."""
    imaging_extractor = generate_dummy_imaging_extractor(num_frames=10, num_rows=5, num_columns=4, num_channels=1)

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
