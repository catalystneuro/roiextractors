from roiextractors.tools.testing import generate_mock_video, generate_mock_imaging_extractor
import pytest
import numpy as np
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("size", [(1, 2, 3), (3, 2, 4), (5, 3, 2)])
def test_generate_mock_video_size(size):
    video = generate_mock_video(size=size)
    assert video.shape == size


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, "uint8", "float32"])
def test_generate_mock_video_dtype(dtype):
    video = generate_mock_video(size=(3, 2, 4), dtype=dtype)
    assert video.dtype == np.dtype(dtype)


def test_generate_mock_video_seed():
    size = (1, 2, 3)
    video1 = generate_mock_video(size=size, seed=0)
    video2 = generate_mock_video(size=size, seed=0)
    video3 = generate_mock_video(size=size, seed=1)
    assert_array_equal(video1, video2)
    assert not np.array_equal(video1, video3)


@pytest.mark.parametrize("num_frames, num_rows, num_columns", [(1, 2, 3), (3, 2, 4), (5, 3, 2)])
def test_generate_mock_imaging_extractor_shape(num_frames, num_rows, num_columns):
    imaging_extractor = generate_mock_imaging_extractor(
        num_frames=num_frames, num_rows=num_rows, num_columns=num_columns
    )
    video = imaging_extractor.get_video()
    assert video.shape == (num_frames, num_rows, num_columns)


@pytest.mark.parametrize("sampling_frequency", [10.0, 20.0, 30.0])
def test_generate_mock_imaging_extractor_sampling_frequency(sampling_frequency):
    imaging_extractor = generate_mock_imaging_extractor(sampling_frequency=sampling_frequency)
    assert imaging_extractor.get_sampling_frequency() == sampling_frequency


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, "uint8", "float32"])
def test_generate_mock_imaging_extractor_dtype(dtype):
    imaging_extractor = generate_mock_imaging_extractor(dtype=dtype)
    assert imaging_extractor.get_dtype() == np.dtype(dtype)


def test_generate_mock_imaging_extractor_seed():
    imaging_extractor1 = generate_mock_imaging_extractor(seed=0)
    imaging_extractor2 = generate_mock_imaging_extractor(seed=0)
    imaging_extractor3 = generate_mock_imaging_extractor(seed=1)
    assert_array_equal(imaging_extractor1.get_video(), imaging_extractor2.get_video())
    assert not np.array_equal(imaging_extractor1.get_video(), imaging_extractor3.get_video())
