from roiextractors.tools.testing import generate_mock_video
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
