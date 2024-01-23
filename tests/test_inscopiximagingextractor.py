import numpy as np
from numpy import dtype
from numpy.testing import assert_array_equal

from roiextractors import InscopixImagingExtractor

from .setup_paths import OPHYS_DATA_PATH


def test_inscopiximagingextractor_movie_128x128x100_part1():

    file_path = OPHYS_DATA_PATH / "imaging_data" / "inscopix" / "movie_128x128x100_part1.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    assert extractor.get_num_frames() == 100
    assert extractor.get_image_size() == (128, 128)
    assert extractor.get_dtype() == dtype("float64")
    assert extractor.get_sampling_frequency() == 10.0
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_video().shape == (100, 128, 128)


def test_inscopiximagingextractor_movie_longer_than_3_min():

    file_path = OPHYS_DATA_PATH / "imaging_data" / "inscopix" / "movie_longer_than_3_min.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    assert extractor.get_num_frames() == 1248
    assert extractor.get_image_size() == (33, 29)
    assert extractor.get_dtype() == dtype("int64")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 5.5563890139076415)
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_video().shape == (1248, 33, 29)


def test_inscopiximagingextractor_movie_u8():

    file_path = OPHYS_DATA_PATH / "imaging_data" / "inscopix" / "movie_u8.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    assert extractor.get_num_frames() == 5
    assert extractor.get_image_size() == (3, 4)
    assert extractor.get_dtype() == dtype("int64")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 20.0)
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_video().shape == (5, 3, 4)
