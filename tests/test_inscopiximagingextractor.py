import numpy as np
import pytest
import platform
from numpy import dtype
from numpy.testing import assert_array_equal

from roiextractors import InscopixImagingExtractor

from tests.setup_paths import OPHYS_DATA_PATH

# Warn about macOS ARM64 environment
pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="For macOS ARM64, please use a special conda environment setup refer to isx ReadMe for instructions.",
)


def test_inscopiximagingextractor_movie_128x128x100_part1():
    """
     Test with a small movie file (128x128 resolution, 100 frames).

     File: movie_128x128x100_part1.isxd
     Metadata:
    - Source: Movie file (Inscopix isxcore v1.8.0)
     - 100 samples (frames)
     - Frame rate: 10.0 Hz
     - Volumetric: False
     - Frame shape: (128, 128) pixels
     - Pixel size: 3 µm x 3 µm
     - Spatial offset (top-left): (0, 0) in global image space
     - Channels: 1 (isx format supports only single channel)

     This test verifies that the extractor correctly:
     1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
     2. Retrieves the video data and raw data with correct shape and type.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "movie_128x128x100_part1.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    assert extractor.get_num_frames() == 100
    assert extractor.get_image_size() == (128, 128)
    assert extractor.get_dtype() is dtype("float32")
    assert extractor.get_sampling_frequency() == 10.0
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_video().shape == (100, 128, 128)
    assert extractor.get_frames(frame_idxs=[0], channel=0).dtype is extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (100, 128, 128)
    assert raw_data.dtype == extractor.get_dtype()


def test_inscopiximagingextractor_movie_longer_than_3_min():
    """
    Test with a movie longer than 3 minutes.

    File: movie_longer_than_3_min.isxd
    Metadata:
    - Source: Movie file (Inscopix isxcore v1.8.1)
    - 1248 samples (frames)
    - Frame rate: ~5.56 Hz (Period = 179973 µs)
    - Volumetric: True (3 active planes: 170 µm, 370 µm, 570 µm)
    - Frame shape: (33, 29) pixels
    - Pixel size: 72 µm x 72 µm
    - Spatial offset (top-left): (333, 0) in global image space
    - Spatial downsampling: 12x
    - Temporal downsampling: 1x
    - Channels: 1 (isx format supports only single channel)
    - Microscope: Inscopix NVista3, Serial: FA-11092903
    - LED Power: 0.4, Exposure: 20 ms, Gain: 5.9
    - Acquisition start time: 2019-10-07
    - Session name: 4D_SAAV_PFC_IM7_20191007

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
    2. Retrieves the video data and raw data with correct shape and type.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "movie_longer_than_3_min.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    assert extractor.get_num_frames() == 1248
    assert extractor.get_image_size() == (33, 29)
    assert extractor.get_dtype() is dtype("uint16")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 5.5563890139076415)
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_video().shape == (1248, 33, 29)
    assert extractor.get_frames(frame_idxs=[0], channel=0).dtype is extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (1248, 33, 29)
    assert raw_data.dtype == extractor.get_dtype()


def test_inscopiximagingextractor_movie_u8():
    """
    Test loading a minimal valid Inscopix movie file with uint8-encoded pixel data.


    File: movie_u8.isxd
    Metadata:
    - Source: Movie file (Inscopix isxcore v1.9.5)
    - 5 samples (frames)
    - Frame rate: 20.0 Hz
    - Volumetric: False
    - Frame shape: (3, 4) pixels
    - Pixel size: 3 µm x 3 µm
    - Channels: 1 (isx format supports only single channel)

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
    2. Retrieves the video data and raw data with correct shape and type.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "movie_u8.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    assert extractor.get_num_frames() == 5
    assert extractor.get_image_size() == (3, 4)
    assert extractor.get_dtype() is dtype("uint8")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 20.0)
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_video().shape == (5, 3, 4)
    assert extractor.get_frames(frame_idxs=[0], channel=0).dtype is extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (5, 3, 4)
    assert raw_data.dtype == extractor.get_dtype()
