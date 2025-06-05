import numpy as np
import pytest
import platform
from numpy import dtype
from numpy.testing import assert_array_equal
from datetime import datetime

from roiextractors import InscopixImagingExtractor

from tests.setup_paths import OPHYS_DATA_PATH

# Skip all tests in this file on macOS
pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="The isx package is currently not natively supported on macOS with Apple Silicon. "
    "Installation instructions can be found at: "
    "https://github.com/inscopix/pyisx?tab=readme-ov-file#install",
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
     - No acquisition info available

     This test verifies that the extractor correctly:
     1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
     2. Retrieves the video data and raw data with correct shape and type.
     3. Tests metadata extraction methods for files with minimal acquisition info.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "movie_128x128x100_part1.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    # Basic video properties
    assert extractor.get_num_samples() == 100
    assert extractor.get_image_shape() == (128, 128)
    assert extractor.get_dtype() is dtype("float32")
    assert extractor.get_sampling_frequency() == 10.0
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_series().shape == (100, 128, 128)
    assert extractor.get_frames(frame_idxs=[0], channel=0).dtype is extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (100, 128, 128)
    assert raw_data.dtype == extractor.get_dtype()

    # Test session start time 
    assert extractor.get_session_start_time() == datetime(1970, 1, 1, 0, 0, 0)

    # Test session info for file with no acquisition info
    session_info = extractor.get_session_info()
    assert "session_name" not in session_info
    assert "experimenter_name" not in session_info

    device_info = extractor.get_device_info()
    assert isinstance(device_info, dict)
    assert "field_of_view_pixels" in device_info
    assert device_info["field_of_view_pixels"] == (128, 128)
    assert "device_name" not in device_info
    assert "device_serial_number" not in device_info
    assert "acquisition_software_version" not in device_info

    subject_info = extractor.get_subject_info()
    assert isinstance(subject_info, dict)
    assert len(subject_info) == 0

    probe_info = extractor.get_probe_info()
    assert isinstance(probe_info, dict)
    assert len(probe_info) == 0


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
    - Acquisition start time: 2019-10-07 16:22:01.524186
    - Session name: 4D_SAAV_PFC_IM7_20191007

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
    2. Retrieves the video data and raw data with correct shape and type.
    3. Tests comprehensive metadata extraction for files with rich acquisition info.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "movie_longer_than_3_min.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    # Basic video properties
    assert extractor.get_num_samples() == 1248
    assert extractor.get_image_shape() == (33, 29)
    assert extractor.get_dtype() is dtype("uint16")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 5.5563890139076415)
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_series().shape == (1248, 33, 29)
    assert extractor.get_frames(frame_idxs=[0], channel=0).dtype is extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (1248, 33, 29)
    assert raw_data.dtype == extractor.get_dtype()

    # Test session start time
    assert extractor.get_session_start_time() == datetime(2019, 10, 7, 16, 22, 1, 524186)
    

    # Test session info
    session_info = extractor.get_session_info()
    start_time = session_info.get("start_time")
    assert session_info["session_name"] == "4D_SAAV_PFC_IM7_20191007"
    assert "experimenter_name" not in session_info

    # Test device info
    device_info = extractor.get_device_info()
    assert isinstance(device_info, dict)
    assert device_info["device_name"] == "NVista3"
    assert device_info["device_serial_number"] == "FA-11092903"
    assert device_info["acquisition_software_version"] == "1.3.0"
    assert device_info["field_of_view_pixels"] == (33, 29)
    assert device_info["exposure_time_ms"] == 20
    assert device_info["microscope_focus"] == 1000
    assert device_info["microscope_gain"] == 5.9
    assert device_info["efocus"] == 370
    assert device_info["led_power_ex_mw_per_mm2"] == 0.4
    assert device_info["led_power_og_mw_per_mm2"] == 0.2

    # Test subject info - only sex should be included (other fields are empty or 0)
    subject_info = extractor.get_subject_info()
    assert isinstance(subject_info, dict)
    assert subject_info["sex"] == "m"
    assert "animal_id" not in subject_info  
    assert "species" not in subject_info    
    assert "weight" not in subject_info    
    assert "date_of_birth" not in subject_info  
    assert "description" not in subject_info    
    
    # Test probe info - should be empty since all probe values are 0/"none"/"None"
    probe_info = extractor.get_probe_info()
    assert isinstance(probe_info, dict)
    assert len(probe_info) == 0  

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
    - No acquisition info available

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
    2. Retrieves the video data and raw data with correct shape and type.
    3. Tests metadata extraction methods for files with minimal acquisition info.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "movie_u8.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    # Basic video properties
    assert extractor.get_num_samples() == 5
    assert extractor.get_image_shape() == (3, 4)
    assert extractor.get_dtype() is dtype("uint8")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 20.0)
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_num_channels() == 1
    assert extractor.get_series().shape == (5, 3, 4)
    assert extractor.get_frames(frame_idxs=[0], channel=0).dtype is extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (5, 3, 4)
    assert raw_data.dtype == extractor.get_dtype()

    # Test session start time 
    assert extractor.get_session_start_time() == datetime(1970, 1, 1, 0, 0, 0)

    # Test session info for file with no acquisition info
    session_info = extractor.get_session_info()
    assert "session_name" not in session_info
    assert "experimenter_name" not in session_info

    # Device info should have minimal content for this file
    device_info = extractor.get_device_info()
    assert isinstance(device_info, dict)
    assert "field_of_view_pixels" in device_info
    assert device_info["field_of_view_pixels"] == (3, 4)
    assert "device_name" not in device_info
    assert "device_serial_number" not in device_info
    assert "acquisition_software_version" not in device_info

    # Subject info should be empty for this file (no acquisition info)
    subject_info = extractor.get_subject_info()
    assert isinstance(subject_info, dict)
    assert len(subject_info) == 0

    # Probe info should be empty for this file (no acquisition info)
    probe_info = extractor.get_probe_info()
    assert isinstance(probe_info, dict)
    assert len(probe_info) == 0