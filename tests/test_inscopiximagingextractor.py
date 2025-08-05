import platform
import sys
from datetime import datetime

import numpy as np
import pytest
from numpy import dtype

from roiextractors import InscopixImagingExtractor
from tests.setup_paths import OPHYS_DATA_PATH

# Skip all tests in this file on macOS and Python 3.13
pytest.importorskip("isx", reason="isx package is required for these tests.")
pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="The isx package is currently not natively supported on macOS with Apple Silicon. "
    "Installation instructions can be found at: "
    "https://github.com/inscopix/pyisx?tab=readme-ov-file#install",
)
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="Tests are skipped on Python 3.13 because of incompatibility with the 'isx' module"
    "See:https://github.com/inscopix/pyisx/issues",
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
    assert extractor.get_series().shape == (100, 128, 128)

    # Test retrieving multiple samples
    samples = extractor.get_samples([0, 1, 2])
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (3, 128, 128)
    assert samples.dtype == extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (100, 128, 128)
    assert raw_data.dtype == extractor.get_dtype()

    # Test session start time
    assert extractor._get_session_start_time() == datetime(1970, 1, 1, 0, 0, 0)

    # Test session info for file with no acquisition info
    session_info = extractor._get_session_info()
    assert "session_name" not in session_info
    assert "experimenter_name" not in session_info

    device_info = extractor._get_device_info()
    assert isinstance(device_info, dict)
    assert "field_of_view_pixels" in device_info
    assert device_info["field_of_view_pixels"] == (128, 128)
    assert "device_name" not in device_info
    assert "device_serial_number" not in device_info
    assert "acquisition_software_version" not in device_info

    subject_info = extractor._get_subject_info()
    assert isinstance(subject_info, dict)
    assert len(subject_info) == 0

    probe_info = extractor._get_probe_info()
    assert isinstance(probe_info, dict)
    assert len(probe_info) == 0

    metadata = extractor._get_metadata()
    # Check that all expected keys are present
    expected_keys = ["device", "subject", "analysis", "session", "probe", "session_start_time"]
    for key in expected_keys:
        assert key in metadata


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
    assert extractor.get_series().shape == (1248, 33, 29)

    # Test retrieving multiple samples
    samples = extractor.get_samples([0, 1, 2])
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (3, 33, 29)
    assert samples.dtype == extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (1248, 33, 29)
    assert raw_data.dtype == extractor.get_dtype()

    # Test session start time
    assert extractor._get_session_start_time() == datetime(2019, 10, 7, 16, 22, 1, 524186)

    # Test session info
    session_info = extractor._get_session_info()
    start_time = session_info.get("start_time")
    assert session_info["session_name"] == "4D_SAAV_PFC_IM7_20191007"
    assert "experimenter_name" not in session_info

    # Test device info
    device_info = extractor._get_device_info()
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
    subject_info = extractor._get_subject_info()
    assert isinstance(subject_info, dict)
    assert subject_info["sex"] == "m"
    assert "animal_id" not in subject_info
    assert "species" not in subject_info
    assert "weight" not in subject_info
    assert "date_of_birth" not in subject_info
    assert "description" not in subject_info

    # Test probe info - should be empty since all probe values are 0/"none"/"None"
    probe_info = extractor._get_probe_info()
    assert isinstance(probe_info, dict)
    assert len(probe_info) == 0

    metadata = extractor._get_metadata()
    # Check that all expected keys are present
    expected_keys = ["device", "subject", "analysis", "session", "probe", "session_start_time"]
    for key in expected_keys:
        assert key in metadata


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
    assert extractor.get_series().shape == (5, 3, 4)

    # Test retrieving multiple samples
    samples = extractor.get_samples([0, 1, 2])
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (3, 3, 4)
    assert samples.dtype == extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (5, 3, 4)
    assert raw_data.dtype == extractor.get_dtype()

    # Test session start time
    assert extractor._get_session_start_time() == datetime(1970, 1, 1, 0, 0, 0)

    # Test session info for file with no acquisition info
    session_info = extractor._get_session_info()
    assert "session_name" not in session_info
    assert "experimenter_name" not in session_info

    # Device info should have minimal content for this file
    device_info = extractor._get_device_info()
    assert isinstance(device_info, dict)
    assert "field_of_view_pixels" in device_info
    assert device_info["field_of_view_pixels"] == (3, 4)
    assert "device_name" not in device_info
    assert "device_serial_number" not in device_info
    assert "acquisition_software_version" not in device_info

    # Subject info should be empty for this file (no acquisition info)
    subject_info = extractor._get_subject_info()
    assert isinstance(subject_info, dict)
    assert len(subject_info) == 0

    # Probe info should be empty for this file (no acquisition info)
    probe_info = extractor._get_probe_info()
    assert isinstance(probe_info, dict)
    assert len(probe_info) == 0

    metadata = extractor._get_metadata()
    # Check that all expected keys are present
    expected_keys = ["device", "subject", "analysis", "session", "probe", "session_start_time"]
    for key in expected_keys:
        assert key in metadata


def test_inscopiximagingextractor_multiplane_movie():
    """
    Test with a multiplane movie file featuring dual-color microscopy.

    File: multiplane_movie.isxd
    Metadata:
    - Source: Movie file (Inscopix isxcore v1.9.0)
    - 30 samples (frames)
    - Frame rate: ~7.0 Hz (Period = 142.8 ms)
    - Volumetric: True (3 active planes at focus depths: 0μm, 500μm, 900μm)
    - Frame shape: (100, 160) pixels
    - Pixel size: Standard (not specified in metadata)
    - Spatial offset: Standard (not specified in metadata)
    - Channels: 1 (isx format supports only single channel)
    - Microscope: Inscopix Dual Color, Serial: 11094108
    - LED Power: EX LED 1: 0.2 mW/mm², EX LED 2: 0.2 mW/mm²
    - Exposure: 143 ms, Focus: 1000, Gain: 2.0
    - Acquisition start time: 2022-06-15 23:41:43.403000
    - Session name: Session 20220614-102826
    - Probe: ProView DC Integrated Lens (Diameter: 0.66mm, Length: 7.5mm, Pitch: 0.5, Rotation: 180°)
    - Recording UUID: CA-11095302-0000000000-1655336502873
    - Multiplane Settings: 3D rotation (X:27°, Y:47°, Z:2°), spacing: 160μm

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
    2. Retrieves the video data and raw data with correct shape and type.
    3. Tests comprehensive metadata extraction for dual-color microscopy with probe info.
    4. Validates probe information extraction for non-empty/non-zero values.

    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "multiplane_movie.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    # Basic video properties
    assert extractor.get_num_samples() == 30
    assert extractor.get_image_shape() == (100, 160)
    assert extractor.get_dtype() is dtype("uint16")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 7.002533951423664, decimal=10)
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_series().shape == (30, 100, 160)
    assert extractor.get_samples([0]).shape == (30, 100, 160)

    # Test session start time
    assert extractor._get_session_start_time() == datetime(2022, 6, 15, 23, 41, 43, 403000)

    # Test session info
    session_info = extractor._get_session_info()
    assert session_info["session_name"] == "Session 20220614-102826"
    assert "experimenter_name" not in session_info

    # Test device info - comprehensive for dual-color microscopy
    device_info = extractor._get_device_info()
    assert isinstance(device_info, dict)
    assert device_info["device_name"] == "Dual Color"
    assert device_info["device_serial_number"] == "11094108"
    assert device_info["acquisition_software_version"] == "1.9.0"
    assert device_info["field_of_view_pixels"] == (100, 160)
    assert device_info["exposure_time_ms"] == 143
    assert device_info["microscope_focus"] == 1000
    assert device_info["microscope_gain"] == 2
    # Note: Dual Color Microscope and Dual LED power fields are not yet handled by the current extractor implementation

    # Test subject info - only sex should be included (other fields are empty or 0)
    subject_info = extractor._get_subject_info()
    assert isinstance(subject_info, dict)
    assert subject_info["sex"] == "m"
    assert "animal_id" not in subject_info
    assert "species" not in subject_info
    assert "weight" not in subject_info
    assert "date_of_birth" not in subject_info
    assert "description" not in subject_info

    # Test probe info - should include non-empty/non-zero values
    probe_info = extractor._get_probe_info()
    assert isinstance(probe_info, dict)
    assert probe_info["Probe Diameter (mm)"] == 0.66
    assert probe_info["Probe Length (mm)"] == 7.5
    assert probe_info["Probe Pitch"] == 0.5
    assert probe_info["Probe Rotation (degrees)"] == 180
    assert probe_info["Probe Type"] == "ProView DC Integrated Lens"
    assert "Probe Flip" not in probe_info


def test_inscopiximagingextractor_dual_color_movie_with_dropped_frames():
    """
    Test with a dual-color movie file containing dropped frames.

    File: dual_color_movie_with_dropped_frames.isxd
    Metadata:
    - Source: Movie file (Inscopix isxcore v1.5.0)
    - 25 samples (frames) with 4 dropped frames [5, 6, 7, 8]
    - Frame rate: 6.0 Hz (Period = 166.7 ms)
    - Volumetric: True 3 planes at focus depths (300, 500, 700) but multiplane disabled(multiplane enabled=False in config)
    - Frame shape: (288, 480) pixels
    - Data type: uint16
    - Channels: 1 (isx format supports only single channel)
    - Microscope: Inscopix Dual Color, Serial: FA-1234567
    - LED Power: EX LED 1: 0.2 mW/mm², EX LED 2: 0.2 mW/mm²
    - Exposure: 79 ms, Focus: 30, Gain: 2.0
    - Acquisition start time: 2020-09-17 15:47:22.847000
    - Session name: Session 20200917-093000
    - Probe: ProView Integrated Lens (Diameter: 1mm, Length: 4mm, Pitch: 0.5, Rotation: 180°)

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
    2. Retrieves the video data and raw data with correct shape and type.
    3. Tests comprehensive metadata extraction for dual-color microscopy with probe info.
    4. Validates probe information extraction for non-empty/non-zero values.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "dual_color_movie_with_dropped_frames.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    # Basic video properties
    assert extractor.get_num_samples() == 25
    assert extractor.get_image_shape() == (288, 480)
    assert extractor.get_dtype() is dtype("uint16")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 6.000177005221654, decimal=10)
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_samples([0]).shape == (25, 288, 480)
    assert extractor.get_series().shape == (25, 288, 480)

    # Test session start time
    assert extractor._get_session_start_time() == datetime(2020, 9, 17, 15, 47, 22, 847000)

    # Test session info
    session_info = extractor._get_session_info()
    assert session_info["session_name"] == "Session 20200917-093000"
    assert "experimenter_name" not in session_info

    # Test device info - comprehensive for dual-color microscopy
    device_info = extractor._get_device_info()
    assert isinstance(device_info, dict)
    assert device_info["device_name"] == "Dual Color"
    assert device_info["device_serial_number"] == "FA-1234567"
    assert device_info["acquisition_software_version"] == "1.5.0"
    assert device_info["field_of_view_pixels"] == (288, 480)
    assert device_info["exposure_time_ms"] == 79
    assert device_info["microscope_focus"] == 30
    assert device_info["microscope_gain"] == 2
    # Note: Dual Color Microscope and Dual LED power fields are not yet handled by the current extractor implementation

    # Test subject info
    subject_info = extractor._get_subject_info()
    assert isinstance(subject_info, dict)
    assert subject_info["sex"] == "m"
    assert "animal_id" not in subject_info
    assert "species" not in subject_info
    assert "weight" not in subject_info
    assert "date_of_birth" not in subject_info
    assert "description" not in subject_info

    # Test probe info
    probe_info = extractor._get_probe_info()
    assert isinstance(probe_info, dict)
    assert probe_info["Probe Diameter (mm)"] == 1
    assert probe_info["Probe Length (mm)"] == 4
    assert probe_info["Probe Pitch"] == 0.5
    assert probe_info["Probe Rotation (degrees)"] == 180
    assert probe_info["Probe Type"] == "ProView Integrated Lens"
    assert "Probe Flip" not in probe_info
