import numpy as np
from numpy import dtype
from numpy.testing import assert_array_equal
from datetime import datetime, timezone
import pytest

from roiextractors.extractors.femtonicsimagingextractor import FemtonicsImagingExtractor
from .setup_paths import OPHYS_DATA_PATH


def test_femtonicsimagingextractor_p29_mesc():
    """
    Test with Femtonics p29.mesc file.

    File: p29.mesc
    Metadata for MSession 0, MUnit 0 (also includes MUnit 1):
    - Source: Femtonics .mesc file (MESc 3.3)
    - Number of Sampples : 5 (frames)
    - Frame rate: ~30.96 Hz (32.297 ms per frame)
    - Frame shape: (512, 512) pixels
    - Pixel size: 1.782 µm x 1.782 µm
    - Channels: 2 (UG, UR)
    - Experimenter: flaviod
    - Date: 2017-09-29 09:53:00.903594 (30 min later for MUnit 1)
    - Setup: SN20150066006-MBMoser_1

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, channels, etc.).
    2. Retrieves the correct channels and properties.
    3. Tests Femtonics-specific functionality.

    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p29.mesc"

    # Test available channels static method
    available_channels = FemtonicsImagingExtractor.get_available_channels(file_path, session_index=0, munit_index=0)
    assert available_channels == ["UG", "UR"]
    assert len(available_channels) == 2

    # Test that extractor throws error when multiple channels exist and none specified
    with pytest.raises(ValueError, match="Multiple channels found in"):
        FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0)

    # Create extractor with explicit channel selection for the rest of the test
    extractor = FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0, channel_name="UG")

    # Basic properties
    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_num_samples() == 5

    # Test metadata claims
    assert extractor._get_image_shape_metadata() == (512, 512, 5)

    # Test sampling frequency calculation (should still work from metadata)
    expected_freq = 1000.0 / 32.29672617170252
    actual_freq = extractor.get_sampling_frequency()
    assert abs(actual_freq - expected_freq) < 0.01  # ~30.96 Hz

    assert extractor.get_channel_names() == ["UG"]  # Selected channel
    assert extractor.extractor_name == "FemtonicsImaging"

    # Test session info (number of MUnits and session UUID)
    session_info = extractor._get_session_uuid()
    expected_uuid = np.array(
        [102, 213, 51, 146, 143, 154, 66, 41, 182, 97, 30, 169, 181, 145, 82, 30], dtype=session_info.dtype
    )
    assert np.array_equal(session_info, expected_uuid)

    # Test channel selection by name
    extractor_ug = FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0, channel_name="UG")
    extractor_ur = FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0, channel_name="UR")

    assert extractor_ug.get_channel_names() == ["UG"]
    assert extractor_ur.get_channel_names() == ["UR"]

    # Test invalid channel name
    with pytest.raises(ValueError, match="Channel 'InvalidChannel' not found"):
        FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0, channel_name="InvalidChannel")

    # Test pixel size extraction - should be exactly 1.782 µm
    pixel_size = extractor._get_pixel_size_in_micrometers()
    assert isinstance(pixel_size, tuple)
    assert len(pixel_size) == 2
    assert pixel_size[0] == 1.7821140546875  # Exact value from metadata
    assert pixel_size[1] == 1.7821140546875  # Exact value from metadata

    # Test measurement date - using new getter method
    measurement_date = extractor._get_session_start_time()
    assert measurement_date == datetime(2017, 9, 29, 7, 53, 0, 903594, tzinfo=timezone.utc)

    # Test experimenter info - using new getter method
    experimenter_info = extractor._get_experimenter_info()
    assert isinstance(experimenter_info, dict)
    assert experimenter_info["username"] == "flaviod"
    assert experimenter_info["setup_id"] == "SN20150066006-MBMoser_1"
    assert experimenter_info["hostname"] == "KI-FEMTO-0185"

    # Test MESc version info - using new getter method
    version_info = extractor._get_mesc_version_info()
    assert isinstance(version_info, dict)
    assert version_info["version"] == "MESc 3.3"
    assert version_info["revision"] == 4356

    # Test geometric transformations
    geo = extractor._get_geometric_transformations()
    assert np.allclose(geo["translation"], np.array([-456.221198, -456.221198, -11608.54]))
    assert np.allclose(geo["rotation"], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(geo["labeling_origin"], np.array([0.0, 0.0, -11474.34]))

    # Test PMT settings (should match XML)
    pmt_settings = extractor._get_pmt_settings()
    assert isinstance(pmt_settings, dict)
    assert "UG" in pmt_settings
    assert "UR" in pmt_settings
    assert pmt_settings["UG"]["voltage"] == 65.0
    assert pmt_settings["UG"]["warmup_time"] == -0.2
    assert pmt_settings["UR"]["voltage"] == 70.0
    assert pmt_settings["UR"]["warmup_time"] == -0.2

    # Test scan parameters (should match XML)
    scan_params = extractor._get_scan_parameters()
    assert isinstance(scan_params, dict)
    assert scan_params["SizeX"] == 512
    assert scan_params["SizeY"] == 512
    assert scan_params["PixelSizeX"] == 1.7821140546875001
    assert scan_params["PixelSizeY"] == 1.7821140546875001
    assert scan_params["Pixelclock"] == 8.116736e6

    # Test data type
    data_type = extractor.get_dtype()
    assert isinstance(data_type, np.dtype)

    # Test data retrieval with the actual 5 frames
    if extractor.get_num_samples() > 0:
        # Test single frame using get_series
        frame = extractor.get_series(start_sample=0, end_sample=1)
        assert frame.shape == (1,) + extractor.get_image_shape()

        # Test series - all 5 frames
        series = extractor.get_series(start_sample=0, end_sample=5)
        assert series.shape == (5,) + extractor.get_image_shape()

        # Test partial series
        partial_series = extractor.get_series(start_sample=1, end_sample=4)
        assert partial_series.shape == (3,) + extractor.get_image_shape()

        # Test that different channels give same dimensions
        series_ug = extractor_ug.get_series(start_sample=0, end_sample=5)
        series_ur = extractor_ur.get_series(start_sample=0, end_sample=5)
        assert series_ug.shape == series_ur.shape
        assert series_ug.shape == (5,) + extractor.get_image_shape()


def test_femtonicsimagingextractor_p30_mesc():
    """
    Test with Femtonics p30.mesc file.

    File: p30.mesc
    Metadata for MSession 0, MUnit 0 (also includes MUnit 1):
    - Source: Femtonics .mesc file (MESc 3.3)
    - Number of Sampples : 5 (frames)
    - Frame rate: ~30.96 Hz (32.297 ms per frame)
    - Frame shape: (512, 512) pixels
    - Pixel size: 1.782 µm x 1.782 µm
    - Channels: 2 (UG, UR)
    - Experimenter: flaviod
    - Session Start time: 2017-09-30 09:36:12.098727 UTC (30 min  later for MUnit 1)
    - Setup: SN20150066006-MBMoser_1

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, channels, etc.).
    2. Retrieves the correct channels and properties.
    3. Tests Femtonics-specific functionality.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p30.mesc"

    # Test available channels static method
    available_channels = FemtonicsImagingExtractor.get_available_channels(file_path, session_index=0, munit_index=0)
    assert available_channels == ["UG", "UR"]
    assert len(available_channels) == 2

    # Test that extractor throws error when multiple channels exist and none specified
    with pytest.raises(ValueError, match="Multiple channels found in"):
        FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0)

    # Create extractor with explicit channel selection for the rest of the test
    extractor = FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0, channel_name="UG")

    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_num_samples() == 5

    # Test metadata claims
    assert extractor._get_image_shape_metadata() == (512, 512, 5)

    # Test sampling frequency calculation
    expected_freq = 1000.0 / 32.29672617170252
    actual_freq = extractor.get_sampling_frequency()
    assert abs(actual_freq - expected_freq) < 0.01  # ~30.96 Hz

    assert extractor.get_channel_names() == ["UG"]  # Selected channel
    assert extractor.extractor_name == "FemtonicsImaging"

    # Test session info (number of MUnits and session UUID)
    session_info = extractor._get_session_uuid()
    expected_uuid = np.array(
        [7, 28, 27, 145, 166, 138, 70, 179, 135, 2, 182, 25, 177, 189, 180, 155], dtype=session_info.dtype
    )
    assert np.array_equal(session_info, expected_uuid)

    # Test channel selection by name
    extractor_ug = FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0, channel_name="UG")
    extractor_ur = FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0, channel_name="UR")

    assert extractor_ug.get_channel_names() == ["UG"]
    assert extractor_ur.get_channel_names() == ["UR"]

    # Test invalid channel name
    with pytest.raises(ValueError, match="Channel 'InvalidChannel' not found"):
        FemtonicsImagingExtractor(file_path=file_path, session_index=0, munit_index=0, channel_name="InvalidChannel")

    # Test pixel size extraction - should be exactly 1.782 µm
    pixel_size = extractor._get_pixel_size_in_micrometers()
    assert isinstance(pixel_size, tuple)
    assert len(pixel_size) == 2
    assert pixel_size[0] == 1.7821140546875  # Exact value from metadata
    assert pixel_size[1] == 1.7821140546875  # Exact value from metadata

    # Test measurement date - using new getter method
    measurement_date = extractor._get_session_start_time()
    assert measurement_date == datetime(2017, 9, 30, 9, 36, 12, 98727, tzinfo=timezone.utc)

    # Test experimenter info - using new getter method
    experimenter_info = extractor._get_experimenter_info()
    assert isinstance(experimenter_info, dict)
    assert experimenter_info["username"] == "flaviod"
    assert experimenter_info["setup_id"] == "SN20150066006-MBMoser_1"
    assert experimenter_info["hostname"] == "KI-FEMTO-0185"

    # Test MESc version info - using new getter method
    version_info = extractor._get_mesc_version_info()
    assert isinstance(version_info, dict)
    assert version_info["version"] == "MESc 3.3"
    assert version_info["revision"] == 4356

    # Test geometric transformations
    geo = extractor._get_geometric_transformations()
    assert np.allclose(geo["translation"], np.array([-456.221198, -456.221198, -11425.51]))
    assert np.allclose(geo["rotation"], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(geo["labeling_origin"], np.array([0.0, 0.0, -11281.89]))

    # Test PMT settings (should match XML)
    pmt_settings = extractor._get_pmt_settings()
    assert isinstance(pmt_settings, dict)
    assert "UG" in pmt_settings
    assert "UR" in pmt_settings
    assert pmt_settings["UG"]["voltage"] == 65.0
    assert pmt_settings["UG"]["warmup_time"] == -0.2
    assert pmt_settings["UR"]["voltage"] == 70.0
    assert pmt_settings["UR"]["warmup_time"] == -0.2

    # Test scan parameters (should match XML)
    scan_params = extractor._get_scan_parameters()
    assert isinstance(scan_params, dict)
    assert scan_params["SizeX"] == 512
    assert scan_params["SizeY"] == 512
    assert scan_params["PixelSizeX"] == 1.7821140546875001
    assert scan_params["PixelSizeY"] == 1.7821140546875001
    assert scan_params["Pixelclock"] == 8.116736e6

    # Test data type
    data_type = extractor.get_dtype()
    assert isinstance(data_type, np.dtype)

    # Test data retrieval with the actual 5 frames
    if extractor.get_num_samples() > 0:
        # Test single frame using get_series
        frame = extractor.get_series(start_sample=0, end_sample=1)
        assert frame.shape == (1,) + extractor.get_image_shape()

        # Test series - all 5 frames
        series = extractor.get_series(start_sample=0, end_sample=5)
        assert series.shape == (5,) + extractor.get_image_shape()

        # Test partial series
        partial_series = extractor.get_series(start_sample=1, end_sample=4)
        assert partial_series.shape == (3,) + extractor.get_image_shape()

        # Test that different channels give same dimensions
        series_ug = extractor_ug.get_series(start_sample=0, end_sample=5)
        series_ur = extractor_ur.get_series(start_sample=0, end_sample=5)
        assert series_ug.shape == series_ur.shape
        assert series_ug.shape == (5,) + extractor.get_image_shape()
