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
    - Metadata claims: 56250 samples (frames)
    - Actual data contains: 5 samples (frames) - this appears to be a preview/subset
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
    3. Tests Femtonics-specific functionality like channel selection and metadata extraction.
    4. Handles metadata vs actual data mismatches.

    Note : the actual data only contains 5 frames, not the 56250 claimed in metadata likely due to being truncated for testing.
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

    # Debug: The actual data has only 5 frames, not 56250 as claimed in metadata
    print(f"\n=== IMPORTANT FINDING ===")
    print(f"Metadata claims: 56250 frames")
    print(f"Actual HDF5 data: 5 frames")
    print(f"This is a metadata vs actual data mismatch!")

    # Basic properties - test against ACTUAL data, not metadata claims
    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_num_samples() == 5  # Test actual data, not metadata

    # Test metadata claims
    assert extractor._get_image_shape_metadata() == (512, 512, 56250)

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
    - Metadata claims: 56250 samples (frames)
    - Actual data contains: 5 samples (frames) - this appears to be a preview/subset
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
    3. Tests Femtonics-specific functionality like channel selection and metadata extraction.
    4. Handles metadata vs actual data mismatches
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

    # Basic properties - test against ACTUAL data, not metadata claims
    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_num_samples() == 5  # Test actual data, not metadata

    # Test metadata claims
    assert extractor._get_image_shape_metadata() == (512, 512, 56250)

    # Test sampling frequency calculation (should still work from metadata)
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


def test_femtonicsimagingextractor_invalid_files():
    """
    Test handling of invalid files and parameters.
    """

    # Test with valid file but invalid parameters
    valid_file = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p29.mesc"

    # Test invalid session index
    with pytest.raises(ValueError, match="Session index 99 not found"):
        FemtonicsImagingExtractor(valid_file, session_index=99, munit_index=0, channel_name="UG")

    # Test invalid unit index
    with pytest.raises(ValueError, match="Unit index 99 not found"):
        FemtonicsImagingExtractor(valid_file, session_index=0, munit_index=99, channel_name="UG")


def test_femtonicsimagingextractor_static_methods_edge_cases():
    """
    Test static methods with edge cases and error conditions.
    """
    valid_file = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p29.mesc"

    # Test get_available_channels with invalid parameters
    with pytest.raises(ValueError):
        FemtonicsImagingExtractor.get_available_channels(valid_file, session_index=99, munit_index=0)

    with pytest.raises(ValueError):
        FemtonicsImagingExtractor.get_available_channels(valid_file, session_index=0, munit_index=99)

    # Test with non-existent file (should raise FileNotFoundError or IOError)
    with pytest.raises((FileNotFoundError, IOError, OSError)):
        FemtonicsImagingExtractor.get_available_channels("/nonexistent.mesc")

    # Test get_available_sessions
    sessions = FemtonicsImagingExtractor.get_available_sessions(valid_file)
    assert isinstance(sessions, list)
    assert len(sessions) > 0  # p29.mesc should have at least one session
    assert all(session.startswith("MSession_") for session in sessions)

    # Test get_available_units
    units = FemtonicsImagingExtractor.get_available_units(valid_file, session_index=0)
    assert isinstance(units, list)
    assert len(units) > 0  # p29.mesc should have at least one unit
    assert all(unit.startswith("MUnit_") for unit in units)

    # Test get_available_units with invalid session
    empty_units = FemtonicsImagingExtractor.get_available_units(valid_file, session_index=99)
    assert isinstance(empty_units, list)
    assert len(empty_units) == 0


def test_femtonicsimagingextractor_data_consistency():
    """
    Test data consistency across different channels and extraction methods.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p29.mesc"

    extractor_ug = FemtonicsImagingExtractor(file_path, session_index=0, munit_index=0, channel_name="UG")
    extractor_ur = FemtonicsImagingExtractor(file_path, session_index=0, munit_index=0, channel_name="UR")

    # Verify metadata consistency across channels
    assert extractor_ug.get_sampling_frequency() == extractor_ur.get_sampling_frequency()
    assert extractor_ug.get_image_shape() == extractor_ur.get_image_shape()
    assert extractor_ug.get_num_samples() == extractor_ur.get_num_samples()
    assert extractor_ug.get_dtype() == extractor_ur.get_dtype()

    # Test that get_series and get_samples return consistent data
    if extractor_ug.get_num_samples() > 0:
        # Get same frame using different methods
        frame_via_series = extractor_ug.get_series(start_sample=0, end_sample=1)
        frame_via_samples = extractor_ug.get_samples([0])

        assert frame_via_series.shape == frame_via_samples.shape
        assert np.array_equal(frame_via_series, frame_via_samples)

        # Test multiple frames consistency
        if extractor_ug.get_num_samples() >= 3:
            frames_series = extractor_ug.get_series(start_sample=0, end_sample=3)
            frames_samples = extractor_ug.get_samples([0, 1, 2])

            assert frames_series.shape == frames_samples.shape
            assert np.array_equal(frames_series, frames_samples)


def test_femtonicsimagingextractor_multi_unit_p29():
    """
    Test multi-unit functionality using p29.mesc which has multiple units.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p29.mesc"

    # Test that both units can be accessed
    available_units = FemtonicsImagingExtractor.get_available_units(file_path, session_index=0)
    assert len(available_units) >= 1  # Should have at least MUnit_0

    # Test MUnit 0
    extractor_unit0 = FemtonicsImagingExtractor(file_path, session_index=0, munit_index=0, channel_name="UG")
    assert extractor_unit0.get_num_samples() == 5
    assert extractor_unit0.get_image_shape() == (512, 512)

    # Test that channels are consistent across units in the same session
    channels_unit0 = FemtonicsImagingExtractor.get_available_channels(file_path, session_index=0, munit_index=0)

    # If there are multiple units, test the second one
    if len(available_units) > 1:
        channels_unit1 = FemtonicsImagingExtractor.get_available_channels(file_path, session_index=0, munit_index=1)
        assert channels_unit0 == channels_unit1  # Should have same channels

        extractor_unit1 = FemtonicsImagingExtractor(file_path, session_index=0, munit_index=1, channel_name="UG")

        # Both units should have similar properties (though data may differ)
        assert extractor_unit0.get_image_shape() == extractor_unit1.get_image_shape()


def test_femtonicsimagingextractor_multi_unit_p30():
    """
    Test multi-unit functionality using p30.mesc which has multiple units.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p30.mesc"

    # Test that both units can be accessed
    available_units = FemtonicsImagingExtractor.get_available_units(file_path, session_index=0)
    assert len(available_units) >= 1  # Should have at least MUnit_0

    # Test MUnit 0
    extractor_unit0 = FemtonicsImagingExtractor(file_path, session_index=0, munit_index=0, channel_name="UG")
    assert extractor_unit0.get_num_samples() == 5
    assert extractor_unit0.get_image_shape() == (512, 512)

    # Test metadata differences between units (timestamps should differ)
    unit0_time = extractor_unit0._get_session_start_time()

    if len(available_units) > 1:
        extractor_unit1 = FemtonicsImagingExtractor(file_path, session_index=0, munit_index=1, channel_name="UG")
        unit1_time = extractor_unit1._get_session_start_time()

        # Times should be different (30 minutes apart according to docstring)
        if unit0_time and unit1_time:
            time_diff = abs((unit1_time - unit0_time).total_seconds())
            assert time_diff > 0  # Should be different timestamps
