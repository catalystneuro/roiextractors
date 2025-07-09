from datetime import datetime, timezone

import numpy as np
import pytest

from roiextractors.extractors.femtonicsimagingextractor import FemtonicsImagingExtractor

from .setup_paths import OPHYS_DATA_PATH


def test_femtonicsimagingextractor_p29_mesc():
    """
    Test with Femtonics p29.mesc file.

    File: p29.mesc
    Metadata for MSession_0, MUnit_0 (also includes MUnit_1):
    - Source: Femtonics .mesc file (MESc 3.3)
    - Number of Sampples : 5 (frames)
    - Frame rate: ~30.96 Hz (32.297 ms per frame)
    - Frame shape: (512, 512) pixels
    - Pixel size: 1.782 µm x 1.782 µm
    - Channels: 2 (UG, UR)
    - Experimenter: flaviod
    - Date: 2017-09-29 09:53:00.903594 (30 min later for MUnit_1)
    - Setup: SN20150066006-MBMoser_1

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, channels, etc.).
    2. Retrieves the correct channels and properties.
    3. Tests Femtonics-specific functionality.

    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p29.mesc"

    # Test available sessions static method
    available_sessions = FemtonicsImagingExtractor.get_available_sessions(file_path)
    assert available_sessions == ["MSession_0"]

    # Test available munits static method
    available_munits = FemtonicsImagingExtractor.get_available_munits(file_path, session_name="MSession_0")
    assert available_munits == ["MUnit_0", "MUnit_1"]

    # Test available channels static method
    available_channels = FemtonicsImagingExtractor.get_available_channels(
        file_path, session_name="MSession_0", munit_name="MUnit_0"
    )
    assert available_channels == ["UG", "UR"]

    # Extractor with explicit session, munit and channel selection for the rest of the test
    extractor = FemtonicsImagingExtractor(
        file_path=file_path, session_name="MSession_0", munit_name="MUnit_0", channel_name="UG"
    )

    # Basic properties
    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_num_samples() == 5

    # Test metadata claims
    assert extractor._get_image_shape_metadata() == (512, 512, 5)

    # Test sampling frequency calculation
    expected_freq = 1000.0 / 32.29672617170252
    actual_freq = extractor.get_sampling_frequency()
    assert abs(actual_freq - expected_freq) < 0.01

    assert extractor.get_channel_names() == ["UG"]
    assert extractor.extractor_name == "FemtonicsImaging"

    # Test session info
    session_info = extractor._get_session_uuid()
    expected_uuid = "66d53392-8f9a-4229-b661-1ea9b591521e"
    assert session_info == expected_uuid

    # Test channel selection by name
    extractor_ug = FemtonicsImagingExtractor(
        file_path=file_path, session_name="MSession_0", munit_name="MUnit_0", channel_name="UG"
    )
    extractor_ur = FemtonicsImagingExtractor(
        file_path=file_path, session_name="MSession_0", munit_name="MUnit_0", channel_name="UR"
    )

    assert extractor_ug.get_channel_names() == ["UG"]
    assert extractor_ur.get_channel_names() == ["UR"]

    # Test pixel size extraction - should be exactly 1.782 µm
    pixel_size = extractor._get_pixels_sizes_and_units()
    assert isinstance(pixel_size, dict)
    assert pixel_size["x_size"] == 1.7821140546875
    assert pixel_size["y_size"] == 1.7821140546875
    assert pixel_size["x_units"] == "µm"
    assert pixel_size["y_units"] == "µm"

    # Test measurement date
    measurement_date = extractor._get_session_start_time()
    assert measurement_date == datetime(2017, 9, 29, 7, 53, 0, 903594, tzinfo=timezone.utc)

    # Test experimenter info
    experimenter_info = extractor._get_experimenter_info()
    assert isinstance(experimenter_info, dict)
    assert experimenter_info["username"] == "flaviod"
    assert experimenter_info["setup_id"] == "SN20150066006-MBMoser_1"
    assert experimenter_info["hostname"] == "KI-FEMTO-0185"

    # Test MESc version info
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

    # Test get_metadata
    metadata = extractor._get_metadata()
    assert isinstance(metadata, dict)

    # Check that metadata now includes session and munit names instead of indices
    assert metadata["session_name"] == "MSession_0"
    assert metadata["munit_name"] == "MUnit_0"
    assert metadata["selected_channel"] == "UG"
    assert metadata["available_channels"] == ["UG", "UR"]

    expected_keys = [
        "session_name",
        "munit_name",
        "selected_channel",
        "available_channels",
        "available_sessions",
        "available_munits",
        "session_uuid",
        "pixel_size_micrometers",
        "image_shape_metadata",
        "session_start_time",
        "experimenter_info",
        "geometric_transformations",
        "mesc_version_info",
        "pmt_settings",
        "scan_parameters",
        "frame_duration_ms",
        "sampling_frequency_hz",
    ]
    for key in expected_keys:
        assert key in metadata, f"Key '{key}' missing from metadata"

    assert metadata["session_uuid"] == session_info
    assert metadata["pixel_size_micrometers"] == pixel_size
    assert metadata["image_shape_metadata"] == (512, 512, 5)
    assert metadata["session_start_time"] == measurement_date
    assert metadata["experimenter_info"] == experimenter_info
    assert metadata["mesc_version_info"] == version_info
    assert metadata["pmt_settings"] == pmt_settings
    assert metadata["scan_parameters"] == scan_params
    assert metadata["frame_duration_ms"] == 32.29672617170252
    assert abs(metadata["sampling_frequency_hz"] - expected_freq) < 0.01

    # Test data type
    data_type = extractor.get_dtype()
    assert isinstance(data_type, np.dtype)

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

    # Test invalid channel name
    with pytest.raises(ValueError, match="Channel 'InvalidChannel' not found"):
        FemtonicsImagingExtractor(
            file_path=file_path, session_name="MSession_0", munit_name="MUnit_0", channel_name="InvalidChannel"
        )

    # Test invalid session name
    with pytest.raises(ValueError, match="Session 'InvalidSession' not found"):
        FemtonicsImagingExtractor(file_path=file_path, session_name="InvalidSession")

    # Test invalid munit name
    with pytest.raises(ValueError, match="MUnit 'InvalidMUnit' not found"):
        FemtonicsImagingExtractor(file_path=file_path, session_name="MSession_0", munit_name="InvalidMUnit")

    # Test that extractor throws error when multiple channels exist and none specified
    with pytest.raises(ValueError, match="Multiple channels found in"):
        FemtonicsImagingExtractor(file_path=file_path, session_name="MSession_0", munit_name="MUnit_0")

    # Test that extractor throws error when multiple munits exist and none specified
    with pytest.raises(ValueError, match="Multiple MUnits found in"):
        FemtonicsImagingExtractor(file_path=file_path, session_name="MSession_0")


def test_femtonicsimagingextractor_p30_mesc():
    """
    Test with Femtonics p30.mesc file.

    File: p30.mesc
    Metadata for MSession_0, MUnit_0 (also includes MUnit_1):
    - Source: Femtonics .mesc file (MESc 3.3)
    - Number of Sampples : 5 (frames)
    - Frame rate: ~30.96 Hz (32.297 ms per frame)
    - Frame shape: (512, 512) pixels
    - Pixel size: 1.782 µm x 1.782 µm
    - Channels: 2 (UG, UR)
    - Experimenter: flaviod
    - Session Start time: 2017-09-30 09:36:12.098727 UTC (30 min  later for MUnit_1)
    - Setup: SN20150066006-MBMoser_1

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, channels, etc.).
    2. Retrieves the correct channels and properties.
    3. Tests Femtonics-specific functionality.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p30.mesc"

    # Test available sessions static method
    available_sessions = FemtonicsImagingExtractor.get_available_sessions(file_path)
    assert available_sessions == ["MSession_0"]

    # Test available munits static method
    available_munits = FemtonicsImagingExtractor.get_available_munits(file_path, session_name="MSession_0")
    assert available_munits == ["MUnit_0", "MUnit_1"]

    # Test available channels static method
    available_channels = FemtonicsImagingExtractor.get_available_channels(
        file_path, session_name="MSession_0", munit_name="MUnit_0"
    )
    assert available_channels == ["UG", "UR"]
    assert len(available_channels) == 2

    # Extractor with explicit session, munit, and channel selection for the rest of the test
    extractor = FemtonicsImagingExtractor(
        file_path=file_path, session_name="MSession_0", munit_name="MUnit_0", channel_name="UG"
    )

    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_num_samples() == 5

    # Test metadata claims
    assert extractor._get_image_shape_metadata() == (512, 512, 5)

    # Test sampling frequency calculation
    expected_freq = 1000.0 / 32.29672617170252
    actual_freq = extractor.get_sampling_frequency()
    assert abs(actual_freq - expected_freq) < 0.01  # ~30.96 Hz

    assert extractor.get_channel_names() == ["UG"]
    assert extractor.extractor_name == "FemtonicsImaging"

    # Test session info
    session_info = extractor._get_session_uuid()
    expected_uuid = "071c1b91-a68a-46b3-8702-b619b1bdb49b"
    assert session_info == expected_uuid

    # Test channel selection by name
    extractor_ug = FemtonicsImagingExtractor(
        file_path=file_path, session_name="MSession_0", munit_name="MUnit_0", channel_name="UG"
    )
    extractor_ur = FemtonicsImagingExtractor(
        file_path=file_path, session_name="MSession_0", munit_name="MUnit_0", channel_name="UR"
    )

    assert extractor_ug.get_channel_names() == ["UG"]
    assert extractor_ur.get_channel_names() == ["UR"]

    # Test pixel size extraction - should be exactly 1.782 µm
    pixel_size = extractor._get_pixels_sizes_and_units()
    assert isinstance(pixel_size, dict)
    assert pixel_size["x_size"] == 1.7821140546875
    assert pixel_size["y_size"] == 1.7821140546875
    assert pixel_size["x_units"] == "µm"
    assert pixel_size["y_units"] == "µm"

    # Test measurement date
    measurement_date = extractor._get_session_start_time()
    assert measurement_date == datetime(2017, 9, 30, 9, 36, 12, 98727, tzinfo=timezone.utc)

    # Test experimenter info
    experimenter_info = extractor._get_experimenter_info()
    assert isinstance(experimenter_info, dict)
    assert experimenter_info["username"] == "flaviod"
    assert experimenter_info["setup_id"] == "SN20150066006-MBMoser_1"
    assert experimenter_info["hostname"] == "KI-FEMTO-0185"

    # Test MESc version info
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

    # Test get_metadata
    metadata = extractor._get_metadata()
    assert isinstance(metadata, dict)

    # Check that metadata now includes session and munit names instead of indices
    assert metadata["session_name"] == "MSession_0"
    assert metadata["munit_name"] == "MUnit_0"
    assert metadata["selected_channel"] == "UG"
    assert metadata["available_channels"] == ["UG", "UR"]

    expected_keys = [
        "session_name",
        "munit_name",
        "selected_channel",
        "available_channels",
        "available_sessions",
        "available_munits",
        "session_uuid",
        "pixel_size_micrometers",
        "image_shape_metadata",
        "session_start_time",
        "experimenter_info",
        "geometric_transformations",
        "mesc_version_info",
        "pmt_settings",
        "scan_parameters",
        "frame_duration_ms",
        "sampling_frequency_hz",
    ]
    for key in expected_keys:
        assert key in metadata, f"Key '{key}' missing from metadata"

    assert metadata["session_uuid"] == session_info
    assert metadata["pixel_size_micrometers"] == pixel_size
    assert metadata["image_shape_metadata"] == (512, 512, 5)
    assert metadata["session_start_time"] == measurement_date
    assert metadata["experimenter_info"] == experimenter_info
    assert metadata["mesc_version_info"] == version_info
    assert metadata["pmt_settings"] == pmt_settings
    assert metadata["scan_parameters"] == scan_params
    assert metadata["frame_duration_ms"] == 32.29672617170252
    assert abs(metadata["sampling_frequency_hz"] - expected_freq) < 0.01

    # Test data type
    data_type = extractor.get_dtype()
    assert isinstance(data_type, np.dtype)

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

    # Test MUnit_1 to ensure both units work
    extractor_munit1 = FemtonicsImagingExtractor(
        file_path=file_path, session_name="MSession_0", munit_name="MUnit_1", channel_name="UG"
    )
    assert extractor_munit1.get_image_shape() == (512, 512)
    assert extractor_munit1.get_num_samples() == 5

    # Test invalid channel name
    with pytest.raises(ValueError, match="Channel 'InvalidChannel' not found"):
        FemtonicsImagingExtractor(
            file_path=file_path, session_name="MSession_0", munit_name="MUnit_0", channel_name="InvalidChannel"
        )

    # Test invalid session name
    with pytest.raises(ValueError, match="Session 'InvalidSession' not found"):
        FemtonicsImagingExtractor(file_path=file_path, session_name="InvalidSession")

    # Test invalid munit name
    with pytest.raises(ValueError, match="MUnit 'InvalidMUnit' not found"):
        FemtonicsImagingExtractor(file_path=file_path, session_name="MSession_0", munit_name="InvalidMUnit")

    # Test that extractor throws error when multiple channels exist and none specified
    with pytest.raises(ValueError, match="Multiple channels found in"):
        FemtonicsImagingExtractor(file_path=file_path, session_name="MSession_0", munit_name="MUnit_0")

    # Test that extractor throws error when multiple munits exist and none specified
    with pytest.raises(ValueError, match="Multiple MUnits found in"):
        FemtonicsImagingExtractor(file_path=file_path, session_name="MSession_0")


def test_femtonicsimagingextractor_single_channel():
    """
    Test FemtonicsImagingExtractor with single channel .mesc file.

    This test uses a single channel file and verifies automatic channel and session selection
    and correct metadata extraction.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "single_channel.mesc"

    # Test that with single channel, no channel_name specification is needed
    extractor = FemtonicsImagingExtractor(file_path=file_path, munit_name="MUnit_60")

    # Test basic properties
    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_channel_names() == ["UG"]  # Single channel should be auto-selected
    assert extractor.extractor_name == "FemtonicsImaging"

    # Test session UUID
    session_uuid = extractor._get_session_uuid()
    assert session_uuid == "eab55dc7-173e-4fcb-8746-65274f1e5f96"

    # Test session start time
    session_start_time = extractor._get_session_start_time()
    assert session_start_time == datetime(2014, 3, 3, 15, 21, 57, 18837, tzinfo=timezone.utc)

    # Test experimenter info
    experimenter_info = extractor._get_experimenter_info()
    assert experimenter_info["username"] == "measurement"

    # Test pixel size extraction
    pixel_size = extractor._get_pixels_sizes_and_units()
    assert isinstance(pixel_size, dict)
    assert pixel_size["x_size"] == pytest.approx(0.8757686997991967, rel=1e-6)
    assert pixel_size["y_size"] == pytest.approx(0.8757686997991966, rel=1e-6)
    assert pixel_size["x_units"] == "µm"
    assert pixel_size["y_units"] == "µm"

    # Test sampling frequency - should be approximately 31.2 Hz
    sampling_freq = extractor.get_sampling_frequency()
    assert sampling_freq == pytest.approx(31.2, rel=1e-2)

    # Test MESc version info
    version_info = extractor._get_mesc_version_info()
    assert version_info["version"] == "MESc 1.0"
    assert version_info["revision"] == 1839

    # Test geometric transformations
    geo = extractor._get_geometric_transformations()
    expected_translation = np.array([-224.19678715, -224.19678715, 0.0])
    expected_rotation = np.array([0.0, 0.0, 0.0, 1.0])
    expected_labeling_origin = np.array([0.0, 0.0, -6724.23])

    assert np.allclose(geo["translation"], expected_translation)
    assert np.allclose(geo["rotation"], expected_rotation)
    assert np.allclose(geo["labeling_origin"], expected_labeling_origin)

    # Test metadata contains all expected information
    metadata = extractor._get_metadata()
    assert metadata["session_name"] == "MSession_0"
    assert metadata["munit_name"] == "MUnit_60"
    assert metadata["selected_channel"] == "UG"
    assert metadata["session_uuid"] == session_uuid
    assert metadata["session_start_time"] == session_start_time
    assert metadata["experimenter_info"] == experimenter_info
    assert metadata["sampling_frequency_hz"] == pytest.approx(31.2, rel=1e-2)

    # Test image shape from metadata
    image_shape_metadata = extractor._get_image_shape_metadata()
    assert image_shape_metadata == (512, 512, extractor.get_num_samples())

    # Test data access
    first_frame = extractor.get_series(start_sample=0, end_sample=1)
    assert first_frame.shape == (1, 512, 512)

    # Test that we can get all frames
    all_frames = extractor.get_series()
    assert all_frames.shape[1:] == (512, 512)
    assert all_frames.shape[0] == extractor.get_num_samples()


def test_femtonicsimagingextractor_single_munit():
    """
    Test FemtonicsImagingExtractor with single MUnit .mesc file.

    This test uses a file with a single MUnit and verifies automatic MUnit selection
    and correct metadata extraction.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "single_m_unit_index.mesc"

    available_sessions = FemtonicsImagingExtractor.get_available_sessions(file_path)
    available_munits = FemtonicsImagingExtractor.get_available_munits(file_path, session_name=available_sessions[0])
    assert available_sessions == ["MSession_0"]
    assert available_munits == ["MUnit_60"]

    extractor = FemtonicsImagingExtractor(file_path=file_path, channel_name="UG")

    # Test basic properties
    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_channel_names() == ["UG"]
    assert extractor.extractor_name == "FemtonicsImaging"

    # Test session UUID
    session_uuid = extractor._get_session_uuid()
    assert session_uuid == "eab55dc7-173e-4fcb-8746-65274f1e5f96"

    # Test session start time
    session_start_time = extractor._get_session_start_time()
    assert session_start_time == datetime(2014, 3, 3, 15, 21, 57, 18837, tzinfo=timezone.utc)

    # Test experimenter info
    experimenter_info = extractor._get_experimenter_info()
    assert experimenter_info["username"] == "measurement"

    # Test pixel size extraction - Grid spacing from Pixel Size X: 0.876 μm, Pixel Size Y: 0.876 μm
    pixel_size = extractor._get_pixels_sizes_and_units()
    assert isinstance(pixel_size, dict)
    assert pixel_size["x_size"] == pytest.approx(0.8757686997991967, rel=1e-6)
    assert pixel_size["y_size"] == pytest.approx(0.8757686997991966, rel=1e-6)
    assert pixel_size["x_units"] == "µm"
    assert pixel_size["y_units"] == "µm"

    # Test sampling frequency - should be approximately 31.2 Hz
    sampling_freq = extractor.get_sampling_frequency()
    assert sampling_freq == pytest.approx(31.2, rel=1e-2)

    # Test MESc version info
    version_info = extractor._get_mesc_version_info()
    assert version_info["version"] == "MESc 1.0"
    assert version_info["revision"] == 1839

    # Test geometric transformations
    geo = extractor._get_geometric_transformations()
    expected_translation = np.array([-224.19678715, -224.19678715, 0.0])
    expected_rotation = np.array([0.0, 0.0, 0.0, 1.0])
    expected_labeling_origin = np.array([0.0, 0.0, -6724.23])

    assert np.allclose(geo["translation"], expected_translation)
    assert np.allclose(geo["rotation"], expected_rotation)
    assert np.allclose(geo["labeling_origin"], expected_labeling_origin)

    # Test metadata contains all expected information
    metadata = extractor._get_metadata()
    assert metadata["session_name"] == "MSession_0"
    assert metadata["selected_channel"] == "UG"
    assert metadata["session_uuid"] == session_uuid
    assert metadata["session_start_time"] == session_start_time
    assert metadata["experimenter_info"] == experimenter_info
    assert metadata["sampling_frequency_hz"] == pytest.approx(31.2, rel=1e-2)

    # Test image shape from metadata - Image dimensions: X Dimension: 512 pixels, Y Dimension: 512 pixels
    image_shape_metadata = extractor._get_image_shape_metadata()
    assert image_shape_metadata == (512, 512, extractor.get_num_samples())

    # Test data access
    first_frame = extractor.get_series(start_sample=0, end_sample=1)
    assert first_frame.shape == (1, 512, 512)

    # Test that we can get all frames
    all_frames = extractor.get_series()
    assert all_frames.shape[1:] == (512, 512)
    assert all_frames.shape[0] == extractor.get_num_samples()
