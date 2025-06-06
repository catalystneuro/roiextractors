import numpy as np
from numpy import dtype
from numpy.testing import assert_array_equal
from datetime import datetime

from roiextractors.extractors.femtonicsimagingextractor import FemtonicsImagingExtractor
from .setup_paths import OPHYS_DATA_PATH


def test_femtonicsimagingextractor_p29_mesc():
    """
    Test with Femtonics p29.mesc file.

    File: p29.mesc
    Metadata (note: metadata vs actual data mismatch):
    - Source: Femtonics .mesc file (MESc 3.3)
    - Metadata claims: 56250 samples (frames)
    - Actual data contains: 5 samples (frames) - this appears to be a preview/subset
    - Frame rate: ~30.96 Hz (32.297 ms per frame)
    - Frame shape: (512, 512) pixels
    - Pixel size: 1.782 µm x 1.782 µm
    - Channels: 2 (UG, UR)
    - Experimenter: flaviod
    - Date: 2017-09-29 09:53:00.903594
    - Setup: SN20150066006-MBMoser_1

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, channels, etc.).
    2. Retrieves the correct channels and properties.
    3. Tests Femtonics-specific functionality like channel selection and metadata extraction.
    4. Handles metadata vs actual data mismatches gracefully.

    Note : the actual data only contains 5 frames, not the 56250 claimed in metadata likely due to being truncated for testing.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Femtonics" / "moser_lab_mec" / "p29.mesc"
    extractor = FemtonicsImagingExtractor(file_path=file_path)

    # Debug: The actual data has only 5 frames, not 56250 as claimed in metadata
    print(f"\n=== IMPORTANT FINDING ===")
    print(f"Metadata claims: 56250 frames")
    print(f"Actual HDF5 data: 5 frames")
    print(f"This is a metadata vs actual data mismatch!")

    # Basic properties - test against ACTUAL data, not metadata claims
    assert extractor.get_image_shape() == (512, 512)
    assert extractor.get_num_samples() == 5  # Test actual data, not metadata

    # Test metadata claims
    assert extractor.get_image_shape_metadata() == (512, 512, 56250)

    # Test sampling frequency calculation (should still work from metadata)
    expected_freq = 1000.0 / 32.29672617170252
    actual_freq = extractor.get_sampling_frequency()
    assert abs(actual_freq - expected_freq) < 0.01  # ~30.96 Hz

    assert extractor.get_channel_names() == ["UG"]  # Default first channel
    assert extractor.get_num_channels() == 1  # Single channel extraction
    assert extractor.extractor_name == "FemtonicsImaging"

    # Test available channels static method
    channels = FemtonicsImagingExtractor.get_available_channels(file_path)
    assert channels == ["UG", "UR"]
    assert len(channels) == 2

    # Test channel selection by name
    extractor_ug = FemtonicsImagingExtractor(file_path=file_path, channel_name="UG")
    extractor_ur = FemtonicsImagingExtractor(file_path=file_path, channel_name="UR")

    assert extractor_ug.get_channel_names() == ["UG"]
    assert extractor_ur.get_channel_names() == ["UR"]

    # Test channel selection by index
    extractor_by_index = FemtonicsImagingExtractor(file_path=file_path, channel_index=1)
    assert extractor_by_index.get_channel_names() == ["UR"]

    # Test pixel size extraction - should be exactly 1.782 µm
    pixel_size = extractor.get_pixel_size()
    assert isinstance(pixel_size, tuple)
    assert len(pixel_size) == 2
    assert pixel_size[0] == 1.7821140546875  # Exact value from metadata
    assert pixel_size[1] == 1.7821140546875  # Exact value from metadata

    # Test measurement date - using new getter method
    measurement_date = extractor.get_measurement_date()
    assert measurement_date == datetime(2017, 9, 29, 9, 53, 0, 903594)

    # Test experimenter info - using new getter method
    experimenter_info = extractor.get_experimenter_info()
    assert isinstance(experimenter_info, dict)
    assert experimenter_info["username"] == "flaviod"
    assert experimenter_info["setup_id"] == "SN20150066006-MBMoser_1"
    assert experimenter_info["hostname"] == "KI-FEMTO-0185"

    # Test MESc version info - using new getter method
    version_info = extractor.get_mesc_version_info()
    assert isinstance(version_info, dict)
    assert version_info["version"] == "MESc 3.3"
    assert version_info["revision"] == 4356

    # Test geometric transformations
    geo = extractor.get_geometric_transformations()
    assert np.allclose(geo["translation"], np.array([-456.221198, -456.221198, -11608.54]))
    assert np.allclose(geo["rotation"], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(geo["labeling_origin"], np.array([0.0, 0.0, -11474.34]))

    # Test PMT settings (should match XML)
    pmt_settings = extractor.get_pmt_settings()
    assert isinstance(pmt_settings, dict)
    assert "UG" in pmt_settings
    assert "UR" in pmt_settings
    assert pmt_settings["UG"]["voltage"] == 65.0
    assert pmt_settings["UG"]["warmup_time"] == -0.2
    assert pmt_settings["UR"]["voltage"] == 70.0
    assert pmt_settings["UR"]["warmup_time"] == -0.2

    # Test scan parameters (should match XML)
    scan_params = extractor.get_scan_parameters()
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
        # Test single frame
        frame = extractor.get_frames(frame_idxs=0)
        assert frame.shape == extractor.get_image_shape()

        # Test series - all 5 frames
        series = extractor.get_series(start_sample=0, end_sample=5)
        assert series.shape == (5,) + extractor.get_image_shape()

        # Test that different channels give same dimensions
        series_ug = extractor_ug.get_series(start_sample=0, end_sample=5)
        series_ur = extractor_ur.get_series(start_sample=0, end_sample=5)
        assert series_ug.shape == series_ur.shape
        assert series_ug.shape == (5,) + extractor.get_image_shape()
