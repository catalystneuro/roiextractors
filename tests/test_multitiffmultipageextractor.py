"""Tests for the MultiTIFFMultiPageExtractor organized by test cases."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from roiextractors.extraction_tools import get_package
from roiextractors.extractors.tiffimagingextractors import (
    MultiTIFFMultiPageExtractor,
    ScanImageImagingExtractor,
)
from tests.setup_paths import OPHYS_DATA_PATH


@pytest.fixture(scope="session")
def test_data_array():
    """Create the reference time series that all tests use.

    The tests for this extractor uses a fixture-based approach where all tests share the same underlying
    TIFF data but interpret it differently based on extractor configuration.

    The time series consists of 12 frames where frame N contains all N's (frame 0 = all 0s, frame 1 = all 1s, etc.).
    This deterministic pattern makes it easy to verify that extractors are reading the correct frames in case
    of debugging or unexpected behavior.

    Index Mappings:
    Each test declares which frames from the shared data should correspond to the extractor's
    output. This makes the expected behavior explicit and helps catch regressions when the
    extractor's interpretation logic changes.

    For example, a single-channel test expects frames [0,1,2,...] while a multi-channel
    test with the same data might expect channel 0 to get frames [0,2,4,...] due to interleaving.

    The 12-frame total ensures all test configurations get complete cycles without warnings.
    """
    # Generate 12 frames total with sequential values
    # Frame 0 = all 0s, Frame 1 = all 1s, etc.
    num_frames = 12
    height, width = 3, 4

    frames = []
    for frame_index in range(num_frames):
        frame = np.full((height, width), frame_index, dtype=np.uint16)
        frames.append(frame)

    return np.stack(frames)


@pytest.fixture(scope="session")
def tiff_file_paths(tmp_path_factory, test_data_array):
    """Write the reference time series to actual TIFF files for extractor testing.

    Takes the 12 frames from test_data_array and writes them to 2 TIFF files
    (6 frames each) to simulate multi-file datasets. Returns the file paths
    for use by MultiTIFFMultiPageExtractor.

    This keeps all tests using the same underlying data while testing different
    interpretations based on extractor configuration.
    """
    tmp_path = tmp_path_factory.mktemp("shared_tiff_data")
    import tifffile

    # Split the 12 frames into 2 files with 6 pages each
    num_files = 2
    pages_per_file = 6

    for file_index in range(num_files):
        file_path = tmp_path / f"test_file_{file_index}.tif"

        # Get the frames for this file
        start_frame = file_index * pages_per_file
        end_frame = start_frame + pages_per_file
        file_frames = test_data_array[start_frame:end_frame]

        with tifffile.TiffWriter(file_path) as writer:
            for frame in file_frames:
                writer.write(frame)

    return sorted(list(tmp_path.glob("*.tif")))


class TestSingleChannelPlanar:
    """Test cases for single channel, planar (num_planes=1) data."""

    # Test data configuration - matches shared fixture (2 files × 6 pages = 12 pages total)
    num_files = 2
    num_channels = 1
    num_planes = 1
    height = 3
    width = 4
    expected_samples = 12  # All 12 pages become samples for single channel planar
    expected_shape = (height, width)
    sampling_frequency = 30.0

    def test_initialization(self, tiff_file_paths):
        """Test initialization of the extractor with single-channel planar data."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape
        assert extractor.get_sample_shape() == self.expected_shape
        assert extractor.get_sampling_frequency() == self.sampling_frequency
        assert extractor.get_num_planes() == self.num_planes
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric == False
        assert extractor.get_channel_names() == [str(i) for i in range(self.num_channels)]
        assert extractor.get_native_timestamps() is None

    def test_get_series(self, tiff_file_paths, test_data_array):
        """Test retrieving series from the extractor with planar data."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        # For single channel planar with CZT order:
        # Each page directly becomes a sample (1 channel × 1 plane = 1 page per sample)
        # Sample to test_data_array mapping: samples 0-11 map directly to test_data_array indices 0-11
        sample_to_test_data_indices = list(range(12))
        expected_time_series = test_data_array[sample_to_test_data_indices]

        # Get actual time series
        actual_time_series = extractor.get_series()

        # Assert equality using numpy testing
        np.testing.assert_array_equal(actual_time_series, expected_time_series)


class TestSingleChannelVolumetric:
    """Test cases for single channel, volumetric (num_planes>1) data."""

    # Test data configuration - using shared 12-page data
    # With 1 channel and 3 planes: 12 pages ÷ 3 planes = 4 samples (complete cycles)
    num_files = 2
    num_channels = 1
    num_planes = 3
    height = 3
    width = 4
    expected_samples = 4  # 12 pages ÷ (1 channel × 3 planes) = 4 complete samples
    expected_shape = (height, width)
    expected_volume_shape = (height, width, num_planes)
    sampling_frequency = 30.0

    def test_initialization(self, tiff_file_paths):
        """Test initialization of the extractor with single-channel volumetric data."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape
        assert extractor.get_sample_shape() == self.expected_volume_shape
        assert extractor.get_volume_shape() == self.expected_volume_shape
        assert extractor.get_sampling_frequency() == self.sampling_frequency
        assert extractor.get_num_planes() == self.num_planes
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric == True
        assert extractor.get_channel_names() == [str(i) for i in range(self.num_channels)]
        assert extractor.get_native_timestamps() is None

    def test_get_series(self, tiff_file_paths, test_data_array):
        """Test retrieving series from the extractor with volumetric data."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        # Get actual time series
        actual_time_series = extractor.get_series()

        # Single channel volumetric sample to test_data_array mapping for CZT order (1 channel, 3 planes)
        sample_to_test_data_mapping = {
            0: [0, 1, 2],  # Sample 0: Z0, Z1, Z2
            1: [3, 4, 5],  # Sample 1: Z0, Z1, Z2
            2: [6, 7, 8],  # Sample 2: Z0, Z1, Z2
            3: [9, 10, 11],  # Sample 3: Z0, Z1, Z2
        }

        # Compare each sample directly to the corresponding test_data_array values using explicit stacking
        for sample_index, test_data_indices in sample_to_test_data_mapping.items():
            # Stack the individual frames into a volume (depth as last dimension)
            frame_stack = []
            for frame_index in test_data_indices:
                frame_stack.append(test_data_array[frame_index])
            expected_volume = np.stack(frame_stack, axis=-1)
            np.testing.assert_array_equal(actual_time_series[sample_index], expected_volume)

    def test_from_folder(self, tiff_file_paths):
        """Test creating an extractor from a folder path."""
        folder_path = tiff_file_paths[0].parent  # Get the folder containing the TIFF files
        extractor = MultiTIFFMultiPageExtractor.from_folder(
            folder_path=folder_path,
            file_pattern="*.tif",
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape


class TestMultiChannelPlanar:
    """Test cases for multi-channel, planar data."""

    # Test data configuration - using shared 12-page data
    # With 2 channels and 1 plane: 12 pages ÷ (2 channels × 1 plane) = 6 samples
    num_files = 2
    num_channels = 2
    num_planes = 1
    height = 3
    width = 4
    expected_samples = 6  # 12 pages ÷ (2 channels × 1 plane) = 6 complete samples
    expected_shape = (height, width)
    sampling_frequency = 30.0

    def test_initialization(self, tiff_file_paths):
        """Test initialization of the extractor with multi-channel planar data."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_name="1",
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape
        assert extractor.get_sample_shape() == self.expected_shape
        assert extractor.get_sampling_frequency() == self.sampling_frequency
        assert extractor.get_num_planes() == self.num_planes
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric == False
        assert extractor.get_channel_names() == [str(i) for i in range(self.num_channels)]
        assert extractor.get_native_timestamps() is None

    def test_get_series_channel_0(self, tiff_file_paths, test_data_array):
        """Test get_series for channel 0 with TCZ dimension order."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_name="0",
            num_planes=self.num_planes,
        )

        actual_data = extractor.get_series()

        # Channel 0 sample to test_data_array mapping for TCZ order (2 channels, 1 plane):
        # Sample 0: [0], Sample 1: [1], ..., Sample 5: [5]
        sample_to_test_data_indices = [0, 1, 2, 3, 4, 5]
        expected_data = test_data_array[sample_to_test_data_indices]

        np.testing.assert_array_equal(actual_data, expected_data)

    def test_get_series_channel_1(self, tiff_file_paths, test_data_array):
        """Test get_series for channel 1 with TCZ dimension order."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_name="1",
            num_planes=self.num_planes,
        )

        actual_data = extractor.get_series()

        # Channel 1 sample to test_data_array mapping for TCZ order (2 channels, 1 plane):
        # Sample 0: [6], Sample 1: [7], ..., Sample 5: [11]
        sample_to_test_data_indices = [6, 7, 8, 9, 10, 11]
        expected_data = test_data_array[sample_to_test_data_indices]

        np.testing.assert_array_equal(actual_data, expected_data)


class TestMultiChannelVolumetric:
    """Test cases for multi-channel, volumetric data."""

    # Test data configuration - using shared 12-page data
    # With 2 channels and 3 planes: 12 pages ÷ (2 channels × 3 planes) = 2 samples (complete cycles)
    num_files = 2
    num_channels = 2
    num_planes = 3
    height = 3
    width = 4
    expected_samples = 2  # 12 pages ÷ (2 channels × 3 planes) = 2 complete samples
    expected_shape = (height, width)
    expected_volume_shape = (height, width, num_planes)
    sampling_frequency = 30.0

    def test_initialization(self, tiff_file_paths):
        """Test initialization of the extractor with multi-channel volumetric data."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_name="0",
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape
        assert extractor.get_sample_shape() == self.expected_volume_shape
        assert extractor.get_volume_shape() == self.expected_volume_shape
        assert extractor.get_sampling_frequency() == self.sampling_frequency
        assert extractor.get_num_planes() == self.num_planes
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric == True
        assert extractor.get_channel_names() == [str(i) for i in range(self.num_channels)]
        assert extractor.get_native_timestamps() is None

    def test_get_series_channel_0(self, tiff_file_paths, test_data_array):
        """Test get_series for channel 0 with TCZ dimension order."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_name="0",
            num_planes=self.num_planes,
        )

        actual_data = extractor.get_series()

        # Channel 0 sample to test_data_array mapping for TCZ order (2 channels, 3 planes)
        sample_to_test_data_mapping = {
            0: [0, 4, 8],  # Sample 0: T0C0Z0, T0C0Z1, T0C0Z2
            1: [1, 5, 9],  # Sample 1: T1C0Z0, T1C0Z1, T1C0Z2
        }

        # Compare each sample directly to the corresponding test_data_array values using explicit stacking
        for sample_index, test_data_indices in sample_to_test_data_mapping.items():
            # Stack the individual frames into a volume (depth as last dimension)
            frame_stack = []
            for frame_index in test_data_indices:
                frame_stack.append(test_data_array[frame_index])
            expected_volume = np.stack(frame_stack, axis=-1)
            np.testing.assert_array_equal(actual_data[sample_index], expected_volume)

    def test_get_series_channel_1(self, tiff_file_paths, test_data_array):
        """Test get_series for channel 1 with TCZ dimension order."""
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=tiff_file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_name="1",
            num_planes=self.num_planes,
        )

        actual_data = extractor.get_series()

        # Channel 1 sample to test_data_array mapping for TCZ order (2 channels, 3 planes)
        sample_to_test_data_mapping = {
            0: [2, 6, 10],  # Sample 0: T0C1Z0, T0C1Z1, T0C1Z2
            1: [3, 7, 11],  # Sample 1: T1C1Z0, T1C1Z1, T1C1Z2
        }

        # Compare each sample directly to the corresponding test_data_array values using explicit stacking
        for sample_index, test_data_indices in sample_to_test_data_mapping.items():
            # Stack the individual frames into a volume (depth as last dimension)
            frame_stack = []
            for frame_index in test_data_indices:
                frame_stack.append(test_data_array[frame_index])
            expected_volume = np.stack(frame_stack, axis=-1)
            np.testing.assert_array_equal(actual_data[sample_index], expected_volume)


@pytest.mark.parametrize(
    "dimension_order,channel_name,expected_sample_mappings",
    [
        # Channel 0 tests - Time-first acquisition patterns
        ("TCZ", "0", {0: [0, 4, 8], 1: [1, 5, 9]}),  # Time varies first: multiple samples per channel at each depth
        ("TZC", "0", {0: [0, 2, 4], 1: [1, 3, 5]}),  # Time varies first: multiple samples per depth for each channel
        # Channel 0 tests - Channel-first acquisition patterns
        ("CZT", "0", {0: [0, 2, 4], 1: [6, 8, 10]}),  # Channels cycle through depths before time advances
        ("CTZ", "0", {0: [0, 4, 8], 1: [2, 6, 10]}),  # All channels acquired multiple times per depth
        # Channel 0 tests - Depth-first acquisition patterns
        ("ZCT", "0", {0: [0, 1, 2], 1: [6, 7, 8]}),  # Complete Z-stack per channel before switching
        ("ZTC", "0", {0: [0, 1, 2], 1: [3, 4, 5]}),  # Complete Z-stack repeated over time per channel
        # Channel 1 tests - Time-first acquisition patterns
        ("TCZ", "1", {0: [2, 6, 10], 1: [3, 7, 11]}),  # Time varies first: multiple samples per channel at each depth
        ("TZC", "1", {0: [6, 8, 10], 1: [7, 9, 11]}),  # Time varies first: multiple samples per depth for each channel
        # Channel 1 tests - Channel-first acquisition patterns
        ("CZT", "1", {0: [1, 3, 5], 1: [7, 9, 11]}),  # Channels cycle through depths before time advances
        ("CTZ", "1", {0: [1, 5, 9], 1: [3, 7, 11]}),  # All channels acquired multiple times per depth
        # Channel 1 tests - Depth-first acquisition patterns
        ("ZCT", "1", {0: [3, 4, 5], 1: [9, 10, 11]}),  # Complete Z-stack per channel before switching
        ("ZTC", "1", {0: [6, 7, 8], 1: [9, 10, 11]}),  # Complete Z-stack repeated over time per channel
    ],
)
def test_dimension_order_variations(
    tiff_file_paths, test_data_array, dimension_order, channel_name, expected_sample_mappings
):
    """
    Comprehensive test for all supported dimension orders with multi-channel volumetric data.

    Tests all 6 dimension orders (TCZ, CZT, CTZ, ZCT, ZTC, TZC) with a stable
    2-channel, 3-plane configuration to validate that the extractor correctly
    interprets the same underlying TIFF data according to different acquisition
    patterns and dimension orders.
    """
    # Fixed configuration for all tests
    num_channels = 2
    num_planes = 3
    expected_samples = 2
    sampling_frequency = 30.0

    extractor = MultiTIFFMultiPageExtractor(
        file_paths=tiff_file_paths,
        sampling_frequency=sampling_frequency,
        dimension_order=dimension_order,
        num_channels=num_channels,
        channel_name=channel_name,
        num_planes=num_planes,
    )

    assert extractor.get_num_samples() == expected_samples
    assert extractor.get_num_planes() == num_planes

    # Get all samples and verify each one
    all_samples = extractor.get_series()

    for sample_index, expected_frames in expected_sample_mappings.items():
        actual_sample = all_samples[sample_index]

        # Volumetric data - stack frames into volume
        frame_stack = []
        for frame_index in expected_frames:
            frame_stack.append(test_data_array[frame_index])
        expected_volume = np.stack(frame_stack, axis=-1)

        np.testing.assert_array_equal(
            actual_sample,
            expected_volume,
            err_msg=f"Sample {sample_index} mismatch for {dimension_order}_ch{channel_name}",
        )


# General/cross-case tests that don't belong to a specific test case
def test_warning_for_indivisible_ifds(tmp_path):
    """Test that a warning is raised when total IFDs is not divisible by ifds_per_cycle."""
    import tifffile

    # Create a TIFF file with 5 pages (not divisible by 2 channels * 2 planes = 4)
    file_path = tmp_path / "test_indivisible.tif"
    with tifffile.TiffWriter(file_path) as writer:
        for i in range(5):
            data = np.full((3, 4), i, dtype=np.uint16)
            writer.write(data)

    # This should raise a warning because 5 is not divisible by 4
    with pytest.warns(UserWarning, match="Total IFDs .* is not divisible by IFDs per cycle"):
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=[file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=2,
            channel_name="0",
            num_planes=2,
        )

    # Should still work but only access 1 complete cycle (4 IFDs out of 5)
    assert extractor.get_num_samples() == 1


def test_invalid_num_channels_error_handling(tmp_path):
    """Test error handling for invalid num_channels."""
    # Use non-existent file since validation happens before file opening
    empty_file_path = tmp_path / "nonexistent.tif"

    # Test that num_channels < 1 raises ValueError
    with pytest.raises(ValueError, match="num_channels must be at least 1"):
        MultiTIFFMultiPageExtractor(
            file_paths=[empty_file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=0,
            num_planes=1,
        )


def test_invalid_num_planes_error_handling(tmp_path):
    """Test error handling for invalid num_planes."""
    # Use non-existent file since validation happens before file opening
    empty_file_path = tmp_path / "nonexistent.tif"

    # Test that num_planes < 1 raises ValueError
    with pytest.raises(ValueError, match="num_planes must be at least 1"):
        MultiTIFFMultiPageExtractor(
            file_paths=[empty_file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=0,
        )


def test_channel_name_validation_error_handling(tmp_path):
    """Test error handling for channel name validation."""
    # Use non-existent file since validation happens before file opening
    empty_file_path = tmp_path / "nonexistent.tif"

    # Test that missing channel_name when num_channels > 1 raises ValueError
    with pytest.raises(ValueError, match="channel_name must be specified when num_channels > 1"):
        MultiTIFFMultiPageExtractor(
            file_paths=[empty_file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=2,
            channel_name=None,
        )

    # Test that invalid channel_name format raises ValueError
    with pytest.raises(ValueError, match="Invalid channel name format.*Expected numeric format"):
        MultiTIFFMultiPageExtractor(
            file_paths=[empty_file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=2,
            channel_name="invalid_name",
        )

    # Test that channel_name out of range raises ValueError
    with pytest.raises(ValueError, match="channel_index 2 is out of range \\(0 to 1\\)"):
        MultiTIFFMultiPageExtractor(
            file_paths=[empty_file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=2,
            channel_name="2",
        )


def test_invalid_dimension_order_error_handling(tmp_path):
    """Test error handling for invalid dimension order."""
    tifffile = get_package(package_name="tifffile")

    # Create a simple test file
    file_path = tmp_path / "test.tif"
    data = np.ones((3, 4), dtype=np.uint16)
    tifffile.imwrite(file_path, data)

    with pytest.raises(ValueError):
        MultiTIFFMultiPageExtractor(
            file_paths=[file_path],
            sampling_frequency=30.0,
            dimension_order="INVALID",
            num_channels=1,
            num_planes=1,
        )


def test_corrupted_file_error_handling(tmp_path):
    """Test error handling when TIFF files are corrupted or invalid."""
    # Create a corrupted/invalid file by writing non-TIFF data
    corrupted_file_path = tmp_path / "corrupted.tif"
    with open(corrupted_file_path, "wb") as f:
        f.write(b"This is not a valid TIFF file")

    # Test that it raises a RuntimeError when encountering a corrupted file
    with pytest.raises(RuntimeError, match=r"Error opening TIFF file.*corrupted\.tif"):
        MultiTIFFMultiPageExtractor(
            file_paths=[corrupted_file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=1,
        )


def test_nonexistent_file_error_handling(tmp_path):
    """Test error handling for non-existent files."""
    nonexistent_file = tmp_path / "nonexistent.tif"
    with pytest.raises(FileNotFoundError, match=r"TIFF file not found:.*nonexistent\.tif"):
        MultiTIFFMultiPageExtractor(
            file_paths=[nonexistent_file],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=1,
        )


def test_from_folder_no_files_found_error(tmp_path):
    """Test error handling when from_folder finds no matching files."""
    # Test with a pattern that won't match any files
    with pytest.raises(ValueError, match=r"No files found matching pattern \*\.nonexistent"):
        MultiTIFFMultiPageExtractor.from_folder(
            folder_path=tmp_path,
            file_pattern="*.nonexistent",
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=1,
        )


################################################
# Test, reproducing other extractors to validate
################################################


def test_comparison_with_scanimage_planar_multi_channel():
    """Test comparison with ScanImageImagingExtractor for planar multi-channel data."""
    # Path to ScanImage multi-file data - explicit file paths for reproducibility
    scanimage_folder_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"
    file_paths = [
        scanimage_folder_path / "scanimage_20240320_multifile_00001.tif",
        scanimage_folder_path / "scanimage_20240320_multifile_00002.tif",
        scanimage_folder_path / "scanimage_20240320_multifile_00003.tif",
    ]

    # Use ScanImageImagingExtractor instead of deprecated extractor
    scanimage_extractor = ScanImageImagingExtractor(
        file_paths=file_paths,
        channel_name="Channel 1",
    )

    # Initialize MultiTIFFMultiPageExtractor with equivalent mapping
    multi_extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=scanimage_extractor.get_sampling_frequency(),
        dimension_order="CZT",  # ScanImage uses this order
        num_channels=2,
        channel_name="0",
        # num_planes=1 omitted since it's the default for planar data
    )

    # Compare basic properties
    assert multi_extractor.get_num_samples() == scanimage_extractor.get_num_samples()

    # Compare frame shapes - ScanImage includes depth dimension, MultiTIFF doesn't
    scanimage_shape = scanimage_extractor.get_frame_shape()
    multi_shape = multi_extractor.get_frame_shape()

    # Check that the first two dimensions (height, width) match
    assert multi_shape[0] == scanimage_shape[0]
    assert multi_shape[1] == scanimage_shape[1]

    assert multi_extractor.get_sampling_frequency() == scanimage_extractor.get_sampling_frequency()

    # Compare the actual series data
    scanimage_series = scanimage_extractor.get_series()
    multi_series = multi_extractor.get_series()

    assert_allclose(scanimage_series, multi_series, rtol=1e-5, atol=1e-8)


def test_comparison_with_scanimage_volumetric_multi_channel():
    """Test comparison with volumetric ScanImageImagingExtractor data.

    Note: ScanImage removes flyback frames automatically, while MultiTIFFMultiPageExtractor
    includes all frames. To compare equivalent data, we configure MultiTIFF with
    num_planes = scan_planes + flyback_frames, then slice away the flyback frames
    from MultiTIFF volumes to match ScanImage's processed output.
    """
    # Path to volumetric ScanImage data
    scanimage_folder_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"
    volumetric_path = (
        scanimage_folder_path / "volumetric_two_channels_single_file" / "vol_two_ch_single_file_00001_00001.tif"
    )

    # Initialize ScanImage extractor for Channel 1 (index 0)
    scanimage_extractor = ScanImageImagingExtractor(
        file_path=volumetric_path,
        channel_name="Channel 1",
    )

    # Get ScanImage metadata
    num_planes_with_data = scanimage_extractor.get_num_planes()  # Should be 9 (actual imaging planes)
    flyback_frames = scanimage_extractor.num_flyback_frames_per_channel  # Should be 7
    sampling_frequency = scanimage_extractor.get_sampling_frequency()

    # Initialize MultiTIFFMultiPageExtractor with raw frame count (including flybacks)
    # ScanImage uses CZT order (channels cycle through depths, then time)
    multi_extractor = MultiTIFFMultiPageExtractor(
        file_paths=[volumetric_path],
        sampling_frequency=sampling_frequency,
        dimension_order="CZT",
        num_channels=2,  # Channel 1 and Channel 2
        channel_name="0",  # Channel 1 corresponds to index 0
        num_planes=num_planes_with_data + flyback_frames,  # Include flyback frames in raw count
    )

    # Compare basic properties (after accounting for flyback frame removal)
    assert multi_extractor.get_frame_shape() == scanimage_extractor.get_frame_shape()
    assert multi_extractor.get_sampling_frequency() == scanimage_extractor.get_sampling_frequency()
    assert multi_extractor.is_volumetric == scanimage_extractor.is_volumetric

    # Both extractors should have the same number of samples when processing the same data correctly
    assert scanimage_extractor.get_num_samples() == multi_extractor.get_num_samples()

    # Get entire series from both extractors
    scanimage_series = scanimage_extractor.get_series()
    multi_raw_series = multi_extractor.get_series()

    # Slice away flyback frames from MultiTIFF series to match ScanImage processing
    # Keep only the first num_planes_with_data (remove the flyback_frames at the end)
    multi_series_without_flyback = multi_raw_series[:, :, :, :num_planes_with_data]

    # Compare the entire processed series
    assert_allclose(
        scanimage_series,
        multi_series_without_flyback,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Complete series do not match between ScanImage and MultiTIFF extractors",
    )
