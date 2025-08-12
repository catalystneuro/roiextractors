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


class TestSingleChannelPlanar:
    """Test cases for single channel, planar (num_planes=1) data."""

    # Test data configuration
    num_files = 2
    num_timepoints = 4
    num_channels = 1
    num_planes = 1
    height = 10
    width = 12
    expected_samples = num_files * num_timepoints  # 8
    expected_shape = (height, width)
    sampling_frequency = 30.0

    @pytest.fixture(scope="class")
    def test_files(self, tmp_path_factory):
        """Create single-channel planar TIFF files for testing."""
        tmp_path = tmp_path_factory.mktemp("single_channel_planar")
        import tifffile

        for file_index in range(self.num_files):
            file_path = tmp_path / f"test_file_{file_index}.tif"

            pages = []
            for time in range(self.num_timepoints):
                for channel in range(self.num_channels):
                    for depth in range(self.num_planes):
                        value = file_index * 100 + time * 10 + channel * 5 + depth + 1
                        page_data = np.ones((self.height, self.width), dtype=np.uint16) * value
                        pages.append(page_data)

            # Write pages individually to create true multi-page TIFF
            with tifffile.TiffWriter(file_path) as writer:
                for page in pages:
                    writer.write(page)

        return tmp_path

    def test_initialization(self, test_files):
        """Test initialization of the extractor with single-channel planar data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape
        assert extractor.get_sampling_frequency() == self.sampling_frequency
        assert extractor.get_num_planes() == self.num_planes
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric == False

    def test_get_series(self, test_files):
        """Test retrieving series from the extractor with planar data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        # Get specific samples
        sample_0 = extractor.get_series(start_sample=0, end_sample=1)
        sample_1 = extractor.get_series(start_sample=1, end_sample=2)

        # Check sample shapes - should NOT include depth dimension for planar data
        assert sample_0.shape == (1, self.height, self.width)
        assert sample_1.shape == (1, self.height, self.width)

        # Get multiple samples
        samples = extractor.get_series(start_sample=0, end_sample=3)
        assert samples.shape == (3, self.height, self.width)

        # Get all samples
        all_samples = extractor.get_series()
        assert all_samples.ndim == 3  # (samples, height, width)

    def test_get_channel_names(self, test_files):
        """Test getting channel names for single channel data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        channel_names = extractor.get_channel_names()
        assert channel_names == [f"Channel {i}" for i in range(self.num_channels)]


class TestSingleChannelVolumetric:
    """Test cases for single channel, volumetric (num_planes>1) data."""

    # Test data configuration
    num_files = 2
    num_timepoints = 2
    num_channels = 1
    num_planes = 3
    height = 10
    width = 12
    expected_samples = num_files * num_timepoints  # 4
    expected_shape = (height, width)
    expected_volume_shape = (height, width, num_planes)
    sampling_frequency = 30.0

    @pytest.fixture(scope="class")
    def test_files(self, tmp_path_factory):
        """Create single-channel volumetric TIFF files for testing."""
        tmp_path = tmp_path_factory.mktemp("single_channel_volumetric")
        tifffile = get_package(package_name="tifffile")

        for file_index in range(self.num_files):
            file_path = tmp_path / f"test_file_{file_index}.tif"

            pages = []
            for time in range(self.num_timepoints):
                for channel in range(self.num_channels):
                    for depth in range(self.num_planes):
                        value = file_index * 100 + time * 10 + channel * 5 + depth + 1
                        page_data = np.ones((self.height, self.width), dtype=np.uint16) * value
                        pages.append(page_data)

            # Write pages individually to create true multi-page TIFF
            with tifffile.TiffWriter(file_path) as writer:
                for page in pages:
                    writer.write(page)

        return tmp_path

    def test_initialization(self, test_files):
        """Test initialization of the extractor with single-channel volumetric data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape
        assert extractor.get_sampling_frequency() == self.sampling_frequency
        assert extractor.get_num_planes() == self.num_planes
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric == True

    def test_get_series(self, test_files):
        """Test retrieving series from the extractor with volumetric data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        # Get specific samples
        sample_0 = extractor.get_series(start_sample=0, end_sample=1)
        sample_1 = extractor.get_series(start_sample=1, end_sample=2)

        # Check sample shapes - should include depth dimension
        assert sample_0.shape == (1, self.height, self.width, self.num_planes)
        assert sample_1.shape == (1, self.height, self.width, self.num_planes)

        # Get multiple samples
        samples = extractor.get_series(start_sample=0, end_sample=2)
        assert samples.shape == (2, self.height, self.width, self.num_planes)

        # Get all samples
        all_samples = extractor.get_series()
        assert all_samples.shape == (self.expected_samples, self.height, self.width, self.num_planes)

    def test_from_folder(self, test_files):
        """Test creating an extractor from a folder path."""
        extractor = MultiTIFFMultiPageExtractor.from_folder(
            folder_path=test_files,
            file_pattern="*.tif",
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape

    def test_get_channel_names(self, test_files):
        """Test getting channel names for single channel data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        channel_names = extractor.get_channel_names()
        assert channel_names == [f"Channel {i}" for i in range(self.num_channels)]

    def test_get_native_timestamps(self, test_files):
        """Test that native timestamps returns None."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        timestamps = extractor.get_native_timestamps()
        assert timestamps is None

        # Test with start/end parameters
        timestamps = extractor.get_native_timestamps(start_sample=0, end_sample=2)
        assert timestamps is None

    def test_volume_shape_getter(self, test_files):
        """Test get_volume_shape method."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="CZT",
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        volume_shape = extractor.get_volume_shape()
        assert volume_shape == self.expected_volume_shape

    @pytest.mark.parametrize("dimension_order", ["CZT", "ZCT", "TCZ", "TZC"])
    def test_dimension_order_variations(self, test_files, dimension_order):
        """Test various dimension order configurations."""
        file_paths = sorted(list(test_files.glob("*.tif")))

        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order=dimension_order,
            num_channels=self.num_channels,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        series = extractor.get_series(start_sample=0, end_sample=1)
        assert series.ndim == 4  # (samples, height, width, depth)

    def test_invalid_num_planes_error_handling(self, test_files):
        """Test error handling for invalid num_planes."""
        file_paths = sorted(list(test_files.glob("*.tif")))

        # Test that num_planes < 1 raises ValueError
        with pytest.raises(ValueError, match="num_planes must be at least 1"):
            MultiTIFFMultiPageExtractor(
                file_paths=file_paths,
                sampling_frequency=self.sampling_frequency,
                dimension_order="CZT",
                num_channels=self.num_channels,
                num_planes=0,
            )


class TestMultiChannelPlanar:
    """Test cases for multi-channel, planar data."""

    # Test data configuration
    num_files = 2
    num_timepoints = 2
    num_channels = 2
    num_planes = 2
    height = 10
    width = 12
    expected_samples = num_files * num_timepoints  # 4
    expected_shape = (height, width)
    expected_volume_shape = (height, width, num_planes)  # Still volumetric due to num_planes=2
    sampling_frequency = 30.0

    @pytest.fixture(scope="class")
    def test_files(self, tmp_path_factory):
        """Create multi-channel planar TIFF files for testing."""
        tmp_path = tmp_path_factory.mktemp("multi_channel_planar")
        tifffile = get_package(package_name="tifffile")

        for file_index in range(self.num_files):
            file_path = tmp_path / f"test_file_{file_index}.tif"

            pages = []
            for time in range(self.num_timepoints):
                for channel in range(self.num_channels):
                    for depth in range(self.num_planes):
                        value = file_index * 100 + time * 10 + channel * 5 + depth + 1
                        page_data = np.ones((self.height, self.width), dtype=np.uint16) * value
                        pages.append(page_data)

            # Write pages individually to create true multi-page TIFF
            with tifffile.TiffWriter(file_path) as writer:
                for page in pages:
                    writer.write(page)

        return tmp_path

    def test_initialization(self, test_files):
        """Test initialization of the extractor with multi-channel planar data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=1,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape
        assert extractor.get_sampling_frequency() == self.sampling_frequency
        assert extractor.get_num_planes() == self.num_planes
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric == True

    def test_channel_extraction(self, test_files):
        """Test extraction of specific channels from planar data."""
        file_paths = sorted(list(test_files.glob("*.tif")))

        # Test extracting first channel
        extractor_ch0 = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=0,
            num_planes=self.num_planes,
        )

        # Test extracting second channel
        extractor_ch1 = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=1,
            num_planes=self.num_planes,
        )

        assert extractor_ch0.get_num_samples() == self.expected_samples
        assert extractor_ch1.get_num_samples() == self.expected_samples

        # Verify different channels have different data
        series_ch0 = extractor_ch0.get_series()
        series_ch1 = extractor_ch1.get_series()

        assert not np.array_equal(series_ch0, series_ch1)
        assert series_ch0.shape == (self.expected_samples, self.height, self.width, self.num_planes)
        assert series_ch1.shape == (self.expected_samples, self.height, self.width, self.num_planes)

    def test_invalid_channel_index_error_handling(self, test_files):
        """Test error handling for invalid channel_index."""
        file_paths = sorted(list(test_files.glob("*.tif")))

        # Test that channel_index >= num_channels raises ValueError
        with pytest.raises(
            ValueError, match=rf"channel_index {self.num_channels} is out of range \(0 to {self.num_channels-1}\)"
        ):
            MultiTIFFMultiPageExtractor(
                file_paths=file_paths,
                sampling_frequency=self.sampling_frequency,
                dimension_order="TCZ",
                num_channels=self.num_channels,
                channel_index=self.num_channels,  # Invalid: equal to num_channels
                num_planes=self.num_planes,
            )


class TestMultiChannelVolumetric:
    """Test cases for multi-channel, volumetric data."""

    # Test data configuration
    num_files = 2
    num_timepoints = 2
    num_channels = 2
    num_planes = 3
    height = 10
    width = 12
    expected_samples = num_files * num_timepoints  # 4
    expected_shape = (height, width)
    expected_volume_shape = (height, width, num_planes)
    sampling_frequency = 30.0

    @pytest.fixture(scope="class")
    def test_files(self, tmp_path_factory):
        """Create multi-channel volumetric TIFF files for testing."""
        tmp_path = tmp_path_factory.mktemp("multi_channel_volumetric")
        tifffile = get_package(package_name="tifffile")

        for file_index in range(self.num_files):
            file_path = tmp_path / f"test_file_{file_index}.tif"

            pages = []
            for time in range(self.num_timepoints):
                for channel in range(self.num_channels):
                    for depth in range(self.num_planes):
                        value = file_index * 100 + time * 10 + channel * 5 + depth + 1
                        page_data = np.ones((self.height, self.width), dtype=np.uint16) * value
                        pages.append(page_data)

            # Write pages individually to create true multi-page TIFF
            with tifffile.TiffWriter(file_path) as writer:
                for page in pages:
                    writer.write(page)

        return tmp_path

    def test_initialization(self, test_files):
        """Test initialization of the extractor with multi-channel volumetric data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=0,
            num_planes=self.num_planes,
        )

        assert extractor.get_num_samples() == self.expected_samples
        assert extractor.get_frame_shape() == self.expected_shape
        assert extractor.get_sampling_frequency() == self.sampling_frequency
        assert extractor.get_num_planes() == self.num_planes
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric == True

    def test_channel_extraction(self, test_files):
        """Test extraction of specific channels."""
        file_paths = sorted(list(test_files.glob("*.tif")))

        # Test extracting first channel
        extractor_ch0 = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=0,
            num_planes=self.num_planes,
        )

        # Test extracting second channel
        extractor_ch1 = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=1,
            num_planes=self.num_planes,
        )

        assert extractor_ch0.get_num_samples() == self.expected_samples
        assert extractor_ch1.get_num_samples() == self.expected_samples

        # Verify different channels have different data
        series_ch0 = extractor_ch0.get_series()
        series_ch1 = extractor_ch1.get_series()

        assert not np.array_equal(series_ch0, series_ch1)
        assert series_ch0.shape == (self.expected_samples, self.height, self.width, self.num_planes)
        assert series_ch1.shape == (self.expected_samples, self.height, self.width, self.num_planes)

    def test_data_consistency_across_channels(self, test_files):
        """Test that data values are consistent with the generation pattern."""
        file_paths = sorted(list(test_files.glob("*.tif")))

        extractor_ch0 = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=0,
            num_planes=self.num_planes,
        )

        extractor_ch1 = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=1,
            num_planes=self.num_planes,
        )

        # Get first sample from each channel
        sample_ch0 = extractor_ch0.get_series(start_sample=0, end_sample=1)
        sample_ch1 = extractor_ch1.get_series(start_sample=0, end_sample=1)

        # Check that channel 1 has higher values than channel 0 (due to generation pattern)
        # Channel 1 should have values offset by 5 compared to channel 0
        assert np.all(sample_ch1 > sample_ch0)

    def test_get_channel_names(self, test_files):
        """Test getting channel names for multi channel data."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=0,
            num_planes=self.num_planes,
        )

        channel_names = extractor.get_channel_names()
        assert channel_names == [f"Channel {i}" for i in range(self.num_channels)]

    def test_volume_shape_getter(self, test_files):
        """Test get_volume_shape method."""
        file_paths = sorted(list(test_files.glob("*.tif")))
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=self.sampling_frequency,
            dimension_order="TCZ",
            num_channels=self.num_channels,
            channel_index=0,
            num_planes=self.num_planes,
        )

        volume_shape = extractor.get_volume_shape()
        assert volume_shape == self.expected_volume_shape


# General/cross-case tests that don't belong to a specific test case
def test_warning_for_indivisible_ifds(tmp_path):
    """Test that a warning is raised when total IFDs is not divisible by ifds_per_cycle."""
    import tifffile

    # Create a TIFF file with 5 pages (not divisible by 2 channels * 2 planes = 4)
    file_path = tmp_path / "test_indivisible.tif"
    with tifffile.TiffWriter(file_path) as writer:
        for i in range(5):
            data = np.ones((10, 12), dtype=np.uint16) * i
            writer.write(data)

    # This should raise a warning because 5 is not divisible by 4
    with pytest.warns(UserWarning, match="Total IFDs .* is not divisible by IFDs per cycle"):
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=[file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=2,
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


def test_comparison_with_scanimage():
    """Test comparison with ScanImageImagingExtractor."""
    # Path to ScanImage folder and file pattern
    scanimage_folder_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"
    multifile_file_pattern = "scanimage_20240320_multifile_*.tif"

    file_paths = sorted(list(scanimage_folder_path.glob(multifile_file_pattern)))

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
        channel_index=0,
        num_planes=1,  # For this example, num_planes is 1
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
    scanimage_series = scanimage_extractor.get_series().squeeze()
    multi_series = multi_extractor.get_series()

    assert_allclose(scanimage_series, multi_series, rtol=1e-5, atol=1e-8)


def test_invalid_dimension_order_error_handling(tmp_path):
    """Test error handling for invalid dimension order."""
    tifffile = get_package(package_name="tifffile")

    # Create a simple test file
    file_path = tmp_path / "test.tif"
    data = np.ones((10, 12), dtype=np.uint16)
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
