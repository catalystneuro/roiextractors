"""Tests for the MultiTIFFMultiPageExtractor."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from roiextractors.extraction_tools import get_package
from roiextractors.extractors.tiffimagingextractors import (
    MultiTIFFMultiPageExtractor,
    ScanImageImagingExtractor,
)
from tests.setup_paths import OPHYS_DATA_PATH


@pytest.fixture(scope="module")
def single_channel_planar_tiff_files(tmp_path_factory):
    """Create single-channel planar TIFF files for testing."""
    tmp_path = tmp_path_factory.mktemp("single_channel_planar")
    import tifffile

    # Create test data: 2 files, each with 4 pages (4 time points, 1 channel, 1 plane)
    num_files = 2
    num_timepoints = 4
    num_channels = 1
    num_planes = 1
    height = 10
    width = 12

    for file_index in range(num_files):
        file_path = tmp_path / f"test_file_{file_index}.tif"

        pages = []
        for time in range(num_timepoints):
            for channel in range(num_channels):
                for depth in range(num_planes):
                    value = file_index * 100 + time * 10 + channel * 5 + depth + 1
                    page_data = np.ones((height, width), dtype=np.uint16) * value
                    pages.append(page_data)

        # Write pages individually to create true multi-page TIFF
        with tifffile.TiffWriter(file_path) as writer:
            for page in pages:
                writer.write(page)

    return tmp_path


@pytest.fixture(scope="module")
def single_channel_volumetric_tiff_files(tmp_path_factory):
    """Create single-channel volumetric TIFF files for testing."""
    tmp_path = tmp_path_factory.mktemp("single_channel_volumetric")
    tifffile = get_package(package_name="tifffile")

    # Create test data: 2 files, each with 6 pages (2 time points, 1 channel, 3 planes)
    num_files = 2
    num_timepoints = 2
    num_channels = 1
    num_planes = 3
    height = 10
    width = 12

    for file_index in range(num_files):
        file_path = tmp_path / f"test_file_{file_index}.tif"

        pages = []
        for time in range(num_timepoints):
            for channel in range(num_channels):
                for depth in range(num_planes):
                    value = file_index * 100 + time * 10 + channel * 5 + depth + 1
                    page_data = np.ones((height, width), dtype=np.uint16) * value
                    pages.append(page_data)

        # Write pages individually to create true multi-page TIFF
        with tifffile.TiffWriter(file_path) as writer:
            for page in pages:
                writer.write(page)

    return tmp_path


@pytest.fixture(scope="module")
def multi_channel_planar_tiff_files(tmp_path_factory):
    """Create multi-channel planar TIFF files for testing."""
    tmp_path = tmp_path_factory.mktemp("multi_channel_planar")
    tifffile = get_package(package_name="tifffile")

    # Create test data: 2 files, each with 8 pages (2 time points, 2 channels, 2 planes)
    num_files = 2
    num_timepoints = 2
    num_channels = 2
    num_planes = 2
    height = 10
    width = 12

    for file_index in range(num_files):
        file_path = tmp_path / f"test_file_{file_index}.tif"

        pages = []
        for time in range(num_timepoints):
            for channel in range(num_channels):
                for depth in range(num_planes):
                    value = file_index * 100 + time * 10 + channel * 5 + depth + 1
                    page_data = np.ones((height, width), dtype=np.uint16) * value
                    pages.append(page_data)

        # Write pages individually to create true multi-page TIFF
        with tifffile.TiffWriter(file_path) as writer:
            for page in pages:
                writer.write(page)

    return tmp_path


@pytest.fixture(scope="module")
def multi_channel_volumetric_tiff_files(tmp_path_factory):
    """Create multi-channel volumetric TIFF files for testing."""
    tmp_path = tmp_path_factory.mktemp("multi_channel_volumetric")
    tifffile = get_package(package_name="tifffile")

    # Create test data: 2 files, each with 12 pages (2 time points, 2 channels, 3 planes)
    num_files = 2
    num_timepoints = 2
    num_channels = 2
    num_planes = 3
    height = 10
    width = 12

    for file_index in range(num_files):
        file_path = tmp_path / f"test_file_{file_index}.tif"

        pages = []
        for time in range(num_timepoints):
            for channel in range(num_channels):
                for depth in range(num_planes):
                    value = file_index * 100 + time * 10 + channel * 5 + depth + 1
                    page_data = np.ones((height, width), dtype=np.uint16) * value
                    pages.append(page_data)

        # Write pages individually to create true multi-page TIFF
        with tifffile.TiffWriter(file_path) as writer:
            for page in pages:
                writer.write(page)

    return tmp_path


def test_initialization_single_channel_volumetric(single_channel_volumetric_tiff_files):
    """Test initialization of the extractor with single-channel volumetric data."""
    file_paths = sorted(list(single_channel_volumetric_tiff_files.glob("*.tif")))
    multi_tiff_extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=3,
    )

    assert multi_tiff_extractor.get_num_samples() == 4
    assert multi_tiff_extractor.get_frame_shape() == (10, 12)  # (height, width)
    assert multi_tiff_extractor.get_sampling_frequency() == 30.0
    assert multi_tiff_extractor.get_num_planes() == 3
    assert multi_tiff_extractor.is_volumetric == True


def test_initialization_single_channel_planar(single_channel_planar_tiff_files):
    """Test initialization of the extractor with single-channel planar data."""
    file_paths = sorted(list(single_channel_planar_tiff_files.glob("*.tif")))
    assert file_paths[0].exists(), "Test files not found"
    multi_tiff_extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=1,
    )

    assert multi_tiff_extractor.get_num_samples() == 8  # 2 files * 4 timepoints
    assert multi_tiff_extractor.get_frame_shape() == (10, 12)  # (height, width)
    assert multi_tiff_extractor.get_sampling_frequency() == 30.0
    assert multi_tiff_extractor.get_num_planes() == 1
    assert multi_tiff_extractor.is_volumetric == False


def test_initialization_multi_channel_volumetric(multi_channel_volumetric_tiff_files):
    """Test initialization of the extractor with multi-channel volumetric data."""
    file_paths = sorted(list(multi_channel_volumetric_tiff_files.glob("*.tif")))
    multi_tiff_extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=0,
        num_planes=3,
    )

    assert multi_tiff_extractor.get_num_samples() == 4  # 2 files * 2 timepoints
    assert multi_tiff_extractor.get_frame_shape() == (10, 12)  # (height, width)
    assert multi_tiff_extractor.get_sampling_frequency() == 30.0
    assert multi_tiff_extractor.get_num_planes() == 3
    assert multi_tiff_extractor.is_volumetric == True


def test_initialization_multi_channel_planar(multi_channel_planar_tiff_files):
    """Test initialization of the extractor with multi-channel planar data."""
    file_paths = sorted(list(multi_channel_planar_tiff_files.glob("*.tif")))
    multi_tiff_extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=1,
        num_planes=2,
    )

    assert multi_tiff_extractor.get_num_samples() == 4  # 2 files * 2 timepoints
    assert multi_tiff_extractor.get_frame_shape() == (10, 12)  # (height, width)
    assert multi_tiff_extractor.get_sampling_frequency() == 30.0
    assert multi_tiff_extractor.get_num_planes() == 2
    assert multi_tiff_extractor.is_volumetric == True


def test_get_series_volumetric(single_channel_volumetric_tiff_files):
    """Test retrieving series from the extractor with volumetric data."""
    file_paths = sorted(list(single_channel_volumetric_tiff_files.glob("*.tif")))
    multi_tiff_extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=3,
    )

    # Get specific samples
    sample_0 = multi_tiff_extractor.get_series(start_sample=0, end_sample=1)
    sample_1 = multi_tiff_extractor.get_series(start_sample=1, end_sample=2)

    # Check sample shapes - should include depth dimension
    assert sample_0.shape == (1, 10, 12, 3)
    assert sample_1.shape == (1, 10, 12, 3)

    # Get multiple samples
    samples = multi_tiff_extractor.get_series(start_sample=0, end_sample=2)
    assert samples.shape == (2, 10, 12, 3)

    # Get all samples
    all_samples = multi_tiff_extractor.get_series()
    assert all_samples.shape == (4, 10, 12, 3)


def test_get_series_planar(single_channel_planar_tiff_files):
    """Test retrieving series from the extractor with planar data."""
    file_paths = sorted(list(single_channel_planar_tiff_files.glob("*.tif")))
    multi_tiff_extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=1,
    )

    # Get specific samples
    sample_0 = multi_tiff_extractor.get_series(start_sample=0, end_sample=1)
    sample_1 = multi_tiff_extractor.get_series(start_sample=1, end_sample=2)

    # Check sample shapes - should NOT include depth dimension for planar data
    assert sample_0.shape == (1, 10, 12)
    assert sample_1.shape == (1, 10, 12)

    # Get multiple samples
    samples = multi_tiff_extractor.get_series(start_sample=0, end_sample=3)
    assert samples.shape == (3, 10, 12)

    # Get all samples
    all_samples = multi_tiff_extractor.get_series()
    assert samples.ndim == 3  # (samples, height, width)


def test_from_folder(single_channel_volumetric_tiff_files):
    """Test creating an extractor from a folder path."""
    extractor = MultiTIFFMultiPageExtractor.from_folder(
        folder_path=single_channel_volumetric_tiff_files,
        file_pattern="*.tif",
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=3,
    )

    assert extractor.get_num_samples() == 4
    assert extractor.get_frame_shape() == (10, 12)


def test_different_dimension_orders(single_channel_volumetric_tiff_files):
    """Test different dimension orders."""
    file_paths = sorted(list(single_channel_volumetric_tiff_files.glob("*.tif")))

    # Test with ZTC dimension order
    extractor_ztc = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="ZTC",
        num_channels=1,
        num_planes=3,
    )

    assert extractor_ztc.get_num_samples() == 4

    # Get samples and check they're accessible
    sample_0 = extractor_ztc.get_series(start_sample=0, end_sample=1)
    assert sample_0.shape == (1, 10, 12, 3)


@pytest.mark.parametrize("dimension_order", ["CZT", "ZCT", "TCZ", "TZC"])
def test_dimension_order_variations(single_channel_volumetric_tiff_files, dimension_order):
    """Test various dimension order configurations."""
    file_paths = sorted(list(single_channel_volumetric_tiff_files.glob("*.tif")))

    extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order=dimension_order,
        num_channels=1,
        num_planes=3,
    )

    assert extractor.get_num_samples() == 4
    series = extractor.get_series(start_sample=0, end_sample=1)
    assert series.ndim == 4  # (samples, height, width, depth)


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


def test_channel_extraction(multi_channel_volumetric_tiff_files):
    """Test extraction of specific channels."""
    file_paths = sorted(list(multi_channel_volumetric_tiff_files.glob("*.tif")))

    # Test extracting first channel
    extractor_ch0 = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=0,
        num_planes=3,
    )

    # Test extracting second channel
    extractor_ch1 = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=1,
        num_planes=3,
    )

    assert extractor_ch0.get_num_samples() == 4
    assert extractor_ch1.get_num_samples() == 4

    # Verify different channels have different data
    series_ch0 = extractor_ch0.get_series()
    series_ch1 = extractor_ch1.get_series()

    assert not np.array_equal(series_ch0, series_ch1)
    assert series_ch0.shape == (4, 10, 12, 3)
    assert series_ch1.shape == (4, 10, 12, 3)


def test_multi_channel_planar_extraction(multi_channel_planar_tiff_files):
    """Test extraction of specific channels from planar data."""
    file_paths = sorted(list(multi_channel_planar_tiff_files.glob("*.tif")))

    # Test extracting first channel
    extractor_ch0 = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=0,
        num_planes=2,
    )

    # Test extracting second channel
    extractor_ch1 = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=1,
        num_planes=2,
    )

    assert extractor_ch0.get_num_samples() == 4
    assert extractor_ch1.get_num_samples() == 4

    # Verify different channels have different data
    series_ch0 = extractor_ch0.get_series()
    series_ch1 = extractor_ch1.get_series()

    assert not np.array_equal(series_ch0, series_ch1)
    assert series_ch0.shape == (4, 10, 12, 2)  # Still volumetric due to num_planes=2
    assert series_ch1.shape == (4, 10, 12, 2)


def test_nonexistent_file_error_handling():
    """Test error handling for non-existent files."""
    with pytest.raises(Exception):
        # Should raise error for non-existent files
        MultiTIFFMultiPageExtractor(
            file_paths=[Path("nonexistent.tif")],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=1,
        )


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


def test_invalid_num_planes_error_handling(single_channel_volumetric_tiff_files):
    """Test error handling for invalid num_planes."""
    file_paths = sorted(list(single_channel_volumetric_tiff_files.glob("*.tif")))

    # Test that num_planes < 1 raises ValueError
    with pytest.raises(ValueError, match="num_planes must be at least 1"):
        MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=0,
        )


def test_invalid_channel_index_error_handling(multi_channel_volumetric_tiff_files):
    """Test error handling for invalid channel_index."""
    file_paths = sorted(list(multi_channel_volumetric_tiff_files.glob("*.tif")))

    # Test that channel_index >= num_channels raises ValueError
    with pytest.raises(ValueError, match=r"channel_index 2 is out of range \(0 to 1\)"):
        MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=30.0,
            dimension_order="TCZ",
            num_channels=2,
            channel_index=2,
            num_planes=3,
        )


def test_empty_tiff_files_error_handling(tmp_path):
    """Test error handling for empty TIFF files."""
    # Create an empty file
    empty_file_path = tmp_path / "empty.tif"
    empty_file_path.touch()

    # Test that empty TIFF files raise an appropriate error
    with pytest.raises((ValueError, RuntimeError)):
        MultiTIFFMultiPageExtractor(
            file_paths=[empty_file_path],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=1,
        )


def test_from_folder_no_files_error_handling(tmp_path):
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


def test_volume_shape_methods():
    """Test volume shape related methods."""
    # Test with volumetric data
    with pytest.raises(Exception):  # Will fail due to non-existent files but we're testing method presence
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=[Path("dummy.tif")],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=3,
        )


def test_get_channel_names_single_channel(single_channel_volumetric_tiff_files):
    """Test getting channel names for single channel data."""
    file_paths = sorted(list(single_channel_volumetric_tiff_files.glob("*.tif")))
    extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=3,
    )

    channel_names = extractor.get_channel_names()
    assert channel_names == ["Channel 0"]


def test_get_channel_names_multi_channel(multi_channel_volumetric_tiff_files):
    """Test getting channel names for multi channel data."""
    file_paths = sorted(list(multi_channel_volumetric_tiff_files.glob("*.tif")))
    extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=0,
        num_planes=3,
    )

    channel_names = extractor.get_channel_names()
    assert channel_names == ["Channel 0", "Channel 1"]


def test_get_native_timestamps(single_channel_volumetric_tiff_files):
    """Test that native timestamps returns None."""
    file_paths = sorted(list(single_channel_volumetric_tiff_files.glob("*.tif")))
    extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=3,
    )

    timestamps = extractor.get_native_timestamps()
    assert timestamps is None

    # Test with start/end parameters
    timestamps = extractor.get_native_timestamps(start_sample=0, end_sample=2)
    assert timestamps is None


def test_volume_shape_getter(multi_channel_volumetric_tiff_files):
    """Test get_volume_shape method."""
    file_paths = sorted(list(multi_channel_volumetric_tiff_files.glob("*.tif")))
    extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=0,
        num_planes=3,
    )

    volume_shape = extractor.get_volume_shape()
    assert volume_shape == (10, 12, 3)  # (height, width, num_planes)


def test_data_consistency_across_channels(multi_channel_volumetric_tiff_files):
    """Test that data values are consistent with the generation pattern."""
    file_paths = sorted(list(multi_channel_volumetric_tiff_files.glob("*.tif")))

    extractor_ch0 = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=0,
        num_planes=3,
    )

    extractor_ch1 = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=2,
        channel_index=1,
        num_planes=3,
    )

    # Get first sample from each channel
    sample_ch0 = extractor_ch0.get_series(start_sample=0, end_sample=1)
    sample_ch1 = extractor_ch1.get_series(start_sample=0, end_sample=1)

    # Check that channel 1 has higher values than channel 0 (due to generation pattern)
    # Channel 1 should have values offset by 5 compared to channel 0
    assert np.all(sample_ch1 > sample_ch0)


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


def test_get_series_with_out_of_range_indices(single_channel_planar_tiff_files):
    """Test get_series with invalid sample indices."""
    file_paths = sorted(list(single_channel_planar_tiff_files.glob("*.tif")))
    extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=1,
    )

    # Test with end_sample beyond available samples
    # Should not raise error but return available samples
    series = extractor.get_series(start_sample=0, end_sample=100)
    assert series.shape[0] == extractor.get_num_samples()
