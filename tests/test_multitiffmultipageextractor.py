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


@pytest.fixture
def temp_tiff_files(tmp_path):
    """Create test TIFF files for testing."""
    tifffile = get_package(package_name="tifffile")

    # Create test data
    # 2 files, each with 6 pages (2 time points, 1 channel, 3 depths)
    # That is, each file has 2 volumes for 4 samples in total
    num_files = 2
    acquisitions_per_file = 2
    num_channels = 1
    num_planes = 3
    height = 10
    width = 12

    for file_idx in range(num_files):
        file_path = tmp_path / f"test_file_{file_idx}.tif"

        # Create pages for this file
        pages = []
        for t in range(acquisitions_per_file):
            for c in range(num_channels):
                for z in range(num_planes):
                    # Create a unique pattern for each page
                    page_data = np.ones((height, width), dtype=np.uint16) * (file_idx * 100 + t * 10 + z + 1)
                    pages.append(page_data)

        # Write pages to TIFF file
        tifffile.imwrite(file_path, pages)

    return tmp_path


@pytest.fixture
def multi_tiff_extractor(temp_tiff_files):
    """Create a MultiTIFFMultiPageExtractor instance."""
    file_paths = sorted(list(temp_tiff_files.glob("*.tif")))
    return MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=3,
    )


def test_initialization(multi_tiff_extractor):
    """Test initialization of the extractor."""
    assert multi_tiff_extractor.get_num_frames() == 4
    assert multi_tiff_extractor.get_image_size() == (10, 12)  # (height, width)
    assert multi_tiff_extractor.get_sampling_frequency() == 30.0
    assert len(multi_tiff_extractor.get_channel_names()) == 1


def test_get_video(multi_tiff_extractor):
    """Test retrieving video from the extractor."""
    # Get specific frames
    frame_0 = multi_tiff_extractor.get_video(frame_idxs=[0])
    frame_1 = multi_tiff_extractor.get_video(frame_idxs=[1])

    # Check frame shapes - should include depth dimension
    assert frame_0.shape == (1, 10, 12, 3)
    assert frame_1.shape == (1, 10, 12, 3)

    # Get multiple frames
    frames = multi_tiff_extractor.get_video(frame_idxs=[0, 1])
    assert frames.shape == (2, 10, 12, 3)

    # Get all frames
    all_frames = multi_tiff_extractor.get_video()
    assert all_frames.shape == (4, 10, 12, 3)


def test_from_folder(temp_tiff_files):
    """Test creating an extractor from a folder path."""
    extractor = MultiTIFFMultiPageExtractor.from_folder(
        folder_path=temp_tiff_files,
        file_pattern="*.tif",
        sampling_frequency=30.0,
        dimension_order="CZT",
        num_channels=1,
        num_planes=3,
        num_acquisition_cycles=2,
    )

    assert extractor.get_num_frames() == 4
    assert extractor.get_image_size() == (10, 12)


def test_different_dimension_orders(temp_tiff_files):
    """Test different dimension orders."""
    file_paths = sorted(list(temp_tiff_files.glob("*.tif")))

    # Test with ZTC dimension order
    extractor_ztc = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order="ZTC",
        num_channels=1,
        num_planes=3,
    )

    assert extractor_ztc.get_num_frames() == 4

    # Get frames and check they're accessible
    frame_0 = extractor_ztc.get_video(frame_idxs=[0])
    assert frame_0.shape == (1, 10, 12, 3)


@pytest.mark.parametrize("dimension_order", ["CZT", "ZCT", "TCZ", "TZC"])
def test_dimension_order_variations(temp_tiff_files, dimension_order):
    """Test various dimension order configurations."""
    file_paths = sorted(list(temp_tiff_files.glob("*.tif")))

    extractor = MultiTIFFMultiPageExtractor(
        file_paths=file_paths,
        sampling_frequency=30.0,
        dimension_order=dimension_order,
        num_channels=1,
        num_planes=3,
    )

    assert extractor.get_num_frames() == 4
    video = extractor.get_video(frame_idxs=[0])
    assert video.ndim == 4  # (frames, height, width, depth)


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
    assert multi_extractor.get_num_frames() == scanimage_extractor.get_num_frames()

    # Compare image sizes - ScanImage includes depth dimension, MultiTIFF doesn't
    scanimage_size = scanimage_extractor.get_image_size()
    multi_size = multi_extractor.get_image_size()

    # Check that the first two dimensions (height, width) match
    assert multi_size[0] == scanimage_size[0]
    assert multi_size[1] == scanimage_size[1]

    assert multi_extractor.get_sampling_frequency() == scanimage_extractor.get_sampling_frequency()

    # Compare the actual video data
    scanimage_video = scanimage_extractor.get_video().squeeze()
    multi_video = multi_extractor.get_video()

    assert_allclose(scanimage_video, multi_video, rtol=1e-5, atol=1e-8)


def test_channel_extraction(tmp_path):
    """Test extraction of specific channels."""
    # Create multi-channel test data
    tifffile = get_package(package_name="tifffile")

    # Create a file with multiple channels
    num_channels = 2
    num_planes = 2
    num_timepoints = 3
    height = 8
    width = 10

    file_path = tmp_path / "multichannel.tif"
    pages = []
    for t in range(num_timepoints):
        for c in range(num_channels):
            for z in range(num_planes):
                page_data = np.ones((height, width), dtype=np.uint16) * (t * 100 + c * 10 + z + 1)
                pages.append(page_data)

    tifffile.imwrite(file_path, pages)

    # Test extracting first channel
    extractor_ch0 = MultiTIFFMultiPageExtractor(
        file_paths=[file_path],
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=num_channels,
        channel_index=0,
        num_planes=num_planes,
    )

    # Test extracting second channel
    extractor_ch1 = MultiTIFFMultiPageExtractor(
        file_paths=[file_path],
        sampling_frequency=30.0,
        dimension_order="TCZ",
        num_channels=num_channels,
        channel_index=1,
        num_planes=num_planes,
    )

    assert extractor_ch0.get_num_frames() == num_timepoints
    assert extractor_ch1.get_num_frames() == num_timepoints

    # Verify different channels have different data
    video_ch0 = extractor_ch0.get_video()
    video_ch1 = extractor_ch1.get_video()

    assert not np.array_equal(video_ch0, video_ch1)


def test_error_handling(tmp_path):
    """Test error handling for invalid inputs."""
    with pytest.raises(Exception):
        # Should raise error for non-existent files
        MultiTIFFMultiPageExtractor(
            file_paths=[Path("nonexistent.tif")],
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=1,
        )

    # Test invalid dimension order
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
