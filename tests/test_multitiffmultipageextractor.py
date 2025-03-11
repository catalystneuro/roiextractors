"""Tests for the MultiTIFFMultiPageExtractor."""

import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
import pytest
from numpy.testing import assert_array_equal

from roiextractors.extractors.tiffimagingextractors import (
    MultiTIFFMultiPageExtractor,
    ScanImageTiffSinglePlaneImagingExtractor,
)
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import extract_extra_metadata, parse_metadata
from roiextractors.extraction_tools import get_package
from tests.setup_paths import OPHYS_DATA_PATH


class TestMultiTIFFMultiPageExtractor(unittest.TestCase):
    """Tests for the MultiTIFFMultiPageExtractor."""

    def setUp(self):
        """Set up the test case."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder_path = Path(self.temp_dir.name)

        # Create test TIFF files
        self._create_test_tiff_files()

    def tearDown(self):
        """Clean up after the test case."""
        self.temp_dir.cleanup()

    def _create_test_tiff_files(self):
        """Create test TIFF files for testing."""
        tifffile = get_package(package_name="tifffile")

        # Create test data
        # 2 files, each with 6 pages (2 time points, 1 channel, 3 depths)
        num_files = 2
        time_size = 2
        channel_size = 1
        depth_size = 3
        height = 10
        width = 12

        for file_idx in range(num_files):
            file_path = self.folder_path / f"test_file_{file_idx}.tif"

            # Create pages for this file
            pages = []
            for t in range(time_size):
                for c in range(channel_size):
                    for z in range(depth_size):
                        # Create a unique pattern for each page
                        page_data = np.ones((height, width), dtype=np.uint16) * (file_idx * 100 + t * 10 + z + 1)
                        pages.append(page_data)

            # Write pages to TIFF file
            tifffile.imwrite(file_path, pages)

    def test_initialization(self):
        """Test initialization of the extractor."""
        file_paths = sorted(list(self.folder_path.glob("*.tif")))

        # Test with default parameters
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            dimension_order="XYZCT",
            channel_size=1,
            depth_size=3,
            time_size=2,
            sampling_frequency=30.0,
        )

        # Check basic properties
        self.assertEqual(extractor.get_num_frames(), 2)  # time_size
        self.assertEqual(extractor.get_image_size(), (10, 12))  # (height, width)
        self.assertEqual(extractor.get_sampling_frequency(), 30.0)
        self.assertEqual(len(extractor.get_channel_names()), 1)

    def test_get_frames(self):
        """Test retrieving frames from the extractor."""
        file_paths = sorted(list(self.folder_path.glob("*.tif")))

        # Initialize with XYZCT dimension order
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            dimension_order="XYZCT",
            channel_size=1,
            depth_size=3,
            time_size=2,
            sampling_frequency=30.0,
        )

        # Get specific frames
        frame_0 = extractor.get_frames(0)
        frame_1 = extractor.get_frames(1)

        # Check frame shapes - should include depth dimension
        self.assertEqual(frame_0.shape, (10, 12, 3))
        self.assertEqual(frame_1.shape, (10, 12, 3))

        # Get multiple frames
        frames = extractor.get_frames([0, 1])
        self.assertEqual(frames.shape, (2, 10, 12, 3))

    def test_get_video(self):
        """Test retrieving video from the extractor."""
        file_paths = sorted(list(self.folder_path.glob("*.tif")))

        # Initialize with XYZCT dimension order
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            dimension_order="XYZCT",
            channel_size=1,
            depth_size=3,
            time_size=2,
            sampling_frequency=30.0,
        )

        # Get full video
        video = extractor.get_video()
        self.assertEqual(video.shape, (2, 10, 12, 3))

        # Get partial video
        partial_video = extractor.get_video(start_frame=0, end_frame=1)
        self.assertEqual(partial_video.shape, (1, 10, 12, 3))

    def test_from_folder(self):
        """Test creating an extractor from a folder path."""
        # Create extractor using from_folder
        extractor = MultiTIFFMultiPageExtractor.from_folder(
            folder_path=self.folder_path,
            file_pattern="*.tif",
            dimension_order="XYZCT",
            channel_size=1,
            depth_size=3,
            time_size=2,
            sampling_frequency=30.0,
        )

        # Check basic properties
        self.assertEqual(extractor.get_num_frames(), 2)
        self.assertEqual(extractor.get_image_size(), (10, 12))

    def test_different_dimension_orders(self):
        """Test different dimension orders."""
        file_paths = sorted(list(self.folder_path.glob("*.tif")))

        # Test with XYZTC dimension order
        extractor_xyztc = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            dimension_order="XYZTC",
            channel_size=1,
            depth_size=3,
            time_size=2,
            sampling_frequency=30.0,
        )

        # Check basic properties
        self.assertEqual(extractor_xyztc.get_num_frames(), 2)

        # Get frames and check they're accessible
        frame_0 = extractor_xyztc.get_frames(0)
        self.assertEqual(frame_0.shape, (10, 12, 3))

    def test_comparison_with_scanimage(self):
        """Test comparison with ScanImageTiffSinglePlaneImagingExtractor."""
        # Skip test if OPHYS_DATA_PATH is not available
        if not hasattr(OPHYS_DATA_PATH, "exists") or not OPHYS_DATA_PATH.exists():
            self.skipTest("OPHYS_DATA_PATH not available")

        # Path to ScanImage test file
        file_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage" / "scanimage_20220801_single.tif"
        if not file_path.exists():
            self.skipTest(f"Test file {file_path} not found")

        # Extract metadata from ScanImage file
        metadata = extract_extra_metadata(file_path)
        parsed_metadata = parse_metadata(metadata)

        # Initialize ScanImageTiffSinglePlaneImagingExtractor
        scanimage_extractor = ScanImageTiffSinglePlaneImagingExtractor(
            file_path=file_path, channel_name="Channel 1", plane_name="0"
        )

        # Initialize MultiTIFFMultiPageExtractor with equivalent mapping
        # For this test, we use depth_size=1 to match ScanImage's single plane
        multi_extractor = MultiTIFFMultiPageExtractor(
            file_paths=[file_path],
            dimension_order="XYZCT",  # ScanImage uses this order
            channel_size=parsed_metadata["num_channels"],
            depth_size=1,  # Use depth_size=1 for single plane
            time_size=scanimage_extractor.get_num_frames(),
            sampling_frequency=parsed_metadata["sampling_frequency"],
        )

        # Compare basic properties
        self.assertEqual(multi_extractor.get_num_frames(), scanimage_extractor.get_num_frames())
        self.assertEqual(multi_extractor.get_image_size(), scanimage_extractor.get_image_size())
        self.assertEqual(multi_extractor.get_sampling_frequency(), scanimage_extractor.get_sampling_frequency())

        # Compare frames - need to reshape ScanImage frames to match MultiTIFF
        for frame_idx in range(scanimage_extractor.get_num_frames()):
            scanimage_frame = scanimage_extractor.get_frames(frame_idx)
            multi_frame = multi_extractor.get_frames(frame_idx)

            # Reshape scanimage_frame to match multi_frame if needed
            if len(scanimage_frame.shape) == 3 and scanimage_frame.shape[0] == 1:
                scanimage_frame = scanimage_frame[0]  # Remove batch dimension

            assert_array_equal(multi_frame, scanimage_frame)

        # Compare video - need to reshape ScanImage video to match MultiTIFF
        scanimage_video = scanimage_extractor.get_video()
        multi_video = multi_extractor.get_video()

        # Reshape scanimage_video to match multi_video if needed
        if len(scanimage_video.shape) == 3 and len(multi_video.shape) == 4:
            # Add depth dimension to scanimage_video
            scanimage_video = scanimage_video[:, :, :, np.newaxis]
        elif len(scanimage_video.shape) == 4 and len(multi_video.shape) == 3:
            # Remove batch dimension from scanimage_video
            multi_video = multi_video[:, :, :, 0]

        assert_array_equal(multi_video, scanimage_video)


if __name__ == "__main__":
    unittest.main()
