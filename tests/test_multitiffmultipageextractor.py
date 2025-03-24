"""Tests for the MultiTIFFMultiPageExtractor."""

import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from roiextractors.extractors.tiffimagingextractors import (
    MultiTIFFMultiPageExtractor,
    ScanImageTiffMultiPlaneMultiFileImagingExtractor,
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
        num_acquisition_cycles = 2
        num_channels = 1
        num_planes = 3
        height = 10
        width = 12

        for file_idx in range(num_files):
            file_path = self.folder_path / f"test_file_{file_idx}.tif"

            # Create pages for this file
            pages = []
            for t in range(num_acquisition_cycles):
                for c in range(num_channels):
                    for z in range(num_planes):
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
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=3,
            num_acquisition_cycles=2,
        )

        # Check basic properties
        self.assertEqual(extractor.get_num_frames(), 2)  # num_acquisition_cycles
        self.assertEqual(extractor.get_image_size(), (10, 12))  # (height, width)
        self.assertEqual(extractor.get_sampling_frequency(), 30.0)
        self.assertEqual(len(extractor.get_channel_names()), 1)

    def test_get_frames(self):
        """Test retrieving frames from the extractor."""
        file_paths = sorted(list(self.folder_path.glob("*.tif")))

        # Initialize with CZT dimension order
        extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=3,
            num_acquisition_cycles=2,
        )

        # Get specific frames
        frame_0 = extractor.get_frames([0])
        frame_1 = extractor.get_frames([1])

        # Check frame shapes - should include depth dimension
        self.assertEqual(frame_0.shape, (1, 10, 12, 3))
        self.assertEqual(frame_1.shape, (1, 10, 12, 3))

        # Get multiple frames
        frames = extractor.get_frames([0, 1])
        self.assertEqual(frames.shape, (2, 10, 12, 3))

    def test_from_folder(self):
        """Test creating an extractor from a folder path."""
        # Create extractor using from_folder
        extractor = MultiTIFFMultiPageExtractor.from_folder(
            folder_path=self.folder_path,
            file_pattern="*.tif",
            sampling_frequency=30.0,
            dimension_order="CZT",
            num_channels=1,
            num_planes=3,
            num_acquisition_cycles=2,
        )

        # Check basic properties
        self.assertEqual(extractor.get_num_frames(), 2)
        self.assertEqual(extractor.get_image_size(), (10, 12))

    def test_different_dimension_orders(self):
        """Test different dimension orders."""
        file_paths = sorted(list(self.folder_path.glob("*.tif")))

        # Test with ZTC dimension order
        extractor_ztc = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=30.0,
            dimension_order="ZTC",
            num_channels=1,
            num_planes=3,
            num_acquisition_cycles=2,
        )

        # Check basic properties
        self.assertEqual(extractor_ztc.get_num_frames(), 2)

        # Get frames and check they're accessible
        frame_0 = extractor_ztc.get_frames([0])
        self.assertEqual(frame_0.shape, (1, 10, 12, 3))

    def test_comparison_with_scanimage_multifile(self):
        """Test comparison with ScanImageTiffMultiPlaneMultiFileImagingExtractor."""
        # Skip test if OPHYS_DATA_PATH is not available
        if not hasattr(OPHYS_DATA_PATH, "exists") or not OPHYS_DATA_PATH.exists():
            self.skipTest("OPHYS_DATA_PATH not available")

        # Path to ScanImage folder and file pattern
        scanimage_folder_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"
        multifile_file_pattern = "scanimage_20240320_multifile_*.tif"

        # Check if the files exist
        file_paths = list(scanimage_folder_path.glob(multifile_file_pattern))
        if not file_paths:
            self.skipTest(f"No files found matching pattern {multifile_file_pattern} in folder {scanimage_folder_path}")

        # Initialize ScanImageTiffMultiPlaneMultiFileImagingExtractor
        try:
            scanimage_extractor = ScanImageTiffMultiPlaneMultiFileImagingExtractor(
                folder_path=scanimage_folder_path,
                file_pattern=multifile_file_pattern,
                channel_name="Channel 1",
            )
        except Exception as e:
            self.skipTest(f"Failed to initialize ScanImageTiffMultiPlaneMultiFileImagingExtractor: {e}")

        # Initialize MultiTIFFMultiPageExtractor with equivalent mapping
        multi_extractor = MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=scanimage_extractor.get_sampling_frequency(),
            dimension_order="CZT",  # ScanImage uses this order
            num_channels=2,
            channel_index=0,
            num_planes=1,  # For this example, num_planes is 1
            num_acquisition_cycles=scanimage_extractor.get_num_frames(),
        )

        # Compare basic properties
        self.assertEqual(multi_extractor.get_num_frames(), scanimage_extractor.get_num_frames())

        # Compare image sizes - ScanImage includes depth dimension, MultiTIFF doesn't
        scanimage_size = scanimage_extractor.get_image_size()
        multi_size = multi_extractor.get_image_size()

        # Check that the first two dimensions (height, width) match
        self.assertEqual(multi_size[0], scanimage_size[0])
        self.assertEqual(multi_size[1], scanimage_size[1])

        self.assertEqual(multi_extractor.get_sampling_frequency(), scanimage_extractor.get_sampling_frequency())

        # For this test, we'll just verify that both extractors can retrieve frames
        # without comparing the actual data, as the data organization might differ
        # between the two extractors
        scanimage_frame = scanimage_extractor.get_video().squeeze()
        multi_frame = multi_extractor.get_video()  # Updated to use get_video()

        assert_allclose(scanimage_frame, multi_frame, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
