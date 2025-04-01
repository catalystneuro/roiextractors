import pytest
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal
from tifffile import TiffReader

from roiextractors import (
    ScanImageImagingExtractor,
)
from .setup_paths import OPHYS_DATA_PATH

# Define the path to the ScanImage test files
SCANIMAGE_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"


class TestScanImageExtractors:
    """Test the ScanImage extractor classes with various ScanImage files."""

    def test_volumetric_data(self):
        """Test with volumetric data that has multiple frames per slice.

        File: scanimage_20220801_volume.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (20 slices)
        - Channels: 1 (single channel)
        - Frames per slice: 8 (not supported)
        - Frame rate: 30.0141 Hz
        - Volume rate: 0.187588 Hz

        This test verifies that the extractor correctly raises a ValueError when
        encountering a ScanImage file with multiple frames per slice, which is not
        currently supported by the ScanImageImagingExtractor.
        """
        # For multifile, we only need to provide the first file
        file_path = SCANIMAGE_PATH / "scanimage_20220801_volume.tif"

        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_path=file_path)

        # Uncomment when adding support
        # Basic properties
        # assert extractor.is_volumetric == True
        # assert extractor.get_num_samples() == 160
        # assert extractor.get_image_shape() == (512, 512)
        # assert extractor.get_sampling_frequency()  == 0.187588

        # # Check if multiple files were detected
        # assert len(extractor.file_paths)  == 1

    def test_scanimage_multifile(self):
        """Test with multi-file planar data acquired in loop acquisition mode.

        Files: scanimage_20240320_multifile_0000[1-3].tif
        Metadata:
        - Acquisition mode: loop (3 acquisitions per loop)
        - Volumetric: False (1 slice)
        - Channels: 2 (using "Channel 3")
        - Frames per slice: 10
        - Frame rate: 98.84 Hz
        - Image shape: 512 x 512

        This test verifies that the extractor correctly:
        1. Detects and loads all 3 files in the series
        2. Extracts the correct metadata (volumetric status, sampling frequency, etc.)
        3. Retrieves the correct data from each file in the series

        The test checks:
        - File detection: Confirms all 3 files in the series are found
        - File ordering: Verifies files are loaded in the correct sequence
        - Metadata extraction: Validates volumetric status, frame count, dimensions, and sampling rate
        - Data integrity: Ensures data from each file is correctly loaded and matches the original files
        """

        # First file of the series should be enough to initialize
        file_path = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 3")
        assert len(extractor.file_paths) == 3

        expected_files_names = [
            "scanimage_20240320_multifile_00001.tif",
            "scanimage_20240320_multifile_00002.tif",
            "scanimage_20240320_multifile_00003.tif",
        ]
        extracted_file_names = [Path(file_path).name for file_path in extractor.file_paths]

        assert extracted_file_names == expected_files_names

        assert extractor.is_volumetric == False
        assert extractor.get_num_samples() == 10 * 3  # 10 samples per file
        assert extractor.get_image_shape() == (512, 512)
        assert extractor.get_sampling_frequency() == 98.83842317607626

        samples_per_file = extractor.get_num_samples() / len(extractor.file_paths)
        start_sample = 0
        stop_sample = samples_per_file
        for file_path in extractor.file_paths:
            with TiffReader(file_path) as tiff_reader:
                data = tiff_reader.asarray()
                channel_data = data[:, extractor._channel_index, ...]
                extractor_samples = extractor.get_series(start_sample=start_sample, stop_sample=stop_sample)
                assert_array_equal(channel_data, extractor_samples)

                start_sample += samples_per_file
                stop_sample += samples_per_file

    def test_scanimage_noroi(self):
        """Test with volumetric data acquired in grab mode with multiple frames per depth.

        File: scanimage_20220923_noroi.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (multiple planes)
        - Channels: Multiple (testing with "Channel 3")
        - Frames per slice: 2
        - Multiple frames per depth: Yes (not currently supported)
        - Frame rate: 29.1248 Hz
        - Volume rate: 7.28119 Hz
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 3")

        # assert extractor.is_volumetric == True
        # assert extractor.get_num_samples() == 12
        # assert extractor.get_image_shape() == (256, 256)
        # assert extractor.get_sampling_frequency()  == 14.5517

    def test_scanimage_roi(self):
        """Test with multi-channel volumetric data with frames per slice > 1.

        File: scanimage_20220923_roi.tif
        Metadata:
        - Volumetric: True (multiple planes)
        - Channels: Multiple
        - Frames per slice: 2
        - Frame rate: 29.1248 Hz
        - Volume rate: 7.28119 Hz

        This test verifies that the extractor correctly raises a ValueError when
        encountering a ScanImage file with frames per slice > 1, which is not
        currently supported by the ScanImageImagingExtractor.

        When support for multiple frames per slice is added, this test should be
        updated to verify correct metadata extraction and data loading.
        """

        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 3")

    def test_scanimage_single(self):
        """Test with planar (non-volumetric) single channel data.

        File: scanimage_20220801_single.tif
        Metadata:
        - Volumetric: False (single plane)
        - Channels: 1 (using "Channel 3")
        - Frames: 3
        - Frame rate: 15.2379 Hz
        - Image shape: 1024 x 1024

        This test verifies that the extractor correctly:
        1. Identifies the data as non-volumetric (planar)
        2. Extracts the correct metadata (frame count, dimensions, sampling rate)
        3. Loads the correct number of samples
        4. Retrieves data that matches the original file content

        Unlike the volumetric tests, this test should pass as single-plane data
        is fully supported by the ScanImageImagingExtractor.
        """
        # This is frame per slice 24 and should fail
        file_path = SCANIMAGE_PATH / "scanimage_20220801_single.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 3")

        assert extractor.is_volumetric == False
        assert extractor.get_num_samples() == 3
        assert extractor.get_image_shape() == (1024, 1024)
        assert extractor.get_sampling_frequency() == 15.2379

        samples_per_file = extractor.get_num_samples() / len(extractor.file_paths)
        start_sample = 0
        stop_sample = samples_per_file
        for file_path in extractor.file_paths:
            with TiffReader(file_path) as tiff_reader:
                data = tiff_reader.asarray()
                extractor_samples = extractor.get_series(start_sample=start_sample, stop_sample=stop_sample)
                assert_array_equal(data, extractor_samples)

                start_sample += samples_per_file
                stop_sample += samples_per_file

    def test_scanimage_multivolume(self):
        """Test with a multi-volume ScanImage file with frames per slice > 1.

        File: scanimage_20220801_multivolume.tif
        Metadata:
        - Volumetric: True (multiple planes)
        - Channels: 1 (single channel)
        - Frames per slice: 8
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220801_multivolume.tif"

        # Create multi-plane extractor
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_path=file_path)

    def test_static_get_channel_names(self):
        """Test the static get_channel_names method.

        This test verifies that the static get_channel_names method correctly extracts
        channel names from ScanImage TIFF files without needing to create an extractor instance.

        The test checks:
        1. Single-channel file: Verifies correct channel name extraction
        2. Multi-channel file: Verifies all channel names are correctly extracted
        """
        # Test with single-channel file
        single_channel_file = SCANIMAGE_PATH / "scanimage_20220801_single.tif"
        single_channel_names = ScanImageImagingExtractor.get_channel_names(single_channel_file)
        assert isinstance(single_channel_names, list), "Channel names should be returned as a list"
        assert len(single_channel_names) == 1, "Single channel file should have one channel"
        assert single_channel_names[0] == "Channel 1", "Channel name should match expected value"

        # Test with multi-channel file
        multi_channel_file = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
        multi_channel_names = ScanImageImagingExtractor.get_channel_names(multi_channel_file)
        assert isinstance(multi_channel_names, list), "Channel names should be returned as a list"
        assert len(multi_channel_names) == 2, "Multi-channel file should have two channels"
        assert multi_channel_names == ["Channel 1", "Channel 2"]

        # Test with volumetric fil
        volumetric_file = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
        volumetric_channel_names = ScanImageImagingExtractor.get_channel_names(volumetric_file)
        assert isinstance(volumetric_channel_names, list), "Channel names should be returned as a list"
        assert len(volumetric_channel_names) >= 1, "Should extract at least one channel name"
        assert volumetric_channel_names == ["Channel 1", "Channel 4"]
