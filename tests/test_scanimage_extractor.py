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


class TestScanImageExtractor:
    """Test the ScanImage extractor classes with various ScanImage files."""

    def test_planar_single_channel_single_file(self):
        """Test with planar (non-volumetric) single channel data.

        File: scanimage_20220801_single.tif
        Metadata:
        - Volumetric: False (single plane)
        - Channels: 1
        - Frames: 3
        - Frame rate: 15.2379 Hz
        - Image shape: 1024 x 1024
        """
        # This is frame per slice 24 and should fail
        file_path = SCANIMAGE_PATH / "scanimage_20220801_single.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path)

        assert extractor.is_volumetric == False
        assert extractor.get_num_samples() == 3
        assert extractor.get_image_shape() == (1024, 1024)
        assert extractor.get_sampling_frequency() == 15.2379

        samples_per_file = extractor.get_num_samples() / len(extractor.file_paths)
        start_sample = 0
        end_sample = samples_per_file
        for file_path in extractor.file_paths:
            with TiffReader(file_path) as tiff_reader:
                data = tiff_reader.asarray()
                extractor_samples = extractor.get_series(start_sample=start_sample, end_sample=end_sample)
                assert_array_equal(data, extractor_samples)

                start_sample += samples_per_file
                end_sample += samples_per_file

    def test_planar_multi_channnel_multi_file(self):
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

        extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")
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
        end_sample = samples_per_file
        for file_path in extractor.file_paths:
            with TiffReader(file_path) as tiff_reader:
                data = tiff_reader.asarray()
                channel_data = data[:, extractor._channel_index, ...]
                extractor_samples = extractor.get_series(start_sample=start_sample, end_sample=end_sample)
                assert_array_equal(channel_data, extractor_samples)

                start_sample += samples_per_file
                end_sample += samples_per_file


def test_get_availale_channel_names():
    """Test the static get_channel_names method.

    This test verifies that the static get_available_channel_names method correctly extracts
    channel names from ScanImage TIFF files without needing to create an extractor instance.

    The test checks:
    1. Single-channel file: Verifies correct channel name extraction
    2. Multi-channel file: Verifies all channel names are correctly extracted
    """
    # Test with single-channel file
    single_channel_file = SCANIMAGE_PATH / "scanimage_20220801_single.tif"
    single_channel_names = ScanImageImagingExtractor.get_available_channel_names(single_channel_file)
    assert isinstance(single_channel_names, list), "Channel names should be returned as a list"
    assert len(single_channel_names) == 1, "Single channel file should have one channel"
    assert single_channel_names[0] == "Channel 1", "Channel name should match expected value"

    # Test with multi-channel file
    multi_channel_file = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
    multi_channel_names = ScanImageImagingExtractor.get_available_channel_names(multi_channel_file)
    assert isinstance(multi_channel_names, list), "Channel names should be returned as a list"
    assert len(multi_channel_names) == 2, "Multi-channel file should have two channels"
    assert multi_channel_names == ["Channel 1", "Channel 2"]

    # Test with volumetric file (should still work even though extractor initialization would fail)
    volumetric_file = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
    volumetric_channel_names = ScanImageImagingExtractor.get_available_channel_names(volumetric_file)
    assert isinstance(volumetric_channel_names, list), "Channel names should be returned as a list"
    assert len(volumetric_channel_names) >= 1, "Should extract at least one channel name"
    assert volumetric_channel_names == ["Channel 1", "Channel 4"]


def test_file_paths_parameter():
    """Test the file_paths parameter for direct file specification.

    This test verifies that the extractor correctly uses the provided file_paths
    parameter instead of relying on automatic file detection heuristics.

    The test checks:
    1. Explicitly providing file paths works correctly
    2. The extractor loads the correct number of files
    3. The files are loaded in the correct order
    """
    # Get the paths to the multi-file dataset
    file_paths = [
        SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif",
        SCANIMAGE_PATH / "scanimage_20240320_multifile_00002.tif",
        SCANIMAGE_PATH / "scanimage_20240320_multifile_00003.tif",
    ]

    # Create extractor with explicit file_paths parameter
    extractor = ScanImageImagingExtractor(file_paths=file_paths, channel_name="Channel 1")

    # Verify the correct files were loaded
    assert len(extractor.file_paths) == 3, "Should load all 3 files"

    # Verify the files are in the correct order
    expected_file_names = [Path(p).name for p in file_paths]
    actual_file_names = [Path(p).name for p in extractor.file_paths]
    assert actual_file_names == expected_file_names, "Files should be loaded in the provided order"

    # Verify the data is loaded correctly
    assert extractor.get_num_samples() == 10 * 3, "Should have 10 samples per file"
    assert extractor.get_image_shape() == (512, 512), "Image shape should be correct"


def test_get_times_planar_multichannel():
    """Test the get_times method.

    This test verifies that the get_times method correctly extracts timestamps
    from ScanImage TIFF files for the selected channel.

    The test checks that for multi-channel data, the timestamps are correctly
    filtered for the selected channel.
    """

    file_path = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"

    extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")
    extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")

    # Get timestamps for each channel
    timestamps_ch1 = extractor_ch1.get_times()
    timestamps_ch2 = extractor_ch2.get_times()

    # Check basic properties
    assert len(timestamps_ch1) == extractor_ch1.get_num_samples(), "Should have one timestamp per frame for channel 1"
    assert len(timestamps_ch2) == extractor_ch2.get_num_samples(), "Should have one timestamp per frame for channel 2"

    # Extract all timestamps from the file to compare
    from tifffile import TiffFile

    all_file_paths = [
        "scanimage_20240320_multifile_00001.tif",
        "scanimage_20240320_multifile_00002.tif",
        "scanimage_20240320_multifile_00003.tif",
    ]

    all_timestamps = []
    for file_name in all_file_paths:
        file_path = SCANIMAGE_PATH / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        # Read the metadata from the TIFF file
        with TiffFile(file_path) as tiff:
            for page in tiff.pages:
                if "ImageDescription" not in page.tags:
                    continue

                metadata_str = page.tags["ImageDescription"].value

                # Look for the timestamp line
                for line in metadata_str.strip().split("\n"):
                    if line.startswith("frameTimestamps_sec"):
                        # Extract the value part after " = "
                        _, value_str = line.split(" = ", 1)
                        try:
                            timestamp = float(value_str.strip())
                            all_timestamps.append(timestamp)
                        except ValueError:
                            all_timestamps.append(None)
                        break

    all_timestamps = np.asarray(all_timestamps)
    assert len(all_timestamps) == len(timestamps_ch1) + len(
        timestamps_ch2
    ), "Total timestamps should match the sum of both channels"
    # Verify that the timestamps for each channel are correctly filtered from all timestamps
    # For a 2-channel recording, channel 3 should have timestamps from even indices (0, 2, 4...)
    # and channel 4 should have timestamps from odd indices (1, 3, 5...)
    ch1_indices = np.arange(0, len(all_timestamps), 2)
    ch2_indices = np.arange(1, len(all_timestamps), 2)

    expected_timestamps_ch1 = np.array(all_timestamps)[ch1_indices]
    expected_timestamps_ch2 = np.array(all_timestamps)[ch2_indices]

    # Compare the first few timestamps to verify correct filtering
    assert np.allclose(timestamps_ch1, expected_timestamps_ch1), "Channel 1 timestamps should match expected values"
    assert np.allclose(timestamps_ch2, expected_timestamps_ch2), "Channel 2 timestamps should match expected values"
