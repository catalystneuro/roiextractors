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


class TestScanImageExtractorVolumetricMultiSample:
    """Test the ScanImage extractor classes with files that have multiple frames per slice."""

    def test_volumetric_data_single_channel_single_file(self):
        """Test with volumetric data that has multiple frames per slice.

        File: scanimage_20220801_volume.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (20 slices)
        - Channels: 1 (single channel)
        - Frames per slice: 8
        - Frame rate: 30.0141 Hz
        - Volume rate: 0.187588 Hz
        - Pages/IDFs: 20
        - 'num_frames_per_volume': 160

        This file does not have enough samples to do a full slice extraction. It has 20 slices
        and 8 frames per slice but only 20 pages of data. This means that only data from the first three
        depths are included. Skipping this test for now
        """

        return

    def test_volumetric_data_single_channel_single_file_2(self):
        """Test with a multi-volume ScanImage file with frames per slice > 1.

        File: scanimage_20220801_multivolume.tif
        Metadata:
        - Volumetric: True (multiple planes)
        - Channels: 1 (single channel)
        - Image shape: 512 x 512
        - Frames per slice: 8
        - num_ifds: 20
        - num_slices: 10

        This test does not have enough samples to do a full slice extraction. It has 20 slices
        and 8 frames per slice but only 20 pages of data. This means that only data from the first three
        depths are included. Skipping this test for now.
        """
        return

    def test_volumetric_data_multi_channel_single_file(self):
        """Test with volumetric data acquired in grab mode with multiple frames per depth.

        File: scanimage_20220923_noroi.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (multiple planes)
        - Channels: Multiple Channels ['Channel 1', 'Channel 4']
        - Frames per depth: 2
        - Frame rate: 29.1248 Hz
        - Volume rate: 7.28119 Hz
        - Image shape: 256 x 256
        - Num_slices: 2
        - IFDS/Pages: 24
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Test that ValueError is raised when slice_sample is not provided
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4")

        # Test that the extractor works correctly when a valid slice_sample is provided
        extractor_sample_1 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=0)

        assert extractor_sample_1.is_volumetric == True
        assert extractor_sample_1.get_frame_shape() == (256, 256)
        assert extractor_sample_1.get_num_planes() == 2
        assert extractor_sample_1.get_sample_shape() == (256, 256, 2)
        assert extractor_sample_1.get_sampling_frequency() == 14.5517
        expected_samples = 24 // 2 // 2 // 2  # 24 pages, 2 channels, 2 slices and 2 frames per slice
        assert extractor_sample_1.get_num_samples() == expected_samples

        extractor_sample_2 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=1)
        assert extractor_sample_2.is_volumetric == True
        assert extractor_sample_2.get_frame_shape() == (256, 256)
        assert extractor_sample_2.get_num_planes() == 2
        assert extractor_sample_2.get_sample_shape() == (256, 256, 2)

        assert extractor_sample_2.get_sampling_frequency() == 14.5517
        expected_samples_2 = 24 // 2 // 2 // 2  # 24 pages, 2 channels, 2 slices and 2 frames per slice
        assert extractor_sample_2.get_num_samples() == expected_samples_2

        # Compare to tiff library extraction for data integrity:
        with TiffReader(file_path) as tiff_reader:
            data_tiff = tiff_reader.asarray()

        # Extract the data for the first slice
        data_sample_1 = extractor_sample_1.get_series()
        data_sample_2 = extractor_sample_2.get_series()

        # Compare all frames for slice sample 0
        tiff_slice_sample_index = 0
        tiff_channel_index = 1  # Channel 4 is at index 1 in the tiff data

        # Iterate through all samples (volumes)
        num_samples = extractor_sample_1.get_num_samples()
        num_planes = extractor_sample_1.get_num_planes()
        for sample_index in range(num_samples):
            sample = data_sample_1[sample_index, ...]  # data is (time, width, height, depth)

            # Iterate through all frames in the sample (depth planes)
            for frame_index in range(num_planes):
                frame_extractor = sample[..., frame_index]

                # Calculate the corresponding index in the tiff data
                # Each sample has 2 frames per slice, and we're using slice_sample=0
                tiff_frame_index = sample_index * 2 + frame_index
                frame_tiff = data_tiff[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )

        # Compare all frames for slice sample 1
        tiff_slice_sample_index = 1

        # Iterate through all samples (volumes)
        num_samples = extractor_sample_2.get_num_samples()
        num_planes = extractor_sample_2.get_num_planes()
        for sample_index in range(num_samples):
            sample = data_sample_2[sample_index, ...]  # data is (time, width, height, depth)

            # Iterate through all frames in the sample (depth planes)
            for frame_index in range(num_planes):
                frame_extractor = sample[..., frame_index]

                # Calculate the corresponding index in the tiff data
                # Each sample has 2 frames per slice, and we're using slice_sample=1
                tiff_frame_index = sample_index * 2 + frame_index
                frame_tiff = data_tiff[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )

    def test_volumetric_data_multi_channel_single_file_2(self):
        """Test with multi-channel volumetric data with frames per slice > 1.

        File: scanimage_20220923_roi.tif
        Metadata:
        - Volumetric: True (multiple planes)
        - Channels: Multiple ['Channel 1', 'Channel 4'
        - Frames per slice: 2
        - Frame rate: 29.1248 Hz
        - Volume rate: 7.28119 Hz
        - IFDS/Pages: 24
        - Num slices: 2
        - Image shape: 528 x 256

        """

        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"

        # Test that ValueError is raised when slice_sample is not provided
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1")

        # Test that the extractor works correctly when a valid slice_sample is provided
        extractor_sample_1 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1", slice_sample=0)
        extractor_sample_2 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1", slice_sample=1)

        assert extractor_sample_1.is_volumetric == True
        assert extractor_sample_1.get_image_shape() == (528, 256)
        assert extractor_sample_1.get_sampling_frequency() == 7.28119
        expected_samples = 24 // 2 // 2 / 2  # 24 pages, 2 frames per slice, 2 channels and 2 volumes
        assert extractor_sample_1.get_num_samples() == expected_samples

        assert extractor_sample_2.is_volumetric == True
        assert extractor_sample_2.get_image_shape() == (528, 256)
        assert extractor_sample_2.get_sampling_frequency() == 7.28119
        expected_samples_2 = 24 // 2 // 2 / 2  # 24 pages, 2 frames per slice, 2 channels and 2 volumes
        assert extractor_sample_2.get_num_samples() == expected_samples_2

        # Compare to tiff library extraction for data integrity:
        with TiffReader(file_path) as tiff_reader:
            data = tiff_reader.asarray()

        # Extract the data for both slices
        data_sample_1 = extractor_sample_1.get_series()
        data_sample_2 = extractor_sample_2.get_series()

        # Compare all frames for slice sample 0
        tiff_slice_sample_index = 0
        tiff_channel_index = 0  # Channel 1 is at index 0 in the tiff data

        # Iterate through all samples (volumes)
        for sample_index in range(extractor_sample_1.get_num_samples()):
            sample = data_sample_1[sample_index, ...]  # data is (time, width, height, depth)

            # Iterate through all frames in the sample (depth planes)
            for frame_index in range(extractor_sample_1.get_num_planes()):
                frame_extractor = sample[..., frame_index]

                # Calculate the corresponding index in the tiff data
                # Each sample has 2 frames per slice, and we're using slice_sample=0
                tiff_frame_index = sample_index * 2 + frame_index
                frame_tiff = data[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )

        # Compare all frames for slice sample 1
        tiff_slice_sample_index = 1

        # Iterate through all samples (volumes)
        for sample_index in range(extractor_sample_2.get_num_samples()):
            sample = data_sample_2[sample_index, ...]  # data is (time, width, height, depth)

            # Iterate through all frames in the sample (depth planes)
            for frame_index in range(extractor_sample_2.get_num_planes()):
                frame_extractor = sample[..., frame_index]

                # Calculate the corresponding index in the tiff data
                # Each sample has 2 frames per slice, and we're using slice_sample=1
                tiff_frame_index = sample_index * 2 + frame_index
                frame_tiff = data[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )


def test_get_slices_per_sample():
    """Test the static get_slices_per_sample method.

    This test verifies that the static get_slices_per_sample method correctly extracts
    the number of slices per sample from ScanImage TIFF files without needing to create
    an extractor instance.
    """
    # Test with single frame per slice file
    single_frame_file = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
    frames_per_slice = ScanImageImagingExtractor.get_slices_per_sample(single_frame_file)
    assert frames_per_slice == 10, "File should have 1 frame per slice"

    # Test with multiple frames per slice file
    multi_frame_file = SCANIMAGE_PATH / "scanimage_20220801_volume.tif"
    frames_per_slice = ScanImageImagingExtractor.get_slices_per_sample(multi_frame_file)
    assert frames_per_slice == 8, "File should have 8 frames per slice"

    # Test with another multiple frames per slice file
    multi_frame_file2 = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
    frames_per_slice = ScanImageImagingExtractor.get_slices_per_sample(multi_frame_file2)
    assert frames_per_slice == 2, "File should have 2 frames per slice"


def test_get_channel_names():
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

    # Test with volumetric file (should still work even though extractor initialization would fail)
    volumetric_file = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
    volumetric_channel_names = ScanImageImagingExtractor.get_channel_names(volumetric_file)
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
