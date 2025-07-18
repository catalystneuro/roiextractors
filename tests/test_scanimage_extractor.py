import shutil
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tifffile import TiffReader

from roiextractors import (
    ScanImageImagingExtractor,
)

from .setup_paths import OPHYS_DATA_PATH

# Define the path to the ScanImage test files
SCANIMAGE_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"


def test_old_scan_image_version():
    """Test that an error informative error is raised when using an old ScanImage version."""
    file_path = SCANIMAGE_PATH / "sample_scanimage_version_3_8.tiff"

    with pytest.raises(ValueError):
        ScanImageImagingExtractor(file_path=file_path)


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

        extractor = ScanImageImagingExtractor(file_path=str(file_path))

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

    def test_planar_two_channels_single_file(self):
        """Test with planar (non-volumetric) two-channel data in a single file.

        File: planar_two_ch_single_files_00001_00001.tif
        Metadata:
        - Acquisition mode: grab
        - 1000 samples (frames)
        - Volumetric: False but `SI.hStackManager.enable` is set to `True`
        - Frame shape: (20, 20)
        - Channels: 2 (`Channel 1` and `Channel 2`)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 523.926 Hz
        - Pages/IDFs: 2000

        This test verifies that the extractor correctly:
        1. Identifies the data as non-volumetric despite `SI.hStackManager.enable` being True
        2. Extracts the correct metadata (sampling frequency, frame count, dimensions)
        3. Retrieves the correct data for both channels
        """
        file_path = SCANIMAGE_PATH / "planar_two_channels_single_file" / "planar_two_ch_single_files_00001_00001.tif"

        # Test with Channel 1
        extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")

        assert extractor_ch1.is_volumetric == False
        assert extractor_ch1.get_sampling_frequency() == 523.926
        assert extractor_ch1.get_image_shape() == (20, 20)
        assert extractor_ch1.get_num_samples() == 1000

        # Test with Channel 2
        extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")

        assert extractor_ch2.is_volumetric == False
        assert extractor_ch2.get_sampling_frequency() == 523.926
        assert extractor_ch2.get_image_shape() == (20, 20)
        assert extractor_ch2.get_num_samples() == 1000

        # Get data from both extractors
        extractor_data_ch1 = extractor_ch1.get_series()
        extractor_data_ch2 = extractor_ch2.get_series()

        # Verify that the extractor data has the correct shape
        # For planar data, shape should be (samples, height, width)
        expected_shape = (extractor_ch1.get_num_samples(), *extractor_ch1.get_image_shape())
        assert (
            extractor_data_ch1.shape == expected_shape
        ), f"Channel 1: Expected shape {expected_shape}, got {extractor_data_ch1.shape}"
        assert (
            extractor_data_ch2.shape == expected_shape
        ), f"Channel 2: Expected shape {expected_shape}, got {extractor_data_ch2.shape}"

        # Read tiff data for comparison
        with TiffReader(file_path) as tiff_reader:
            tiff_data = tiff_reader.asarray()

            # For multi-channel data, tiff_data shape is (frames, channels, height, width)
            # Verify the shape of the tiff data
            assert len(tiff_data.shape) == 4, "Multi-channel tiff data should have 4 dimensions"
            assert tiff_data.shape[1] == 2, "Tiff data should have 2 channels"

            # Compare a subset of the data for each channel
            # We'll check the first 10 frames
            num_frames_to_check = 10

            # Get the tiff data for each channel
            tiff_ch1 = tiff_data[:num_frames_to_check, 0, :, :]  # Channel 1 (index 0)
            tiff_ch2 = tiff_data[:num_frames_to_check, 1, :, :]  # Channel 2 (index 1)

            # Get the extractor data for the same frames
            extractor_ch1_subset = extractor_data_ch1[:num_frames_to_check]
            extractor_ch2_subset = extractor_data_ch2[:num_frames_to_check]

            # Compare the data for Channel 1
            np.testing.assert_array_equal(
                extractor_ch1_subset,
                tiff_ch1,
                f"Channel 1 data does not match tiff data",
            )

            # Compare the data for Channel 2
            np.testing.assert_array_equal(
                extractor_ch2_subset,
                tiff_ch2,
                f"Channel 2 data does not match tiff data",
            )

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

    def test_volumetric_single_channel_single_file(self):
        """Test with volumetric data in a single file.

        File: vol_no_flyback_00001_00001.tif
        Metadata:
        - Acquisition state: grab
        - Volumetric: True (9 slices per volume)
        - Frame shape: 20 x 20
        - Channels saved: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 79.8033 Hz
        - Volume rate: 8.86703 Hz
        - Number of volumes: 10
        - Pages/IDFs: 90
        """
        file_path = (
            SCANIMAGE_PATH / "volumetric_single_channel_single_file_no_flyback" / "vol_no_flyback_00001_00001.tif"
        )

        extractor = ScanImageImagingExtractor(file_path=file_path)

        assert extractor.is_volumetric == True
        assert extractor.get_num_samples() == 10
        assert extractor.get_image_shape() == (20, 20)
        assert extractor.get_sampling_frequency() == 8.86703
        assert extractor.get_num_planes() == 9

        # Get data from extractor
        extractor_data = extractor.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (extractor.get_num_samples(), *extractor.get_image_shape(), extractor.get_num_planes())
        assert extractor_data.shape == expected_shape, f"Expected shape {expected_shape}, got {extractor_data.shape}"

        # Read tiff data for comparison
        with TiffReader(file_path) as tiff_reader:
            tiff_data = tiff_reader.asarray()

            # For each sample volume in the extractor data
            for sample_index in range(extractor.get_num_samples()):
                # Calculate the slice of frames for this sample
                start_frame = sample_index * extractor.get_num_planes()
                end_frame = start_frame + extractor.get_num_planes()
                tiff_volume_frames = tiff_data[start_frame:end_frame]

                # Reshape the extractor data to match the tiff data format for comparison
                # extractor_data is (samples, height, width, planes)
                # We need to compare with tiff_volume_frames which is (planes, height, width)
                extractor_volume = np.moveaxis(extractor_data[sample_index], -1, 0)

                # Compare the data
                np.testing.assert_array_equal(
                    extractor_volume,
                    tiff_volume_frames,
                    f"Sample {sample_index} volume does not match tiff data",
                )


class TestScanImageVolumetricWithFlybackFrames:
    """Test the ScanImage extractor classes with files that have flyback frames."""

    def test_volumetric_single_channel_single_file(self):
        """
        First File: vol_one_ch_single_files_00002_00001.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 1600
        """
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path)

        assert extractor.is_volumetric == True
        assert extractor.num_flyback_frames_per_channel == 7
        assert extractor.get_sampling_frequency() == 32.7454
        assert extractor.get_image_shape() == (20, 20)
        assert extractor.get_num_planes() == 9

        ifds = 1600
        total_frames_per_cycle = extractor.get_num_planes() + extractor.num_flyback_frames_per_channel
        num_acquisition_cycles = ifds // total_frames_per_cycle
        assert extractor.get_num_samples() == num_acquisition_cycles

        # Get data from extractor
        extractor_data = extractor.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (num_acquisition_cycles, *extractor.get_image_shape(), extractor.get_num_planes())
        assert extractor_data.shape == expected_shape, f"Expected shape {expected_shape}, got {extractor_data.shape}"

        # Read tiff data for comparison
        with TiffReader(file_path) as tiff_reader:
            tiff_data = tiff_reader.asarray()

            # For each sample volume in the extractor data
            for sample_index in range(num_acquisition_cycles):
                # Calculate the slice of frames for this sample (excluding flyback frames)
                start_frame = sample_index * total_frames_per_cycle
                end_frame = start_frame + extractor.get_num_planes()
                tiff_volume_frames = tiff_data[start_frame:end_frame]

                # Reshape the extractor data to match the tiff data format for comparison
                # extractor_data is (samples, height, width, planes)
                # We need to compare with tiff_volume_frames which is (planes, height, width)
                extractor_volume = np.moveaxis(extractor_data[sample_index], -1, 0)

                # Compare the data
                np.testing.assert_array_equal(
                    extractor_volume, tiff_volume_frames, f"Sample {sample_index} volume does not match tiff data"
                )

    def test_volumetric_single_channel_multi_file(self):
        """
        First File: vol_one_ch_single_files_00002_00001.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 160 in each of the 10 files.
        """
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_multi_file" / "vol_one_ch_multi_files_00001_00001.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path)

        assert extractor.is_volumetric == True
        assert extractor.num_flyback_frames_per_channel == 7
        assert extractor.get_sampling_frequency() == 32.7454
        assert extractor.get_image_shape() == (20, 20)
        assert extractor.get_num_planes() == 9

        ifds = 160 * 10
        total_frames_per_cycle = extractor.get_num_planes() + extractor.num_flyback_frames_per_channel
        num_acquisition_cycles = ifds // total_frames_per_cycle
        assert extractor.get_num_samples() == num_acquisition_cycles

        # Get the entire series data from the extractor
        extractor_data = extractor.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (extractor.get_num_samples(), *extractor.get_image_shape(), extractor.get_num_planes())
        assert extractor_data.shape == expected_shape, f"Expected shape {expected_shape}, got {extractor_data.shape}"

        # For each sample volume in the extractor data
        for sample_index in range(extractor.get_num_samples()):
            # Determine which file contains this sample
            frames_per_file = 160
            frames_per_sample = total_frames_per_cycle
            samples_per_file = frames_per_file // frames_per_sample
            file_index = sample_index // samples_per_file

            # Get the corresponding file path
            current_file_path = extractor.file_paths[file_index]

            # Calculate the relative sample index within this file
            relative_sample_index = sample_index % samples_per_file

            # Read tiff data for comparison
            with TiffReader(current_file_path) as tiff_reader:
                tiff_data = tiff_reader.asarray()

                # Calculate the slice of frames for this sample (excluding flyback frames)
                start_frame = relative_sample_index * total_frames_per_cycle
                end_frame = start_frame + extractor.get_num_planes()
                tiff_volume_frames = tiff_data[start_frame:end_frame]

                # Reshape the extractor data to match the tiff data format for comparison
                # extractor_data is (samples, height, width, planes)
                # We need to compare with tiff_volume_frames which is (planes, height, width)
                extractor_volume = np.moveaxis(extractor_data[sample_index], -1, 0)

                # Compare the data
                np.testing.assert_array_equal(
                    extractor_volume,
                    tiff_volume_frames,
                    f"File {file_index}, sample {sample_index} volume does not match tiff data",
                )

    def test_volumetric_two_channels_single_file(self):
        """
        file_name: `vol_two_ch_single_file_00001_00001.tif`
        Metadata:
        - Acquisition mode: grab
        - 100 samples (volumes)
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 2 (`Channel 1` and `Channel 2`)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 3200
        """
        file_path = SCANIMAGE_PATH / "volumetric_two_channels_single_file" / "vol_two_ch_single_file_00001_00001.tif"

        # Test with Channel 1
        extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")

        assert extractor_ch1.is_volumetric == True
        assert extractor_ch1.num_flyback_frames_per_channel == 7
        assert extractor_ch1.get_sampling_frequency() == 32.7454
        assert extractor_ch1.get_image_shape() == (20, 20)
        assert extractor_ch1.get_num_planes() == 9

        ifds = 3200
        num_channels = 2
        total_frames_per_cycle = (
            extractor_ch1.get_num_planes() + extractor_ch1.num_flyback_frames_per_channel
        ) * num_channels
        num_acquisition_cycles = ifds // total_frames_per_cycle
        assert extractor_ch1.get_num_samples() == num_acquisition_cycles

        # Test with Channel 2
        extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")

        assert extractor_ch2.is_volumetric == True
        assert extractor_ch2.num_flyback_frames_per_channel == 7
        assert extractor_ch2.get_sampling_frequency() == 32.7454
        assert extractor_ch2.get_image_shape() == (20, 20)
        assert extractor_ch2.get_num_planes() == 9
        assert extractor_ch2.get_num_samples() == num_acquisition_cycles

        # Get data from both extractors
        extractor_data_ch1 = extractor_ch1.get_series()
        extractor_data_ch2 = extractor_ch2.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (
            extractor_ch1.get_num_samples(),
            *extractor_ch1.get_image_shape(),
            extractor_ch1.get_num_planes(),
        )
        assert (
            extractor_data_ch1.shape == expected_shape
        ), f"Channel 1: Expected shape {expected_shape}, got {extractor_data_ch1.shape}"
        assert (
            extractor_data_ch2.shape == expected_shape
        ), f"Channel 2: Expected shape {expected_shape}, got {extractor_data_ch2.shape}"

        # Read tiff data for comparison
        with TiffReader(file_path) as tiff_reader:
            tiff_data = tiff_reader.asarray()

            # For multi-channel data, tiff_data shape is (frames, channels, height, width)
            # Verify the shape of the tiff data
            assert len(tiff_data.shape) == 4, "Multi-channel tiff data should have 4 dimensions"
            assert tiff_data.shape[1] == 2, "Tiff data should have 2 channels"

            # For each sample volume in the extractor data
            for sample_index in range(num_acquisition_cycles):
                # Calculate the slice of frames for this sample (excluding flyback frames)
                # For multi-channel data, we need to select the appropriate frames
                start_frame = sample_index * total_frames_per_cycle // num_channels
                end_frame = start_frame + extractor_ch1.get_num_planes()

                # Get the tiff data for each channel
                tiff_volume_ch1 = tiff_data[start_frame:end_frame, 0, :, :]  # Channel 1 (index 0)
                tiff_volume_ch2 = tiff_data[start_frame:end_frame, 1, :, :]  # Channel 2 (index 1)

                # Reshape the extractor data to match the tiff data format for comparison
                # extractor_data is (samples, height, width, planes)
                # We need to compare with tiff_volume which is (planes, height, width)
                extractor_volume_ch1 = np.moveaxis(extractor_data_ch1[sample_index], -1, 0)
                extractor_volume_ch2 = np.moveaxis(extractor_data_ch2[sample_index], -1, 0)

                # Compare the data for Channel 1
                np.testing.assert_array_equal(
                    extractor_volume_ch1,
                    tiff_volume_ch1,
                    f"Sample {sample_index}, Channel 1 volume does not match tiff data",
                )

                # Compare the data for Channel 2
                np.testing.assert_array_equal(
                    extractor_volume_ch2,
                    tiff_volume_ch2,
                    f"Sample {sample_index}, Channel 2 volume does not match tiff data",
                )

    def test_volumetric_two_channels_multi_file(self):
        """
        First File: vol_two_ch_multi_files_00001_00001.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 2 (`Channel 1` and `Channel 2`)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 320 in each of the 10 files.
        """
        file_path = SCANIMAGE_PATH / "volumetric_two_channels_multi_file" / "vol_two_ch_multi_files_00001_00001.tif"

        # Test with Channel 1
        extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")

        assert extractor_ch1.is_volumetric == True
        assert extractor_ch1.num_flyback_frames_per_channel == 7
        assert extractor_ch1.get_sampling_frequency() == 32.7454
        assert extractor_ch1.get_image_shape() == (20, 20)
        assert extractor_ch1.get_num_planes() == 9

        ifds = 320 * 10
        num_channels = 2
        total_frames_per_cycle = (
            extractor_ch1.get_num_planes() + extractor_ch1.num_flyback_frames_per_channel
        ) * num_channels
        num_acquisition_cycles = ifds // total_frames_per_cycle
        assert extractor_ch1.get_num_samples() == num_acquisition_cycles

        # Test with Channel 2
        extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")

        assert extractor_ch2.is_volumetric == True
        assert extractor_ch2.num_flyback_frames_per_channel == 7
        assert extractor_ch2.get_sampling_frequency() == 32.7454
        assert extractor_ch2.get_image_shape() == (20, 20)
        assert extractor_ch2.get_num_planes() == 9
        assert extractor_ch2.get_num_samples() == num_acquisition_cycles

        # Get the entire series data from both extractors
        extractor_data_ch1 = extractor_ch1.get_series()
        extractor_data_ch2 = extractor_ch2.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (
            extractor_ch1.get_num_samples(),
            *extractor_ch1.get_image_shape(),
            extractor_ch1.get_num_planes(),
        )
        assert (
            extractor_data_ch1.shape == expected_shape
        ), f"Channel 1: Expected shape {expected_shape}, got {extractor_data_ch1.shape}"
        assert (
            extractor_data_ch2.shape == expected_shape
        ), f"Channel 2: Expected shape {expected_shape}, got {extractor_data_ch2.shape}"

        ifds_per_file = 320  # Number of frames in each file
        samples_per_file = ifds_per_file // total_frames_per_cycle

        # Test only the samples that fit completely within each file
        for file_index, file_path in enumerate(extractor_ch1.file_paths):
            with TiffReader(file_path) as tiff_reader:
                tiff_data = tiff_reader.asarray()

                # For multi-channel data, tiff_data shape is (samples, num_channels, x, y)
                # Verify the shape of the tiff data
                assert len(tiff_data.shape) == 4, "Multi-channel tiff data should have 4 dimensions"
                assert tiff_data.shape[1] == 2, "Tiff data should have 2 channels"

                # Calculate the sample range for this file
                start_sample = file_index * samples_per_file
                end_sample = (file_index + 1) * samples_per_file

                # Only test samples that are within the range of the extractor
                end_sample = min(end_sample, extractor_ch1.get_num_samples())

                # For each sample in this file
                for sample_index in range(start_sample, end_sample):
                    # Calculate the relative sample index within this file
                    relative_sample_index = sample_index - start_sample

                    # Calculate the slice of frames for this sample (excluding flyback frames)
                    # For multi-channel data, we need to select the appropriate frames
                    # Each volume has planes * channels frames, with flyback frames at the end
                    start_frame = relative_sample_index * total_frames_per_cycle // num_channels

                    # Make sure we don't go out of bounds
                    if start_frame + extractor_ch1.get_num_planes() > tiff_data.shape[0]:
                        continue

                    # Get the tiff data for each channel
                    tiff_volume_ch1 = tiff_data[
                        start_frame : start_frame + extractor_ch1.get_num_planes(), 0, :, :
                    ]  # Channel 1 (index 0)
                    tiff_volume_ch2 = tiff_data[
                        start_frame : start_frame + extractor_ch1.get_num_planes(), 1, :, :
                    ]  # Channel 2 (index 1)

                    # Reshape the extractor data to match the tiff data format for comparison
                    # extractor_data is (samples, height, width, planes)
                    # We need to compare with tiff_volume which is (planes, height, width)
                    extractor_volume_ch1 = np.moveaxis(extractor_data_ch1[sample_index], -1, 0)
                    extractor_volume_ch2 = np.moveaxis(extractor_data_ch2[sample_index], -1, 0)

                    # Compare the data for Channel 1
                    np.testing.assert_array_equal(
                        extractor_volume_ch1,
                        tiff_volume_ch1,
                        f"File {file_index}, Channel 1, sample {sample_index} volume does not match tiff data",
                    )

                    # Compare the data for Channel 2
                    np.testing.assert_array_equal(
                        extractor_volume_ch2,
                        tiff_volume_ch2,
                        f"File {file_index}, Channel 2, sample {sample_index} volume does not match tiff data",
                    )


class TestScanImageExtractorVolumetricMultiSamplesPerDepth:
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
        - num_frames_per_volume: 160

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

        frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(file_path)
        assert frames_per_slice == 2, "File should have 2 slices per sample"

        # Test that an error is raised when neither slice_sample nor interleave_slice_samples is provided
        with pytest.raises(ValueError):
            ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4")

        # Test that the extractor works correctly when a valid slice_sample is provided
        extractor_sample_1 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=0)

        assert extractor_sample_1.is_volumetric == True
        assert extractor_sample_1.get_frame_shape() == (256, 256)
        assert extractor_sample_1.get_num_planes() == 2
        assert extractor_sample_1.get_sample_shape() == (256, 256, 2)
        assert extractor_sample_1.get_sampling_frequency() == 14.5517

        frames_in_dataset = 24
        num_channels = len(extractor_sample_1.get_channel_names())
        num_planes = extractor_sample_1.get_num_planes()
        frames_in_a_sample = num_channels * num_planes * frames_per_slice
        expected_samples = frames_in_dataset // frames_in_a_sample
        assert extractor_sample_1.get_num_samples() == expected_samples

        extractor_sample_2 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=1)
        assert extractor_sample_2.is_volumetric == True
        assert extractor_sample_2.get_frame_shape() == (256, 256)
        assert extractor_sample_2.get_num_planes() == 2
        assert extractor_sample_2.get_sample_shape() == (256, 256, 2)

        assert extractor_sample_2.get_sampling_frequency() == 14.5517

        frames_in_dataset = 24
        num_channels = 2  # ['Channel 1', 'Channel 4']
        num_planes = extractor_sample_2.get_num_planes()

        frames_in_a_sample = num_channels * num_planes * frames_per_slice
        expected_samples_2 = frames_in_dataset // frames_in_a_sample
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

        frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(file_path)
        assert frames_per_slice == 2, "File should have 2 slices per sample"

        # Test that an error is raised when neither slice_sample nor interleave_slice_samples is provided
        with pytest.raises(ValueError):
            ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1")

        # Test that the extractor works correctly when a valid slice_sample is provided
        extractor_sample_1 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1", slice_sample=0)
        extractor_sample_2 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1", slice_sample=1)

        assert extractor_sample_1.is_volumetric == True
        assert extractor_sample_1.get_image_shape() == (528, 256)
        assert extractor_sample_1.get_sampling_frequency() == 7.28119

        frames_in_dataset = 24
        num_channels = 2  # ['Channel 1', 'Channel 4']
        num_planes = extractor_sample_1.get_num_planes()

        frames_in_a_sample = num_channels * num_planes * frames_per_slice
        expected_samples = frames_in_dataset // frames_in_a_sample
        assert extractor_sample_1.get_num_samples() == expected_samples

        assert extractor_sample_2.is_volumetric == True
        assert extractor_sample_2.get_image_shape() == (528, 256)
        assert extractor_sample_2.get_sampling_frequency() == 7.28119
        expected_samples_2 = expected_samples
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
                tiff_frame_index = sample_index * frames_per_slice + frame_index
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
                tiff_frame_index = sample_index * frames_per_slice + frame_index
                frame_tiff = data[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )


class TestScanImageExtractorVMultiSamplesPerDepthAsOneVolumetricSeries:
    """Test the ScanImage extractor's ability to interleave multiple slice samples into a single volumetric series.

    When interleave_slice_samples=True and frames_per_slice > 1, the extractor will interleave samples
    from different slice_samples to create a continuous volumetric series. This is useful when you want
    to treat each slice_sample as a separate time point rather than selecting a specific slice_sample.
    """

    def test_volumetric_data_multi_channel_single_file(self):
        """Test interleaving of slice samples with volumetric data.

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

        This test verifies that:
        1. When slice_sample is None, all slice samples are interleaved in the output
        2. The interleaved data matches the individual slice_sample data in the correct order
        3. The number of samples is multiplied by frames_per_slice
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Create extractors with and without slice_sample
        extractor_full = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 1",
            interleave_slice_samples=True,
        )
        extractor_slice_sample0 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 1",
            slice_sample=0,
        )
        extractor_slice_sample1 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 1",
            slice_sample=1,
        )

        # Get data from all extractors
        full_series = extractor_full.get_series()
        series_slice_sample0 = extractor_slice_sample0.get_series()
        series_slice_sample1 = extractor_slice_sample1.get_series()

        # Verify that the full extractor has frames_per_slice times more samples
        frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(file_path)
        assert extractor_full.get_num_samples() == extractor_slice_sample0.get_num_samples() * frames_per_slice

        # Verify that the data is interleaved correctly
        # The pattern should be: slice_sample0[0], slice_sample1[0], slice_sample0[1], slice_sample1[1], ...
        assert np.allclose(full_series[0], series_slice_sample0[0])
        assert np.allclose(full_series[1], series_slice_sample1[0])
        assert np.allclose(full_series[2], series_slice_sample0[1])
        assert np.allclose(full_series[3], series_slice_sample1[1])
        assert np.allclose(full_series[4], series_slice_sample0[2])
        assert np.allclose(full_series[5], series_slice_sample1[2])
        # These are all the samples here

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
        extractor_full = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 1",
            interleave_slice_samples=True,
        )
        extractor_slice_sample0 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 1",
            slice_sample=0,
        )
        extractor_slice_sample1 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 1",
            slice_sample=1,
        )

        # Verify that the full extractor has frames_per_slice times more samples
        frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(file_path)
        assert extractor_full.get_num_samples() == extractor_slice_sample0.get_num_samples() * frames_per_slice

        # Get data from all extractors
        full_series = extractor_full.get_series()
        series_slice_sample0 = extractor_slice_sample0.get_series()
        series_slice_sample1 = extractor_slice_sample1.get_series()

        # Verify that the data is interleaved correctly
        # The pattern should be: slice_sample0[0], slice_sample1[0], slice_sample0[1], slice_sample1[1], ...
        assert np.allclose(full_series[0], series_slice_sample0[0])
        assert np.allclose(full_series[1], series_slice_sample1[0])
        assert np.allclose(full_series[2], series_slice_sample0[1])
        assert np.allclose(full_series[3], series_slice_sample1[1])
        assert np.allclose(full_series[4], series_slice_sample0[2])
        assert np.allclose(full_series[5], series_slice_sample1[2])
        # These are all the samples here


class TestScanImageVolumetricPlaneSlicing:
    """Test the plane_index parameter of the ScanImage extractor classes."""

    def test_plane_index_parameter_single_channel(self):
        """Test the plane_index parameter with volumetric single channel data.

        File: vol_one_ch_single_files_00002_00001.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 1600

        This test verifies that the plane_index parameter correctly extracts a specific plane
        from volumetric data and behaves as if the data is not volumetric. It tests:
        1. Extracting the first plane (plane_index=0)
        2. Extracting a middle plane (plane_index=4)
        3. Extracting the last plane (plane_index=8)
        4. Comparing the extracted planes with the corresponding planes in the full volumetric data
        """
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"

        # Create an extractor with plane_index=0 (first plane)
        extractor_plane0 = ScanImageImagingExtractor(file_path=file_path, plane_index=0)

        # Create an extractor with plane_index=4 (middle plane)
        extractor_plane4 = ScanImageImagingExtractor(file_path=file_path, plane_index=4)

        # Create an extractor with plane_index=8 (last plane)
        extractor_plane8 = ScanImageImagingExtractor(file_path=file_path, plane_index=8)

        # Create a regular extractor for comparison
        extractor_full = ScanImageImagingExtractor(file_path=file_path)

        # Verify that the extractors with plane_index are not volumetric
        assert extractor_plane0.is_volumetric == False, "Extractor with plane_index should not be volumetric"
        assert extractor_plane4.is_volumetric == False, "Extractor with plane_index should not be volumetric"
        assert extractor_plane8.is_volumetric == False, "Extractor with plane_index should not be volumetric"

        # Verify that the full extractor is volumetric
        assert extractor_full.is_volumetric == True, "Full extractor should be volumetric"

        # Verify that the extractors with plane_index have num_planes=1
        assert extractor_plane0.get_num_planes() == 1, "Extractor with plane_index should have num_planes=1"
        assert extractor_plane4.get_num_planes() == 1, "Extractor with plane_index should have num_planes=1"
        assert extractor_plane8.get_num_planes() == 1, "Extractor with plane_index should have num_planes=1"

        # Verify that the full extractor has num_planes=9
        assert extractor_full.get_num_planes() == 9, "Full extractor should have num_planes=9"

        # Get data from all extractors
        data_plane0 = extractor_plane0.get_series()
        data_plane4 = extractor_plane4.get_series()
        data_plane8 = extractor_plane8.get_series()
        data_full = extractor_full.get_series()

        # Predefine expected shapes for easier readability
        expected_shape_plane = (extractor_plane0.get_num_samples(), 20, 20)
        expected_shape_full = (extractor_full.get_num_samples(), 20, 20, 9)

        # Verify the shape of the data
        assert data_plane0.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_plane4.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_plane8.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_full.shape == expected_shape_full, f"Data shape should be {expected_shape_full}"

        # Verify that the data from the plane extractors matches the corresponding plane in the full extractor
        # Use assert_allclose instead of assert_array_equal to allow for small differences
        # Compare the full series at once
        np.testing.assert_allclose(
            data_plane0,
            data_full[:, :, :, 0],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Data from plane 0 extractor does not match plane 0 in full extractor",
        )
        np.testing.assert_allclose(
            data_plane4,
            data_full[:, :, :, 4],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Data from plane 4 extractor does not match plane 4 in full extractor",
        )
        np.testing.assert_allclose(
            data_plane8,
            data_full[:, :, :, 8],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Data from plane 8 extractor does not match plane 8 in full extractor",
        )

    def test_plane_index_with_multiple_frames_per_slice(self):
        """Test the plane_index parameter with slice_sample for multi-channel volumetric data.

        File: scanimage_20220923_noroi.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (2 planes)
        - Channels: Multiple Channels ['Channel 1', 'Channel 4']
        - Frames per slice: 2
        - Frame rate: 29.1248 Hz
        - Volume rate: 7.28119 Hz
        - Image shape: 256 x 256
        - IFDS/Pages: 24

        This test verifies that the plane_index parameter works correctly when combined with
        the slice_sample parameter for volumetric data with multiple frames per slice. It tests:
        1. Extracting plane 0 with slice_sample=0
        2. Extracting plane 1 with slice_sample=0
        3. Extracting plane 0 with slice_sample=1
        4. Extracting plane 1 with slice_sample=1
        5. Comparing the extracted data with the corresponding planes in the regular extractors
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Create extractors with different combinations of slice_sample and plane_index
        extractor_slice_sample_0_plane_0 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=0,
            plane_index=0,
        )

        extractor_slice_sample_0_plane_1 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=0,
            plane_index=1,
        )

        extractor_slice_sample_1_plane_0 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=1,
            plane_index=0,
        )

        extractor_slice_sample_1_plane_1 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=1,
            plane_index=1,
        )

        # Create regular extractors for comparison
        extractor_slice_sample_0 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=0,
        )

        extractor_slice_sample_1 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=1,
        )
        # Define the error message for easier readability
        error_message = "Extractor with plane_index should not be volumetric"
        # Verify that the extractors with plane_index are not volumetric
        assert extractor_slice_sample_0_plane_0.is_volumetric == False, error_message
        assert extractor_slice_sample_0_plane_1.is_volumetric == False, error_message
        assert extractor_slice_sample_1_plane_0.is_volumetric == False, error_message
        assert extractor_slice_sample_1_plane_1.is_volumetric == False, error_message

        # Verify that the regular extractors are volumetric
        assert extractor_slice_sample_0.is_volumetric == True, "Regular extractor should be volumetric"
        assert extractor_slice_sample_1.is_volumetric == True, "Regular extractor should be volumetric"

        # Verify that the extractors with plane_index have num_planes=1
        error_message = "Extractor with plane_index should have num_planes=1"
        assert extractor_slice_sample_0_plane_0.get_num_planes() == 1, error_message
        assert extractor_slice_sample_0_plane_1.get_num_planes() == 1, error_message
        assert extractor_slice_sample_1_plane_0.get_num_planes() == 1, error_message
        assert extractor_slice_sample_1_plane_1.get_num_planes() == 1, error_message

        # Verify that the regular extractors have num_planes=2
        assert extractor_slice_sample_0.get_num_planes() == 2, "Regular extractor should have num_planes=2"
        assert extractor_slice_sample_1.get_num_planes() == 2, "Regular extractor should have num_planes=2"

        # Get data from all extractors
        data_slice_sample_0_plane_0 = extractor_slice_sample_0_plane_0.get_series()
        data_slice_sample_0_plane_1 = extractor_slice_sample_0_plane_1.get_series()
        data_slice_sample_1_plane_0 = extractor_slice_sample_1_plane_0.get_series()
        data_slice_sample_1_plane_1 = extractor_slice_sample_1_plane_1.get_series()
        data_slice_sample_0 = extractor_slice_sample_0.get_series()
        data_slice_sample_1 = extractor_slice_sample_1.get_series()

        # Predefine expected shapes for easier readability
        expected_shape_plane = (extractor_slice_sample_0_plane_0.get_num_samples(), 256, 256)
        expected_shape_full = (extractor_slice_sample_0.get_num_samples(), 256, 256, 2)

        # Verify the shape of the data
        assert data_slice_sample_0_plane_0.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_slice_sample_0_plane_1.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_slice_sample_1_plane_0.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_slice_sample_1_plane_1.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_slice_sample_0.shape == expected_shape_full, f"Data shape should be {expected_shape_full}"
        assert data_slice_sample_1.shape == expected_shape_full, f"Data shape should be {expected_shape_full}"

        # Verify that the data from the plane extractors matches the corresponding plane in the regular extractors
        for sample_index in range(
            min(extractor_slice_sample_0_plane_0.get_num_samples(), extractor_slice_sample_0.get_num_samples())
        ):
            np.testing.assert_array_equal(
                data_slice_sample_0_plane_0[sample_index],
                data_slice_sample_0[sample_index, :, :, 0],
                f"Data from slice_sample_0_plane_0 extractor does not match plane 0 in slice_sample_0 extractor for sample {sample_index}",
            )
            np.testing.assert_array_equal(
                data_slice_sample_0_plane_1[sample_index],
                data_slice_sample_0[sample_index, :, :, 1],
                f"Data from slice_sample_0_plane_1 extractor does not match plane 1 in slice_sample_0 extractor for sample {sample_index}",
            )

        for sample_index in range(
            min(extractor_slice_sample_1_plane_0.get_num_samples(), extractor_slice_sample_1.get_num_samples())
        ):
            np.testing.assert_array_equal(
                data_slice_sample_1_plane_0[sample_index],
                data_slice_sample_1[sample_index, :, :, 0],
                f"Data from slice_sample_1_plane_0 extractor does not match plane 0 in slice_sample_1 extractor for sample {sample_index}",
            )
            np.testing.assert_array_equal(
                data_slice_sample_1_plane_1[sample_index],
                data_slice_sample_1[sample_index, :, :, 1],
                f"Data from slice_sample_1_plane_1 extractor does not match plane 1 in slice_sample_1 extractor for sample {sample_index}",
            )


class TestTimestampExtraction:
    """Test the get_times method for ScanImageImagingExtractor class."""

    def test_get_times_planar_multichannel(self):
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
        assert (
            len(timestamps_ch1) == extractor_ch1.get_num_samples()
        ), "Should have one timestamp per frame for channel 1"
        assert (
            len(timestamps_ch2) == extractor_ch2.get_num_samples()
        ), "Should have one timestamp per frame for channel 2"

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

    def test_get_times_volumetric_data(self):
        """Test get_times with volumetric data (with and without plane_index)

        This test verifies that:
        1. For volumetric data:
            - The correct number of timestamps are returned (equal to number of samples)
            - Timestamps are monotonically increasing
            - Timestamps match the expected values from the TIFF metadata

        2. For plane-sliced data:
            - The get_times method works correctly when accessing a specific plane using plane_index
            - When plane_index is used, timestamps reflect the timestamps of the frames at that plane

        File: vol_no_flyback_00001_00001.tif
        """

        # Test file path - volumetric data without flyback frames
        file_path = (
            SCANIMAGE_PATH / "volumetric_single_channel_single_file_no_flyback" / "vol_no_flyback_00001_00001.tif"
        )

        # Get raw timestamps for reference
        with TiffReader(file_path) as tiff:
            raw_timestamps = [ScanImageImagingExtractor.extract_timestamp_from_page(page) for page in tiff.pages]
        raw_timestamps = np.asarray(raw_timestamps)

        # Create the extractor and get timestamps
        volumetric_extractor = ScanImageImagingExtractor(file_path=file_path)
        volumetric_timestamps = volumetric_extractor.get_times()

        # Basic validation
        num_samples = volumetric_extractor.get_num_samples()
        assert len(volumetric_timestamps) == num_samples, "Number of timestamps should match number of samples"
        assert np.all(np.diff(volumetric_timestamps) > 0), "Timestamps should be monotonically increasing"

        # Calculate expected timestamps for volumetric data
        # For volumetric data without flyback frames, we use the last plane of each volume
        num_planes = volumetric_extractor.get_num_planes()
        sample_indices = np.arange(num_samples, dtype=int)

        # Each volume has num_planes frames, and we want the last frame of each volume
        last_plane_offset = num_planes - 1
        expected_indices = sample_indices * num_planes + last_plane_offset
        expected_timestamps = raw_timestamps[expected_indices]

        # Verify timestamps match expected values
        np.testing.assert_array_equal(
            volumetric_timestamps,
            expected_timestamps,
            "Volumetric timestamps should match expected values from TIFF metadata",
        )

        # Test that plane slicing timestamps are extracted correctly
        plane_indices = [0, 4, 8]
        # Calculate expected timestamps for each plane first plane (0), middle plane (4), and last plane (8)
        for plane_idx in plane_indices:
            plane_timestamps = ScanImageImagingExtractor(file_path=file_path, plane_index=plane_idx).get_times()
            assert len(plane_timestamps) == num_samples, "Number of timestamps should match number of samples"

            # For each plane, we need to select frames at positions: plane_index, plane_index + num_planes, ...
            plane_frame_indices = sample_indices * num_planes + plane_idx
            expected_plane_timestamps = raw_timestamps[plane_frame_indices]

            # Verify timestamps match expected values
            np.testing.assert_array_equal(
                plane_timestamps,
                expected_plane_timestamps,
                f"Timestamps for plane_index={plane_idx} should match expected values",
            )

    def test_get_times_volumetric_with_flyback_frames(self):
        """Test get_times with volumetric data with flyback frames (with and without plane_index)

        This test verifies that:
        1. For volumetric data with flyback frames:
            - The correct number of timestamps are returned (equal to number of samples)
            - Flyback frames are properly excluded from timestamps
            - Timestamps match the expected values from the TIFF metadata

        2. For plane-sliced data with flyback frames:
            - The get_times method works correctly when accessing a specific plane using plane_index
            - When plane_index is used, timestamps reflect the timestamps of the frames at that plane

        File: vol_one_ch_single_files_00002_00001.tif

        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 1600
        """

        # Test file path - volumetric data with flyback frames
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"

        # Extract raw timestamps directly from TIFF file
        with TiffReader(file_path) as tiff:
            raw_timestamps = [ScanImageImagingExtractor.extract_timestamp_from_page(page) for page in tiff.pages]
        raw_timestamps = np.array(raw_timestamps)

        # Create the extractor and get timestamps
        volumetric_extractor = ScanImageImagingExtractor(file_path=file_path)
        volumetric_timestamps = volumetric_extractor.get_times()

        # Basic validation
        num_samples = volumetric_extractor.get_num_samples()
        assert len(volumetric_timestamps) == num_samples, "Number of timestamps should match number of samples"
        assert np.all(np.diff(volumetric_timestamps) > 0), "Timestamps should be monotonically increasing"

        # Calculate frame cycle parameters for volumetric data with flyback frames
        num_planes = volumetric_extractor.get_num_planes()
        num_flyback_frames = volumetric_extractor.num_flyback_frames_per_channel
        frames_per_slice = volumetric_extractor._frames_per_slice
        num_channels = len(volumetric_extractor.channel_names)
        channel_index = volumetric_extractor._channel_index

        # Calculate derived parameters
        image_frames_per_cycle = num_planes * frames_per_slice * num_channels
        flyback_frames_total = num_flyback_frames * num_channels
        total_frames_per_cycle = image_frames_per_cycle + flyback_frames_total

        # Calculate expected timestamps for volumetric data
        # For volumetric data with flyback frames, we use the last frame before flyback frames
        sample_indices = np.arange(num_samples, dtype=int)
        last_frame_offset = image_frames_per_cycle - num_channels

        # Calculate indices for expected timestamps
        timestamp_indices = sample_indices * total_frames_per_cycle + last_frame_offset + channel_index
        expected_timestamps = raw_timestamps[timestamp_indices]

        # Verify timestamps match expected values
        np.testing.assert_array_equal(
            volumetric_timestamps,
            expected_timestamps,
            "Volumetric timestamps should match expected values from TIFF metadata",
        )

        # Create extractors with plane_index
        plane_index_0 = 0
        plane_index_4 = 4

        plane0_extractor = ScanImageImagingExtractor(file_path=file_path, plane_index=plane_index_0)
        plane4_extractor = ScanImageImagingExtractor(file_path=file_path, plane_index=plane_index_4)

        # Get timestamps from plane extractors
        plane0_timestamps = plane0_extractor.get_times()
        plane4_timestamps = plane4_extractor.get_times()

        # Verify all extractors have the same number of samples and timestamps
        assert len(plane0_timestamps) == num_samples, "Plane 0 timestamps count should match number of samples"
        assert len(plane4_timestamps) == num_samples, "Plane 4 timestamps count should match number of samples"

        # Calculate expected timestamps for plane 0
        # For plane-sliced data, we use the frame at the specified plane index
        plane0_offset = plane_index_0
        plane0_timestamp_indices = sample_indices * total_frames_per_cycle + plane0_offset + channel_index
        expected_plane0_timestamps = raw_timestamps[plane0_timestamp_indices]

        # Calculate expected timestamps for plane 4
        plane4_offset = plane_index_4
        plane4_timestamp_indices = sample_indices * total_frames_per_cycle + plane4_offset + channel_index
        expected_plane4_timestamps = raw_timestamps[plane4_timestamp_indices]

        # Verify plane0 extractor timestamps match expected values
        np.testing.assert_array_equal(
            plane0_timestamps,
            expected_plane0_timestamps,
            "Timestamps for plane_index=0 should match expected values",
        )

        # Verify plane4 extractor timestamps match expected values
        np.testing.assert_array_equal(
            plane4_timestamps,
            expected_plane4_timestamps,
            "Timestamps for plane_index=4 should match expected values",
        )

    def test_get_times_multi_samples_per_slice_data(self):
        """Test get_times with multiple samples per slice in volumetric data (with and without plane_index)

        This test verifies that:
        1. For volumetric data with multiple samples per slice:
            - The correct timestamps are returned when slice_sample is specified
            - Timestamps match the expected values from the TIFF metadata

        2. For plane-sliced data with multiple samples per slice:
            - The get_times method works correctly when using both slice_sample and plane_index
            - When plane_index is used, timestamps reflect the timestamps of the frames at that plane

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

        # Part 1: Test with regular volumetric extractor
        # Create extractor with slice_sample=0
        slice_sample = 0
        extractor = ScanImageImagingExtractor(
            file_paths=[file_path], channel_name="Channel 4", slice_sample=slice_sample
        )

        # Get timestamps from extractor
        timestamps = extractor.get_times()

        # Check that number of timestamps equals number of samples
        assert len(timestamps) == extractor.get_num_samples(), "Number of timestamps should match number of samples"

        # Verify monotonically increasing (if there's more than one timestamp)
        if len(timestamps) > 1:
            assert np.all(np.diff(timestamps) > 0), "Timestamps should be monotonically increasing"

        # Extract raw timestamps directly from TIFF file using list comprehension
        with TiffReader(file_path) as tiff:
            raw_timestamps = [ScanImageImagingExtractor.extract_timestamp_from_page(page) for page in tiff.pages]
        raw_timestamps = np.array(raw_timestamps)

        # For the flyback frames case, the timestamp assigned is the
        # last timestamp before the flyback frames
        num_planes = extractor.get_num_planes()
        num_flyback_frames_per_channel = extractor.num_flyback_frames_per_channel
        num_samples_volumetric = extractor.get_num_samples()

        num_frames_per_slice = extractor._frames_per_slice
        num_channels = len(extractor.channel_names)
        image_frames_per_cycle = num_planes * num_frames_per_slice * num_channels
        flyback_frames = num_flyback_frames_per_channel * num_channels
        total_frames_per_cycle = image_frames_per_cycle + flyback_frames

        # As a timestamp we get every last frame before the flyback frames
        sample_indices = np.arange(0, num_samples_volumetric, dtype=int)
        last_volume_offset = image_frames_per_cycle - num_channels * num_frames_per_slice
        channel_offset = extractor._channel_index
        slice_offset = slice_sample * num_channels

        timestamp_indices = sample_indices * total_frames_per_cycle + last_volume_offset + channel_offset + slice_offset
        expected_timestamps = raw_timestamps[timestamp_indices]

        # Compare the timestamps with expected values
        np.testing.assert_array_equal(
            timestamps,
            expected_timestamps,
            "Timestamps from get_times should match expected values extracted from TIFF metadata",
        )

        # Part 2: Test with plane_index
        # Create extractors with and without plane_index
        slice_sample = 1
        plane_index = 1
        volumetric_extractor = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=slice_sample,
        )
        planarized_extractor = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=slice_sample,
            plane_index=plane_index,
        )

        # Get timestamps
        timestamps_full = volumetric_extractor.get_times()
        planarized_timestamps = planarized_extractor.get_times()

        # Check that number of timestamps equals number of samples for each extractor
        assert len(timestamps_full) == volumetric_extractor.get_num_samples()
        assert len(planarized_timestamps) == planarized_extractor.get_num_samples()

        # Should have the same number of samples and timestamps
        assert volumetric_extractor.get_num_samples() == planarized_extractor.get_num_samples()

        # Calculate expected timestamps for the planarized extractor
        num_planes = volumetric_extractor.get_num_planes()
        num_flyback_frames_per_channel = volumetric_extractor.num_flyback_frames_per_channel
        num_samples_volumetric = volumetric_extractor.get_num_samples()

        num_frames_per_slice = volumetric_extractor._frames_per_slice
        num_channels = len(volumetric_extractor.channel_names)
        image_frames_per_cycle = num_planes * num_frames_per_slice * num_channels
        flyback_frames = num_flyback_frames_per_channel * num_channels
        total_frames_per_cycle = image_frames_per_cycle + flyback_frames

        # As a timestamp we get every last frame before the flyback frames
        sample_indices = np.arange(0, num_samples_volumetric, dtype=int)
        plane_slice_offset = plane_index * num_channels * num_frames_per_slice
        frames_per_slice_offset = slice_sample * num_channels
        channel_offset = planarized_extractor._channel_index

        timestamp_indices = (
            sample_indices * total_frames_per_cycle + plane_slice_offset + frames_per_slice_offset + channel_offset
        )
        expected_planarized_timestamps = raw_timestamps[timestamp_indices]

        # Verify planarized extractor timestamps
        np.testing.assert_array_equal(
            planarized_timestamps,
            expected_planarized_timestamps,
            "Timestamps for plane 1 should match expected values extracted from TIFF metadata",
        )


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


def test_missing_file_detection(tmp_path):
    """Test that a warning is thrown when a file in the middle of a sequence is missing.

    This test creates a temporary directory with copies of files 1 and 3,
    but not file 2, and verifies that a warning is thrown when the extractor is
    initialized with the first file.
    """

    # Create copies of files 1 and 3 (skip file 2)
    source_files = ["scanimage_20240320_multifile_00001.tif", "scanimage_20240320_multifile_00003.tif"]

    # Copy the files, resolving any symlinks to avoid problems with git-annex
    for file_name in source_files:
        source_path = (SCANIMAGE_PATH / file_name).resolve()
        dest_path = tmp_path / file_name
        shutil.copy(source_path, dest_path)

        # Verify the file was copied correctly
        assert dest_path.exists(), f"File {dest_path} was not copied correctly"
        assert dest_path.stat().st_size > 0, f"File {dest_path} is empty"

    #  Check the copies were created correctly
    all_files = list(tmp_path.glob("*.tif"))
    assert len(all_files) == 2, f"Expected 2 files in directory, found {len(all_files)}: {[f.name for f in all_files]}"

    # Initialize extractor with first file and check for warning
    with pytest.warns(UserWarning, match="Missing files detected.*00002.tif"):
        extractor = ScanImageImagingExtractor(file_path=tmp_path / source_files[0], channel_name="Channel 1")

        # Verify that the extractor still works with the available files
        assert len(extractor.file_paths) == 2
        assert all(Path(fp).name in source_files for fp in extractor.file_paths)


def test_non_integer_file_warning(tmp_path):
    """Test that a warning is thrown when a file with a non-integer index is found.

    This test creates a temporary directory with copies of all normal files (1, 2, 3)
    plus a file with a non-integer index, and verifies that a warning is thrown
    about the non-integer file.
    """
    # Create copies of all normal files (1, 2, 3)
    source_files = [
        "scanimage_20240320_multifile_00001.tif",
        "scanimage_20240320_multifile_00002.tif",
        "scanimage_20240320_multifile_00003.tif",
    ]

    # Copy the files, resolving any symlinks to ensure cross-platform compatibility
    for file_name in source_files:
        source_path = (SCANIMAGE_PATH / file_name).resolve()
        dest_path = tmp_path / file_name
        shutil.copy(source_path, dest_path)

        # Verify the file was copied correctly
        assert dest_path.exists(), f"File {dest_path} was not copied correctly"
        assert dest_path.stat().st_size > 0, f"File {dest_path} is empty"

    # Create a file that does not follow the integer sequence
    non_integer_file = "scanimage_20240320_multifile_abc.tif"
    non_integer_path = tmp_path / non_integer_file
    shutil.copy((SCANIMAGE_PATH / source_files[0]).resolve(), non_integer_path)

    # Verify the the copy was created correctly
    assert non_integer_path.exists(), f"Non-integer file {non_integer_path} was not created correctly"
    assert non_integer_path.stat().st_size > 0, f"Non-integer file {non_integer_path} is empty"

    # List all files in the directory to confirm setup
    all_files = list(tmp_path.glob("*.tif"))
    assert len(all_files) == 4, f"Expected 4 files in directory, found {len(all_files)}: {[f.name for f in all_files]}"

    # Initialize extractor with first file and check for warning about non-sequence files
    with pytest.warns(UserWarning, match="Non-sequence files detected"):
        extractor = ScanImageImagingExtractor(file_path=tmp_path / source_files[0], channel_name="Channel 1")

        # Verify that the extractor still works with the available files (only integer files)
        assert len(extractor.file_paths) == 3
        assert all(Path(fp).name in source_files for fp in extractor.file_paths), "Unexpected files in extractor"


def test_get_frames_per_slice():
    """
    Test the static get_frames_per_slice method.

    This test verifies that the static get_frames_per_slice method correctly extracts
    the number of slices per sample from ScanImage TIFF files without needing to create
    an extractor instance.
    """
    # Test with single frame per slice file
    single_frame_file = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
    frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(single_frame_file)
    assert frames_per_slice == 10, "File should have 1 frame per slice"

    # Test with multiple frames per slice file
    multi_frame_file = SCANIMAGE_PATH / "scanimage_20220801_volume.tif"
    frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(multi_frame_file)
    assert frames_per_slice == 8, "File should have 8 frames per slice"

    # Test with another multiple frames per slice file
    multi_frame_file2 = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
    frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(multi_frame_file2)
    assert frames_per_slice == 2, "File should have 2 frames per slice"


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


class TestGetOriginalFrameIndices:
    """Test the get_original_frame_indices method for ScanImageImagingExtractor class."""

    def test_planar_single_channel_single_file(self):
        """Test get_original_frame_indices with planar single channel data.

        File: scanimage_20220801_single.tif
        Metadata:
        - Volumetric: False (single plane)
        - Channels: 1
        - Frames: 3
        - Frame rate: 15.2379 Hz
        - Image shape: 1024 x 1024

        This test verifies that:
        1. The method returns the correct number of frame indices
        2. Frame indices are of the correct dtype (int64)
        3. Frame indices are within the valid range
        4. For planar data, frame indices should be sequential (0, 1, 2, ...)
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220801_single.tif"
        extractor = ScanImageImagingExtractor(file_path=str(file_path))

        # Get original frame indices
        frame_indices = extractor.get_original_frame_indices()

        assert len(frame_indices) == extractor.get_num_samples(), "Should have one index per sample"

        # For planar single-channel data, frame indices should be sequential
        expected_indices = np.arange(extractor.get_num_samples(), dtype=np.int64)
        np.testing.assert_array_equal(frame_indices, expected_indices)

    def test_planar_multi_channel_single_file(self):
        """Test get_original_frame_indices with planar multi-channel data.

        File: planar_two_ch_single_files_00001_00001.tif
        Metadata:
        - Volumetric: False
        - Channels: 2 (Channel 1 and Channel 2)
        - Frames per channel: 1000
        - Total frames: 2000
        - Frame rate: 523.926 Hz

        This test verifies that:
        1. Frame indices correctly account for channel interleaving
        2. Different channels have different frame index patterns
        3. Frame indices map to the correct channel data
        """
        file_path = SCANIMAGE_PATH / "planar_two_channels_single_file" / "planar_two_ch_single_files_00001_00001.tif"

        # Test with Channel 1
        extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")
        frame_indices_ch1 = extractor_ch1.get_original_frame_indices()

        # Test with Channel 2
        extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")
        frame_indices_ch2 = extractor_ch2.get_original_frame_indices()

        # Basic validation
        assert len(frame_indices_ch1) == extractor_ch1.get_num_samples()
        assert len(frame_indices_ch2) == extractor_ch2.get_num_samples()

        # For multi-channel planar data, channels are interleaved
        # Channel 1 should have indices: 0, 2, 4, 6, ...
        # Channel 2 should have indices: 1, 3, 5, 7, ...
        expected_ch1_indices = np.arange(0, 2 * extractor_ch1.get_num_samples(), 2, dtype=np.int64)
        expected_ch2_indices = np.arange(1, 2 * extractor_ch2.get_num_samples(), 2, dtype=np.int64)

        np.testing.assert_array_equal(
            frame_indices_ch1, expected_ch1_indices, "Channel 1 should have even frame indices"
        )
        np.testing.assert_array_equal(
            frame_indices_ch2, expected_ch2_indices, "Channel 2 should have odd frame indices"
        )

    def test_planar_multi_channel_multi_file(self):
        """Test get_original_frame_indices with multi-file planar data.

        Files: scanimage_20240320_multifile_0000[1-3].tif
        Metadata:
        - Volumetric: False
        - Channels: 2
        - Files: 3
        - Frames per file: 20 (10 per channel)
        - Total samples per channel: 30

        This test verifies that:
        1. Frame indices correctly account for file boundaries
        2. File offsets are properly calculated
        3. Frame indices span across multiple files correctly
        """
        file_path = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
        extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")

        frame_indices = extractor.get_original_frame_indices()

        # Basic validation
        assert len(frame_indices) == extractor.get_num_samples()

        # Verify that frame indices span the expected range across all files
        total_frames = sum(extractor._ifds_per_file)
        assert np.all(frame_indices >= 0), "All frame indices should be non-negative"
        assert np.all(frame_indices < total_frames), "All frame indices should be within total dataset bounds"

        # For multi-file data with Channel 1, indices should be: 0, 2, 4, ..., 18, 20, 22, 24, ..., 38, 40, 42, ...
        # This is basically testing that they don't repeat when spanning multiple files
        expected_indices = np.arange(0, 2 * extractor.get_num_samples(), 2, dtype=np.int64)
        np.testing.assert_array_equal(
            frame_indices, expected_indices, "Frame indices should account for channel interleaving across files"
        )

    def test_volumetric_single_channel_no_flyback(self):
        """Test get_original_frame_indices with volumetric data without flyback frames.

        File: vol_no_flyback_00001_00001.tif
        Metadata:
        - Volumetric: True (9 planes)
        - Channels: 1
        - Samples: 10
        - No flyback frames
        - Total frames: 90

        This test verifies that:
        1. Default behavior uses the last plane of each volume
        2. Frame indices correctly map to volume boundaries
        3. No flyback frame handling is needed
        """
        file_path = (
            SCANIMAGE_PATH / "volumetric_single_channel_single_file_no_flyback" / "vol_no_flyback_00001_00001.tif"
        )
        extractor = ScanImageImagingExtractor(file_path=file_path)

        frame_indices = extractor.get_original_frame_indices()

        # Basic validation
        assert len(frame_indices) == extractor.get_num_samples()

        # For volumetric data without flyback frames, the method should use the last plane of each volume
        # With 9 planes per volume, the last plane indices should be: 8, 17, 26, 35, 44, 53, 62, 71, 80, 89
        num_planes = extractor.get_num_planes()
        expected_indices = np.arange(
            num_planes - 1, num_planes * extractor.get_num_samples(), num_planes, dtype=np.int64
        )

        np.testing.assert_array_equal(
            frame_indices, expected_indices, "Should use last plane of each volume for volumetric data"
        )

    def test_volumetric_single_channel_with_flyback(self):
        """Test get_original_frame_indices with volumetric data with flyback frames.

        File: vol_one_ch_single_files_00002_00001.tif
        Metadata:
        - Volumetric: True (9 planes + 7 flyback frames)
        - Channels: 1
        - Total frames per cycle: 16
        - Flyback frames are excluded from the mapping

        This test verifies that:
        1. Flyback frames are properly excluded from frame index calculation
        2. Frame indices correctly map to the last imaging plane before flyback
        3. File offsets are correctly calculated
        """
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"
        extractor = ScanImageImagingExtractor(file_path=file_path)

        frame_indices = extractor.get_original_frame_indices()

        # Basic validation
        assert len(frame_indices) == extractor.get_num_samples()
        assert frame_indices.dtype == np.int64

        # Calculate expected frame indices accounting for flyback frames
        num_planes = extractor.get_num_planes()
        num_flyback = extractor.num_flyback_frames_per_channel
        total_frames_per_cycle = num_planes + num_flyback

        # The last imaging frame in each cycle should be at position (num_planes - 1) within each cycle
        expected_indices = []
        for sample_idx in range(extractor.get_num_samples()):
            cycle_start = sample_idx * total_frames_per_cycle
            last_imaging_frame = cycle_start + num_planes - 1  # Note this is the last frame before the flyback frames
            expected_indices.append(last_imaging_frame)

        expected_indices = np.array(expected_indices, dtype=np.int64)
        np.testing.assert_array_equal(
            frame_indices, expected_indices, "Should correctly handle flyback frames in frame index calculation"
        )

    def test_plane_index_parameter(self):
        """Test get_original_frame_indices with plane_index parameter.

        This test verifies that:
        1. When plane_index is specified, it uses that plane instead of the default last plane
        2. Different plane_index values produce different frame indices
        3. The plane_index parameter is correctly validated
        """
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"

        # Test with plane_index=0 (first plane)
        full_extractor = ScanImageImagingExtractor(file_path=file_path)
        indices_of_first_plane = full_extractor.get_original_frame_indices(plane_index=0)

        extractor_first_plane = ScanImageImagingExtractor(file_path=file_path, plane_index=0)
        indices_of_first_plane_extractor = extractor_first_plane.get_original_frame_indices()

        np.testing.assert_array_equal(
            indices_of_first_plane,
            indices_of_first_plane_extractor,
            "Indices for plane_index=0 should match when using full extractor and plane extractor",
        )

    def test_slice_sample_parameter(self):
        """Test get_original_frame_indices with slice_sample parameter.

        File: scanimage_20220923_noroi.tif
        Dataset characteristics:
        - Volumetric: True (2 planes)
        - Channels: 2 (['Channel 1', 'Channel 4'])
        - Frames per slice: 2
        - Total frames: 24
        - Data layout: [ch1_plane1_slice1, ch2_plane1_slice1, ch1_plane1_slice2, ch2_plane1_slice2,
                        ch1_plane2_slice1, ch2_plane2_slice1, ch1_plane2_slice2, ch2_plane2_slice2, ...]

        This test verifies that the method returns the correct frame indices
        for a specific slice_sample, accounting for the data interleaving pattern.
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Test with slice_sample=0, Channel 4
        extractor = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=0)
        frame_indices = extractor.get_original_frame_indices()

        # Calculate expected frame indices manually based on data structure
        num_channels = 2
        num_planes = extractor.get_num_planes()  # Should be 2
        frames_per_slice = 2
        channel_index = extractor._channel_index  # Channel 4 should be index 1

        # For slice_sample=0, we want the frame for slice_sample=0 of the last plane for each sample
        expected_indices = []
        for sample_index in range(extractor.get_num_samples()):
            # Each sample corresponds to a volume
            # For the last plane (plane 1, 0-indexed), slice_sample 0, channel 4 (index 1)
            base_frame = sample_index * (num_planes * frames_per_slice * num_channels)
            last_plane_offset = (num_planes - 1) * frames_per_slice * num_channels
            slice_sample_offset = 0 * num_channels  # slice_sample=0
            channel_offset = channel_index

            frame_index = base_frame + last_plane_offset + slice_sample_offset + channel_offset
            expected_indices.append(frame_index)

        expected_indices = np.array(expected_indices, dtype=np.int64)

        # Verify that the method returns the expected indices
        np.testing.assert_array_equal(
            frame_indices, expected_indices, "Frame indices should match manually calculated expected values"
        )
